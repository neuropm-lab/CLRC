"""Optuna hyperparameter optimization for LOBO XGBoost.

In addition to the param space / LOBO evaluation loop, this module exposes
two infrastructure helpers that support resumable HPO:

``build_journal_storage``
    Construct an Optuna :class:`JournalStorage` backed by a
    :class:`JournalFileBackend` on a user-controlled path.  A study created
    against this storage persists trial state to disk after every event, so
    the study can be resumed from the same path after a process restart.

``cached_fold_split``
    Disk cache for deterministic per-fold train/test index splits, keyed by
    ``(seed, holdout_region, feature_mode)``.  Successive Optuna trials
    compute the same split; this removes the re-derivation cost.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import optuna
import xgboost as xgb
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from clrc.core.io import stable_hash_int
from clrc.core.metrics import aggregate_fold_metrics_for_loss, mae, rmse
from clrc.core.types import AlignmentData, FoldArtifact
from clrc.prediction.lobo import (
    LoboFoldSplit,
    compute_lobo_fold_split,
    infer_regions,
    iter_lobo_folds,
    precompute_fold_masks,
)
from clrc.prediction.xgboost import train_predict_xgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Resumability — JournalStorage
# ---------------------------------------------------------------------------

def build_journal_storage(storage_path: Path | str) -> JournalStorage:
    """Construct a JournalFileBackend-backed Optuna storage at ``storage_path``.

    The parent directory is created if absent.  Pass the returned storage
    into :func:`optuna.create_study(..., storage=..., load_if_exists=True)`.
    A study created this way persists every trial event to the journal log
    on disk, so reopening the same storage + study name after a restart
    recovers the full trial history.
    """
    path = Path(storage_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path)))


def default_journal_storage_path(
    output_dir: Path | str, study_name: str
) -> Path:
    """Canonical on-disk location for a study's journal log.

    We keep one file per study under ``<output_dir>/optuna_journal/<study>.log``
    so multiple HPO runs in the same output tree don't collide.
    """
    return Path(output_dir) / "optuna_journal" / f"{study_name}.log"


# ---------------------------------------------------------------------------
#  Fold-split cache
# ---------------------------------------------------------------------------

def _fold_cache_filename(
    *, seed: int, holdout_region: str, feature_mode: str
) -> str:
    """Deterministic cache filename for a (seed, region, mode) triple.

    The region name is embedded verbatim (with ``/`` sanitized) so cache
    files are human-identifiable on disk.
    """
    safe_region = str(holdout_region).replace("/", "_").replace(" ", "_")
    safe_mode = str(feature_mode).replace("/", "_")
    return f"seed{int(seed)}__{safe_region}__{safe_mode}.pkl"


def cached_fold_split(
    *,
    compute_fn: Callable[..., Optional[Mapping[str, Any]]],
    cache_dir: Path | str,
    seed: int,
    holdout_region: str,
    feature_mode: str,
    **compute_kwargs: Any,
) -> Optional[Mapping[str, Any]]:
    """Return a fold split, reading from / writing to a pickle cache.

    The cache key is ``(seed, holdout_region, feature_mode)``.  On cache
    miss, ``compute_fn(seed=..., holdout_region=..., feature_mode=...,
    **compute_kwargs)`` is invoked and its result is pickled.  Successive
    Optuna trials that ask for the same key receive the cached split
    without invoking ``compute_fn``.

    The return value is whatever ``compute_fn`` produces — this helper is
    intentionally agnostic about the split representation.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _fold_cache_filename(
        seed=seed, holdout_region=holdout_region, feature_mode=feature_mode
    )
    if cache_file.exists():
        with cache_file.open("rb") as fh:
            return pickle.load(fh)
    split = compute_fn(
        seed=seed,
        holdout_region=holdout_region,
        feature_mode=feature_mode,
        **compute_kwargs,
    )
    # Do NOT persist None: a skipped fold (empty train or test after
    # eps filtering) can become valid again if upstream data changes,
    # and a stale None cache would silently mask it.
    if split is None:
        return None
    tmp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        pickle.dump(split, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_file)
    return split


# ---------------------------------------------------------------------------
#  Param space
# ---------------------------------------------------------------------------

def make_xgb_gpu_param_space(trial: optuna.Trial, *, seed: int) -> Dict:
    """XGBoost core-training param space for Optuna trials."""
    params = {
        "eta": trial.suggest_float("eta", 1e-2, 3e-1, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-4, 10, log=True
        ),
        "gamma": trial.suggest_float("gamma", 1e-3, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
        "lambda": trial.suggest_float("reg_lambda", 5e-3, 5, log=True),
        "alpha": trial.suggest_float("reg_alpha", 5e-3, 5, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
        "seed": int(seed),
    }
    return params


# ---------------------------------------------------------------------------
#  Loss / metric helpers
# ---------------------------------------------------------------------------

def apply_loss_and_metrics(
    base_params: Dict,
    *,
    loss: str,
    huber_slope: Optional[float],
    eval_metrics: Sequence[str],
) -> Dict:
    """Set XGBoost objective and eval_metric based on loss choice."""
    params = dict(base_params)
    params["eval_metric"] = list(eval_metrics)
    if loss == "pseudohuber":
        params["objective"] = "reg:pseudohubererror"
        params["huber_slope"] = float(1.0 if huber_slope is None else huber_slope)
    elif loss == "mae":
        params["objective"] = "reg:absoluteerror"
        params.pop("huber_slope", None)
    else:
        params["objective"] = "reg:squarederror"
        params.pop("huber_slope", None)
    return params


def choose_fixed_region_subset(
    regions_all: Sequence[str], *, n: int, seed: int
) -> List[str]:
    """Deterministically pick *n* regions for HPO subset."""
    regions_all = list(regions_all)
    if n <= 0:
        raise ValueError("n must be positive")
    if n >= len(regions_all):
        return list(regions_all)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(regions_all), size=n, replace=False)
    return [regions_all[i] for i in idx]


# ---------------------------------------------------------------------------
#  LOBO evaluation over a subset (for HPO)
# ---------------------------------------------------------------------------

def _iter_cached_fold_tuples(
    *,
    X: np.ndarray,
    y_all: np.ndarray,
    fold_masks: Dict[str, np.ndarray],
    regions_use: Sequence[str],
    eps: float,
    y_transform: str,
    data_type: str,
    fold_cache_dir: Path,
    seed: int,
    feature_mode: str,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, Any, np.ndarray]]:
    """Yield fold tuples (same shape as iter_lobo_folds) using cached_fold_split.

    Each region's index/transform split is pickled under ``fold_cache_dir``
    keyed by ``(seed, holdout_region, feature_mode)``. X_train / X_test are
    sliced from the full ``X`` matrix via the cached indices on every call;
    we do not cache the (potentially large) ``X`` slices themselves.

    Hardcodes ``edge_table_test=None`` in the yielded tuple because
    ``eval_params_lobo`` does not consume it -- this adapter is therefore
    NOT suitable for full training drivers (e.g. train_xgboost.py) that
    rely on ``include_edge_tables=True``. Use ``iter_lobo_folds`` directly
    for those callers.
    """
    for region in regions_use:
        split = cached_fold_split(
            compute_fn=_fold_split_compute_fn,
            cache_dir=fold_cache_dir,
            seed=seed,
            holdout_region=region,
            feature_mode=feature_mode,
            y_all=y_all,
            fold_masks=fold_masks,
            eps=eps,
            y_transform=y_transform,
            data_type=data_type,
        )
        if split is None:
            # Skip-warning already emitted inside compute_lobo_fold_split.
            continue
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        yield (
            region,
            X[train_idx, :],
            split["y_train_t"],
            X[test_idx, :],
            split["y_test_t"],
            split["y_train_raw"],
            split["y_test_raw"],
            split["ecdf"],
            None,  # edge_table_test unused by eval_params_lobo
            test_idx,
        )


def _fold_split_compute_fn(
    *,
    seed: int,
    holdout_region: str,
    feature_mode: str,
    y_all: np.ndarray,
    fold_masks: Dict[str, np.ndarray],
    eps: float,
    y_transform: str,
    data_type: str,
) -> Optional[LoboFoldSplit]:
    """Adapter matching the ``cached_fold_split`` compute_fn signature.

    The ``seed`` and ``feature_mode`` kwargs are accepted (required by
    :func:`cached_fold_split`) but not used here -- they only participate
    in the cache key. The actual split is determined by
    ``holdout_region`` + ``y_all`` + ``fold_masks`` + ``eps`` + ``y_transform``
    + ``data_type``.
    """
    del seed, feature_mode  # cache-key-only
    # y_transform/data_type arrive here as plain str from YAML config;
    # compute_lobo_fold_split validates them against its Literal internally.
    return compute_lobo_fold_split(
        holdout_region=holdout_region,
        y_all=y_all,
        fold_masks=fold_masks,
        eps=eps,
        y_transform=y_transform,  # ty: ignore[invalid-argument-type]
        data_type=data_type,  # ty: ignore[invalid-argument-type]
    )


def eval_params_lobo(
    data: AlignmentData,
    *,
    version: Literal["naive", "voxel"],
    metric: str,
    params: Dict,
    num_boost_round: int,
    seed: int,
    eps: float,
    y_transform: Literal["none", "log1p", "ecdf"],
    data_type: Literal["SC", "FC"],
    device: Literal["cpu", "cuda"],
    fixed_regions_subset: Optional[Sequence[str]] = None,
    valid_fraction: float = 0.15,
    early_stopping_rounds: int = 50,
    fold_cache_dir: Optional[Path | str] = None,
    feature_mode: str = "cci_only",
) -> Tuple[float, float, float, float, List[FoldArtifact]]:
    """Run LOBO evaluation with given params, optionally on a region subset.

    When ``fold_cache_dir`` is provided, per-region train/test index splits
    + y-transforms are cached to disk via :func:`cached_fold_split`, keyed
    by ``(seed, holdout_region, feature_mode)``. Successive calls with the
    same keys skip the re-derivation and read indices from pickle. Pass
    ``feature_mode`` (the value used upstream in ``select_features``) so
    the cache key segregates runs that applied different feature subsets.
    """
    metric_names = data.metric_names
    j = metric_names.index(metric)
    SC = data.SC_voxel if version == "voxel" else data.SC_naive
    y_all = SC[:, j].astype(float)

    regions_all = infer_regions(data.edge_table)
    regions_use = (
        list(fixed_regions_subset) if fixed_regions_subset is not None else regions_all
    )

    # Use precomputed masks for full region set, restrict via regions_use
    fold_masks = precompute_fold_masks(data.edge_table, regions_all)

    fold_artifacts: List[FoldArtifact] = []
    fold_rmses: List[float] = []
    fold_maes: List[float] = []
    fold_ns: List[int] = []

    if fold_cache_dir is not None:
        fold_iter: Iterator = _iter_cached_fold_tuples(
            X=data.X,
            y_all=y_all,
            fold_masks=fold_masks,
            regions_use=regions_use,
            eps=eps,
            y_transform=y_transform,
            data_type=data_type,
            fold_cache_dir=Path(fold_cache_dir),
            seed=seed,
            feature_mode=feature_mode,
        )
    else:
        fold_iter = iter_lobo_folds(
            data.X,
            data.edge_table,
            y_all,
            fold_masks,
            eps=eps,
            y_transform=y_transform,
            data_type=data_type,
            include_edge_tables=False,
            regions=regions_use,
        )

    for fold in fold_iter:
        (
            holdout_region,
            X_train,
            y_train_t,
            X_test,
            y_test_t,
            _y_train_raw,
            y_test_raw,
            ecdf,
            edge_table_test,
            test_idx,
        ) = fold

        split_seed = int(stable_hash_int(f"{seed}_{holdout_region}"))

        y_pred, best_iter, model_raw = train_predict_xgb(
            X_train,
            y_train_t,
            X_test,
            params=params,
            num_boost_round=num_boost_round,
            split_seed=split_seed,
            booster_seed=int(seed),
            device=device,
            valid_fraction=valid_fraction,
            early_stopping_rounds=early_stopping_rounds,
        )

        fold_rmse = rmse(y_test_t, y_pred)
        fold_mae = mae(y_test_t, y_pred)

        fa = FoldArtifact(
            holdout_region=holdout_region,
            metric=metric,
            version=version,
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            n_features=int(X_train.shape[1]),
            eps=float(eps),
            y_transform=str(y_transform),
            params=dict(params),
            best_iteration=int(best_iter),
            model_raw=model_raw,
            ecdf=ecdf,
            test_idx=np.asarray(test_idx),
            y_test_raw=np.asarray(y_test_raw),
            y_test_t=np.asarray(y_test_t),
            y_pred=np.asarray(y_pred),
            fold_rmse=float(fold_rmse),
            fold_mae=float(fold_mae),
            edge_table_test=edge_table_test,
            eval_metrics=list(params.get("eval_metric", [])),
        )

        fold_artifacts.append(fa)
        fold_rmses.append(float(fold_rmse))
        fold_maes.append(float(fold_mae))
        fold_ns.append(int(fa.n_test))

    if len(fold_artifacts) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), []

    macro_rmse = float(np.mean(fold_rmses))
    macro_mae = float(np.mean(fold_maes))
    n_tot = float(np.sum(fold_ns))
    micro_mse = float(
        np.sum([n * (r**2) for n, r in zip(fold_ns, fold_rmses)]) / n_tot
    )
    micro_rmse = float(np.sqrt(micro_mse))
    micro_mae = float(
        np.sum([n * m for n, m in zip(fold_ns, fold_maes)]) / n_tot
    )

    return micro_rmse, macro_rmse, micro_mae, macro_mae, fold_artifacts


# ---------------------------------------------------------------------------
#  Optuna orchestration
# ---------------------------------------------------------------------------

def run_optuna_hpo(
    data: AlignmentData,
    *,
    version: Literal["naive", "voxel"],
    metric: str,
    loss: str,
    huber_slope: Optional[float],
    eval_metrics: List[str],
    seed: int,
    eps: float,
    y_transform: Literal["none", "log1p", "ecdf"],
    data_type: Literal["SC", "FC"],
    device: Literal["cpu", "cuda"],
    n_trials: int,
    n_hpo_regions: int,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 300,
    valid_fraction: float = 0.20,
    tune_huber_slope: bool = False,
    storage_path: Optional[Path | str] = None,
    study_name: Optional[str] = None,
    fold_cache_dir: Optional[Path | str] = None,
    feature_mode: str = "cci_only",
) -> Tuple[optuna.Study, Dict]:
    """Run full Optuna HPO and return (study, best_params_xgb).

    If ``storage_path`` is provided, the study is backed by a
    :class:`JournalStorage` journal file and can be resumed by re-invoking
    this function with the same ``storage_path`` + ``study_name`` (trials
    persist to disk after every event).  If ``storage_path`` is ``None``,
    the study is in-memory (legacy behaviour).

    If ``fold_cache_dir`` is provided, per-region LOBO splits are cached
    to disk via :func:`cached_fold_split` and reused across trials, keyed
    by ``(seed, holdout_region, feature_mode)``. Pass ``feature_mode`` so
    the cache key segregates runs that applied different feature subsets.

    The study is multi-objective
    (``directions=["minimize", "minimize"]`` over micro / macro loss) so
    Optuna pruners are not used here — ``trial.report`` raises
    ``NotImplementedError`` on multi-objective studies.
    """
    regions_all = infer_regions(data.edge_table)
    fixed_subset = choose_fixed_region_subset(
        regions_all, n=n_hpo_regions, seed=seed
    )
    logger.info("Using %d/%d regions for HPO", len(fixed_subset), len(regions_all))

    create_kwargs: Dict[str, Any] = {
        "directions": ["minimize", "minimize"],
    }
    if storage_path is not None:
        create_kwargs["storage"] = build_journal_storage(storage_path)
        create_kwargs["load_if_exists"] = True
        if study_name is None:
            raise ValueError(
                "study_name must be provided when storage_path is set "
                "(journal-backed studies require a stable name for resume)."
            )
        create_kwargs["study_name"] = study_name
        logger.info(
            "HPO: journal storage=%s, study=%s (resumable)",
            storage_path, study_name,
        )
    elif study_name is not None:
        create_kwargs["study_name"] = study_name

    study = optuna.create_study(**create_kwargs)

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        params = make_xgb_gpu_param_space(trial, seed=seed)

        huber_slope_trial = huber_slope
        if loss == "pseudohuber" and tune_huber_slope:
            huber_slope_trial = trial.suggest_categorical(
                "huber_slope", [1e-3, 1e-2, 1e-1, 1]
            )
        params = apply_loss_and_metrics(
            params, loss=loss, huber_slope=huber_slope_trial, eval_metrics=eval_metrics
        )

        _, _, _, _, fold_artifacts = eval_params_lobo(
            data,
            version=version,
            metric=metric,
            params=params,
            num_boost_round=num_boost_round,
            seed=seed,
            eps=eps,
            y_transform=y_transform,
            data_type=data_type,
            device=device,
            fixed_regions_subset=fixed_subset,
            valid_fraction=valid_fraction,
            early_stopping_rounds=early_stopping_rounds,
            fold_cache_dir=fold_cache_dir,
            feature_mode=feature_mode,
        )
        opt_micro, opt_macro, _ = aggregate_fold_metrics_for_loss(
            fold_artifacts, loss=loss, huber_slope=huber_slope_trial
        )
        logger.info(
            "Trial %d: micro_%s=%.6f, macro_%s=%.6f",
            trial.number, loss, opt_micro, loss, opt_macro,
        )
        return opt_micro, opt_macro

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=sys.stderr.isatty(),
        catch=(xgb.core.XGBoostError,),
    )

    # Extract best params
    # optuna types FrozenTrial.values as Optional to cover running/pruned
    # trials; every entry in study.best_trials is COMPLETE and has values.
    best_trial = min(study.best_trials, key=lambda t: (t.values[0], t.values[1]))  # ty: ignore[not-subscriptable]
    best_huber = huber_slope
    if loss == "pseudohuber" and tune_huber_slope:
        best_huber = float(best_trial.params.get("huber_slope", huber_slope))

    base_best = {
        "eta": float(best_trial.params["eta"]),
        "max_depth": int(best_trial.params["max_depth"]),
        "min_child_weight": float(best_trial.params["min_child_weight"]),
        "gamma": float(best_trial.params["gamma"]),
        "subsample": float(best_trial.params["subsample"]),
        "colsample_bytree": float(best_trial.params["colsample_bytree"]),
        "lambda": float(best_trial.params["reg_lambda"]),
        "alpha": float(best_trial.params["reg_alpha"]),
        "max_bin": int(best_trial.params["max_bin"]),
        "seed": int(seed),
    }
    best_params_xgb = apply_loss_and_metrics(
        base_best, loss=loss, huber_slope=best_huber, eval_metrics=eval_metrics
    )

    logger.info(
        "HPO done. Best trial #%d, values=%s",
        best_trial.number,
        [f"{v:.6f}" for v in best_trial.values],  # ty: ignore[not-iterable]
    )
    return study, best_params_xgb
