#!/usr/bin/env python3
"""Optuna HPO driver for LOBO XGBoost (SC or FC target).

Delegates all HPO logic to :func:`clrc.prediction.hpo.run_optuna_hpo`; this
driver wires config plumbing, journal storage for resumability, fold
caching, and rich best-params output.

Usage
-----
    uv run python src/pipeline/connectivity_prediction/hpo.py \\
        --config configs/abc_expanded.yaml \\
        --target sc \\
        --n-trials 50

Outputs (under ``<cfg.output.base_dir>/hpo/<target>/``):
    - ``best_params.json``       -- consumed by ``train_xgboost.py``
    - ``study_trials.csv``       -- Optuna trial-level summary
    - ``config.json``            -- reproducibility snapshot
    - ``journal.log``            -- Optuna JournalStorage (resume source)
    - ``fold_cache/*.pkl``       -- per-(seed, region, feature_mode) split cache
    - ``hpo.log``                -- driver run log

Resumability
------------
By default, if ``journal.log`` exists in the output directory, the
study is loaded and additional trials continue the existing search.
Pass ``--fresh`` to start a new study: the existing journal is
archived to ``journal.log.bak-<timestamp>`` before the new one is
created, so no history is silently discarded.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import xgboost as xgb

from clrc.core.io import load_alignment_data, load_yaml_config, save_json
from clrc.core.logging import setup_logging
from clrc.core.types import AlignmentData
from clrc.prediction.hpo import run_optuna_hpo
from clrc.prediction.lobo import select_features

logger = logging.getLogger("clrc.pipeline.hpo")


def _parse_eval_metrics(raw: str | None, *, loss: str) -> List[str]:
    """Parse --eval-metrics CLI value, defaulting based on loss."""
    if raw is None or raw.strip() == "":
        if loss == "rmse":
            return ["rmse", "mae"]
        if loss == "mae":
            return ["mae", "rmse"]
        if loss == "pseudohuber":
            return ["mae", "rmse"]
        return ["rmse", "mae"]
    return [m.strip() for m in raw.split(",") if m.strip()]


def _resolve_huber_slope(loss: str, raw: float | None) -> float | None:
    if loss == "pseudohuber":
        return float(1.0 if raw is None else raw)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for LOBO XGBoost (SC or FC).",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="YAML config (e.g. configs/abc_expanded.yaml).",
    )
    parser.add_argument(
        "--target", choices=["sc", "fc"], required=True,
        help="Which target to optimize. Uses cfg.xgboost.<target>.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override cfg.hpo.n_trials.",
    )
    parser.add_argument(
        "--n-hpo-regions", type=int, default=None,
        help="Override cfg.hpo.n_hpo_regions.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["cci_only", "distance_only", "cci_distance"],
        default=None,
        help="Override cfg.xgboost.<target>.feature_mode.",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help=(
            "Start a fresh study. If a journal.log exists under the output "
            "directory it is archived to journal.log.bak-<timestamp> before "
            "a new journal is created. Without this flag, an existing "
            "journal is reused (resume semantics, default)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Cap trials at 2 and short-circuit early (smoke-test mode).",
    )
    parser.add_argument(
        "--huber-slope", type=float, default=None,
        help="Huber slope for pseudohuber loss (ignored otherwise).",
    )
    parser.add_argument(
        "--tune-huber-slope", action="store_true",
        help="Let Optuna search huber_slope (only when loss=pseudohuber).",
    )
    parser.add_argument(
        "--eval-metrics", type=str, default=None,
        help="Comma-separated XGBoost eval metrics. Defaults based on loss.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    target_cfg = cfg["xgboost"][args.target]
    if args.feature_mode is not None:
        target_cfg["feature_mode"] = args.feature_mode
    feature_mode: str = target_cfg["feature_mode"]

    loss: str = cfg["xgboost"]["loss"]
    eval_metrics = _parse_eval_metrics(args.eval_metrics, loss=loss)
    huber_slope = _resolve_huber_slope(loss, args.huber_slope)

    n_trials = args.n_trials if args.n_trials is not None else cfg["hpo"]["n_trials"]
    n_hpo_regions = (
        args.n_hpo_regions if args.n_hpo_regions is not None else cfg["hpo"]["n_hpo_regions"]
    )
    if args.dry_run:
        n_trials = min(n_trials, 2)
        n_hpo_regions = min(n_hpo_regions, 5)

    out_base = Path(cfg["output"]["base_dir"])
    # Suffix non-default feature_modes so cci_only / distance_only /
    # cci_distance runs do not stomp each other's best_params.json.
    # The default cci_only path (<out>/hpo/<target>/) matches existing
    # abc_expanded_hpobest.yaml expectations for backward compat.
    feature_mode_suffix = "" if feature_mode == "cci_only" else f"_{feature_mode}"
    exp_dir = out_base / "hpo" / f"{args.target}{feature_mode_suffix}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    setup_logging("hpo", output_dir=exp_dir)
    logger.info("HPO target=%s, config=%s", args.target, args.config)
    logger.info("Output dir: %s", exp_dir)
    logger.info("n_trials=%d, n_hpo_regions=%d, feature_mode=%s, loss=%s",
                n_trials, n_hpo_regions, feature_mode, loss)

    # Load + select features
    alignment_pkl = target_cfg.get("alignment_pkl", cfg["data"]["alignment_pkl"])
    data = load_alignment_data(
        alignment_pkl,
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )
    X_selected, feature_names, meta = select_features(data, feature_mode=feature_mode)
    logger.info("Loaded alignment pickle; X_selected shape=%s", X_selected.shape)
    data_for_hpo = AlignmentData(
        edge_table=data.edge_table,
        X=X_selected,
        metric_names=data.metric_names,
        SC_naive=data.SC_naive,
        SC_voxel=data.SC_voxel,
        feature_names=feature_names,
        meta=meta,
        distance_vec=data.distance_vec,
    )

    # Resumability + fold caching paths. Journal storage is always on so
    # SIGTERM / crash restarts recover trials via --fresh's archive path.
    storage_path = exp_dir / "journal.log"
    if args.fresh and storage_path.exists():
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = storage_path.with_name(f"journal.log.bak-{stamp}")
        storage_path.rename(archive_path)
        logger.info("Archived existing journal.log -> %s", archive_path)
    study_name = f"hpo_{args.target}_{feature_mode}"
    fold_cache_dir = exp_dir / "fold_cache"

    # Reproducibility snapshot
    cfg_snapshot = {
        "target": args.target,
        "config_path": str(args.config),
        "alignment_pkl": str(alignment_pkl),
        "version": target_cfg["version"],
        "metric": target_cfg["metric"],
        "data_type": target_cfg["data_type"],
        "feature_mode": feature_mode,
        "device": cfg["xgboost"]["device"],
        "loss": loss,
        "huber_slope": huber_slope,
        "tune_huber_slope": bool(args.tune_huber_slope),
        "eval_metrics": eval_metrics,
        "y_transform": cfg["xgboost"]["y_transform"],
        "seed": cfg["xgboost"]["seed"],
        "eps": 0.0,
        "valid_fraction": cfg["xgboost"]["valid_fraction"],
        "early_stopping_rounds": cfg["xgboost"]["early_stopping_rounds"],
        "max_boost_rounds": cfg["xgboost"]["max_boost_rounds"],
        "n_trials": n_trials,
        "n_hpo_regions": n_hpo_regions,
        "fresh": bool(args.fresh),
        "storage_path": str(storage_path),
        "study_name": study_name,
        "fold_cache_dir": str(fold_cache_dir),
        "n_features": int(X_selected.shape[1]),
        "xgboost_version": getattr(xgb, "__version__", "unknown"),
    }
    save_json(cfg_snapshot, exp_dir / "config.json")

    # Run HPO
    study, best_params_xgb = run_optuna_hpo(
        data_for_hpo,
        version=target_cfg["version"],
        metric=target_cfg["metric"],
        loss=loss,
        huber_slope=huber_slope,
        eval_metrics=eval_metrics,
        seed=cfg["xgboost"]["seed"],
        eps=0.0,
        y_transform=cfg["xgboost"]["y_transform"],
        data_type=target_cfg["data_type"],
        device=cfg["xgboost"]["device"],
        n_trials=n_trials,
        n_hpo_regions=n_hpo_regions,
        num_boost_round=cfg["xgboost"]["max_boost_rounds"],
        early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        valid_fraction=cfg["xgboost"]["valid_fraction"],
        tune_huber_slope=args.tune_huber_slope,
        storage_path=storage_path,
        study_name=study_name,
        fold_cache_dir=fold_cache_dir,
        feature_mode=feature_mode,
    )

    # Trials CSV
    try:
        study.trials_dataframe().to_csv(exp_dir / "study_trials.csv", index=False)
    except (OSError, ValueError) as e:
        logger.warning("Failed to write study_trials.csv: %r", e)

    # Rich best_params.json, consumed by train_xgboost.py
    best_trial = min(study.best_trials, key=lambda t: (t.values[0], t.values[1]))
    best_huber = huber_slope
    if loss == "pseudohuber" and args.tune_huber_slope:
        best_huber = float(best_trial.params.get("huber_slope", huber_slope))
    save_json(
        {
            "best_trial_number": int(best_trial.number),
            "best_values": [float(v) for v in best_trial.values],
            "best_trial_params_raw": dict(best_trial.params),
            "best_params_xgb": best_params_xgb,
            "loss": loss,
            "optimization_objectives": [f"micro_{loss}", f"macro_{loss}"],
            "huber_slope": float(best_huber) if best_huber is not None else None,
            "eval_metrics": eval_metrics,
        },
        exp_dir / "best_params.json",
    )

    logger.info(
        "HPO done. Best trial #%d, values=%s",
        best_trial.number,
        [f"{v:.6f}" for v in best_trial.values],
    )
    logger.info("Artifacts written under %s", exp_dir)


if __name__ == "__main__":
    main()
