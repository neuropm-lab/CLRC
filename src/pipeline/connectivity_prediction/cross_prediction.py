#!/usr/bin/env python3
"""Bias-validation driver: refit LOBO on bias-labeled feature subsets.

Tests whether the identified SC-biased LR features actually predict SC
better than FC (and vice versa). Refits LOBO XGBoost on six feature
subsets:

    * ``sc_biased``     -- only SC-biased LR pairs (HPO-best categorization).
    * ``fc_biased``     -- only FC-biased LR pairs.
    * ``balanced``      -- only Balanced LR pairs.
    * ``uniform_null``  -- N draws of ``k`` random LR pairs drawn uniformly
      from all non-zero pairs.
    * ``matched_null``  -- N draws of ``k`` random LR pairs with summed
      combined importance matched to the bias-label target sum;
      defense-in-depth convention from Markello 2021 / Hansen 2022.

Training uses **target-model HPO params** -- predicting SC uses SC
HPO-best, predicting FC uses FC HPO-best.

Example — dry-run 2 random draws × 2 LOBO folds::

    python src/pipeline/connectivity_prediction/cross_prediction.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --target sc \\
        --subset-type uniform_null \\
        --n-draws 2 \\
        --seed 0 \\
        --max-folds 2 \\
        --output-dir /tmp/cross_prediction_dry_run

Outputs (under ``<output_dir>/<target>/<subset-type>/``):

    subsets.json        — every drawn LR subset, by draw index.
    run_config.json     — resolved config + CLI args + seeds (reproducibility).
    fold_metrics.csv    — (draw_idx × holdout_region) per-fold RMSE / MAE.
    summary.json        — mean / median fold metrics per draw.

The driver reuses :func:`clrc.prediction.xgboost.train_predict_xgb` as-is.
It does NOT train SHAP, save boosters, or recompute importance aggregation —
those are downstream concerns handled by separate pipeline stages.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from clrc.core.io import (
    load_alignment_data,
    load_yaml_config,
    save_json,
    safe_filename,
    stable_hash_int,
    timestamp,
)
from clrc.core.logging import setup_logging
from clrc.core.metrics import mae, rmse
from clrc.core.types import AlignmentData
from clrc.prediction.bias_validation import (
    draw_random_lr_subsets_importance_matched,
    draw_random_lr_subsets_uniform,
    feature_mask_for_lr_subset,
)
from clrc.prediction.lobo import (
    infer_regions,
    iter_lobo_folds,
    precompute_fold_masks,
    select_features,
)
from clrc.prediction.xgboost import train_predict_xgb


SubsetType = str  # one of: sc_biased, fc_biased, balanced, uniform_null, matched_null

_CATEGORY_LABELS = {
    "sc_biased": "SC-biased",
    "fc_biased": "FC-biased",
    "balanced": "Balanced",
}


# ---------------------------------------------------------------------------
#  LR subset loading / drawing
# ---------------------------------------------------------------------------


def _resolve_categories_csv(cfg: dict) -> Path:
    """Locate the feature_categories.csv on disk.

    Convention: ``<output.base_dir>/interpretation/feature_categories.csv``.
    This file is written by ``src/pipeline/connectivity_prediction/cross_target_biology.py``
    and contains columns: group_name, importance_sc, importance_fc,
    importance_combined, category, ...
    """
    base_dir = Path(cfg["output"]["base_dir"])
    candidate = base_dir / "interpretation" / "feature_categories.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected categorization CSV at {candidate}. "
            f"Run cross_target_biology.py on HPO-best params first."
        )
    return candidate


def _load_categories(cfg: dict) -> pd.DataFrame:
    path = _resolve_categories_csv(cfg)
    df = pd.read_csv(path)
    required = {"group_name", "importance_sc", "importance_fc",
                "importance_combined", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"feature_categories.csv at {path} is missing columns {missing}."
        )
    return df


def _build_lr_subsets(
    subset_type: SubsetType,
    target: str,
    categories: pd.DataFrame,
    n_draws: int,
    seed: int,
    matched_tol: float = 0.10,
    matched_max_attempts: int = 100000,
) -> Tuple[List[List[str]], dict]:
    """Return a list of LR-pair subsets plus bookkeeping metadata."""
    pool = categories["group_name"].astype(str).tolist()
    imp_map = dict(
        zip(
            categories["group_name"].astype(str),
            categories["importance_combined"].astype(float),
        )
    )

    if subset_type in _CATEGORY_LABELS:
        label = _CATEGORY_LABELS[subset_type]
        sub = categories.loc[categories["category"] == label, "group_name"]
        subsets = [sub.astype(str).tolist()]
        meta = {
            "kind": "named_category",
            "category": label,
            "k": int(len(subsets[0])),
            "sum_importance_combined": float(
                categories.loc[
                    categories["category"] == label, "importance_combined"
                ].sum()
            ),
        }
        return subsets, meta

    # Random nulls match the k of the target category (sc_biased for predict-SC,
    # fc_biased for predict-FC -- conventional same-label-to-same-target framing).
    anchor_label = "SC-biased" if target == "sc" else "FC-biased"
    anchor = categories.loc[categories["category"] == anchor_label]
    k = int(len(anchor))
    target_sum = float(anchor["importance_combined"].sum())

    if subset_type == "uniform_null":
        subsets = draw_random_lr_subsets_uniform(
            pool, k=k, n_draws=n_draws, seed=seed
        )
        meta = {
            "kind": "uniform_null",
            "anchor_category": anchor_label,
            "k": k,
            "pool_size": len(pool),
            "target_sum_importance_combined": target_sum,
        }
    elif subset_type == "matched_null":
        subsets = draw_random_lr_subsets_importance_matched(
            pool,
            imp_map,
            target_sum=target_sum,
            k=k,
            n_draws=n_draws,
            seed=seed,
            tol=matched_tol,
            max_attempts_per_draw=matched_max_attempts,
        )
        meta = {
            "kind": "matched_null",
            "anchor_category": anchor_label,
            "k": k,
            "pool_size": len(pool),
            "target_sum_importance_combined": target_sum,
            "tol": matched_tol,
        }
    else:
        raise ValueError(
            f"Unknown subset_type: {subset_type!r}. "
            f"Expected one of: sc_biased, fc_biased, balanced, "
            f"uniform_null, matched_null."
        )
    return subsets, meta


# ---------------------------------------------------------------------------
#  Training loop for one LR subset
# ---------------------------------------------------------------------------


def _run_one_subset(
    *,
    data: AlignmentData,
    feature_names: Sequence[str],
    meta: Sequence[dict],
    lr_subset: Sequence[str],
    y_all: np.ndarray,
    fold_masks: Dict[str, np.ndarray],
    regions_use: Sequence[str],
    target_cfg: dict,
    xgb_cfg: dict,
    params_blob: dict,
    logger: logging.Logger,
) -> List[dict]:
    """Run full LOBO on a single LR subset; return per-fold metrics."""
    mask = feature_mask_for_lr_subset(feature_names, meta, lr_subset)
    if mask.sum() == 0:
        raise ValueError("Subset produced an all-False feature mask; aborting.")

    X_sub = data.X[:, mask]
    logger.info(
        "LR subset: %d pairs → %d features (of %d total).",
        len(lr_subset), int(mask.sum()), int(data.X.shape[1]),
    )

    best_params_xgb = params_blob["best_params_xgb"]
    fold_rows: List[dict] = []

    for fold in iter_lobo_folds(
        X_sub, data.edge_table, y_all, fold_masks,
        eps=0.0,
        y_transform=xgb_cfg["y_transform"],
        data_type=target_cfg["data_type"],
        include_edge_tables=False,
        regions=regions_use,
    ):
        (holdout_region, X_train, y_train_t, X_test, y_test_t,
         _y_train_raw, _y_test_raw, _ecdf, _et_test, _test_idx) = fold

        split_seed = int(stable_hash_int(f"{xgb_cfg['seed']}_{holdout_region}"))
        y_pred, best_iter, _model_raw = train_predict_xgb(
            X_train, y_train_t, X_test,
            params=best_params_xgb,
            num_boost_round=xgb_cfg["max_boost_rounds"],
            split_seed=split_seed,
            booster_seed=xgb_cfg["seed"],
            device=xgb_cfg["device"],
            valid_fraction=xgb_cfg["valid_fraction"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
        )

        fold_rows.append({
            "holdout_region": holdout_region,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "best_iteration": int(best_iter),
            "fold_rmse": float(rmse(y_test_t, y_pred)),
            "fold_mae": float(mae(y_test_t, y_pred)),
            "n_features_used": int(mask.sum()),
        })
        logger.debug(
            "[%s] n_test=%d best_iter=%d RMSE=%.6f MAE=%.6f",
            holdout_region, fold_rows[-1]["n_test"], best_iter,
            fold_rows[-1]["fold_rmse"], fold_rows[-1]["fold_mae"],
        )

    return fold_rows


# ---------------------------------------------------------------------------
#  Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bias-validation cross-prediction driver."
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config (e.g. abc_expanded_hpobest.yaml).")
    parser.add_argument("--target", choices=["sc", "fc"], required=True,
                        help="Which connectome to predict.")
    parser.add_argument(
        "--subset-type",
        choices=["sc_biased", "fc_biased", "balanced",
                 "uniform_null", "matched_null"],
        required=True,
        help="Feature-subset selector.",
    )
    parser.add_argument("--n-draws", type=int, default=100,
                        help="Number of random subsets (nulls only). Default 100.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for null-subset drawing.")
    parser.add_argument(
        "--matched-tol",
        type=float,
        default=0.10,
        help=(
            "Relative tolerance for importance-matched null (matched_null "
            "only). Default 0.10. On HPO-best expanded ABC, the SC-biased "
            "sum (0.327) is ~8 sigma below the uniform-draw mean (0.370), so "
            "a tolerance of 5%% is infeasible; 10%% captures ~2%% of proposals. "
            "Tighten only with a correspondingly larger --matched-max-attempts."
        ),
    )
    parser.add_argument(
        "--matched-max-attempts",
        type=int,
        default=100000,
        help=(
            "Max rejection-sampling proposals per accepted matched-null draw. "
            "Default 100000."
        ),
    )
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Dry-run cap: only run this many LOBO folds. "
                             "Omit for full LOBO (all regions).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory. "
                             "Default: <base_dir>/bias_validation/<target>/<subset-type>/.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    target_cfg = cfg["xgboost"][args.target]
    xgb_cfg = cfg["xgboost"]
    out_base = Path(cfg["output"]["base_dir"])

    logger = setup_logging("cross_prediction", out_base)
    logger.info("Bias-validation run: target=%s subset_type=%s",
                args.target, args.subset_type)

    # ---- Load alignment data + feature selection ------------------------
    data = load_alignment_data(
        target_cfg.get("alignment_pkl", cfg["data"]["alignment_pkl"]),
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )
    X_selected, feature_names, meta = select_features(
        data, feature_mode=target_cfg["feature_mode"]
    )
    data = AlignmentData(
        edge_table=data.edge_table,
        X=X_selected,
        metric_names=data.metric_names,
        SC_naive=data.SC_naive,
        SC_voxel=data.SC_voxel,
        feature_names=feature_names,
        meta=meta,
        distance_vec=data.distance_vec,
    )

    # ---- Load HPO-best params (target-model HPO) ------------------------
    from clrc.core.io import find_repo_root
    params_path = Path(target_cfg["params_json"])
    if not params_path.is_absolute():
        params_path = find_repo_root() / params_path
    with params_path.open() as f:
        params_blob = json.load(f)
    logger.info("Loaded target-model HPO params from %s", params_path)

    # ---- Build LR subsets ----------------------------------------------
    categories = _load_categories(cfg)
    subsets, subset_meta = _build_lr_subsets(
        subset_type=args.subset_type,
        target=args.target,
        categories=categories,
        n_draws=args.n_draws,
        seed=args.seed,
        matched_tol=args.matched_tol,
        matched_max_attempts=args.matched_max_attempts,
    )
    logger.info("Built %d LR subset(s) for subset_type=%s (meta=%s).",
                len(subsets), args.subset_type, subset_meta)

    # ---- Target vector -------------------------------------------------
    j = data.metric_names.index(target_cfg["metric"])
    SC = data.SC_voxel if target_cfg["version"] == "voxel" else data.SC_naive
    y_all = SC[:, j].astype(float)

    regions_all = infer_regions(data.edge_table)
    fold_masks = precompute_fold_masks(data.edge_table, regions_all)
    regions_use = list(regions_all)
    if args.max_folds is not None and args.max_folds < len(regions_use):
        regions_use = regions_use[: args.max_folds]
        logger.warning(
            "Dry-run: capped LOBO folds to %d (of %d).",
            len(regions_use), len(regions_all),
        )

    # ---- Output directory ----------------------------------------------
    out_dir = args.output_dir or (
        out_base / "bias_validation" / args.target / args.subset_type /
        f"{timestamp()}_{safe_filename(target_cfg['metric'])}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing outputs to %s", out_dir)

    # Write subsets + config BEFORE training so reproducibility is trivially
    # verifiable even if a later fit crashes.
    save_json(
        {"subsets": subsets, "meta": subset_meta},
        out_dir / "subsets.json",
    )
    save_json(
        {
            "cli": {
                "config": str(args.config),
                "target": args.target,
                "subset_type": args.subset_type,
                "n_draws": args.n_draws,
                "seed": args.seed,
                "max_folds": args.max_folds,
                "matched_tol": args.matched_tol,
                "matched_max_attempts": args.matched_max_attempts,
            },
            "resolved_cfg": {
                "output_base_dir": str(out_base),
                "xgboost_seed": xgb_cfg["seed"],
                "y_transform": xgb_cfg["y_transform"],
                "device": xgb_cfg["device"],
                "target_cfg": target_cfg,
                "params_json": str(params_path),
            },
            "n_regions_total": len(regions_all),
            "n_regions_used": len(regions_use),
        },
        out_dir / "run_config.json",
    )

    # ---- Per-subset LOBO training --------------------------------------
    all_fold_rows: List[dict] = []
    per_draw_summary: List[dict] = []

    for draw_idx, lr_subset in enumerate(subsets):
        logger.info(
            "--- Subset %d / %d (size=%d) ---",
            draw_idx + 1, len(subsets), len(lr_subset),
        )
        fold_rows = _run_one_subset(
            data=data,
            feature_names=feature_names,
            meta=meta,
            lr_subset=lr_subset,
            y_all=y_all,
            fold_masks=fold_masks,
            regions_use=regions_use,
            target_cfg=target_cfg,
            xgb_cfg=xgb_cfg,
            params_blob=params_blob,
            logger=logger,
        )
        for r in fold_rows:
            r["draw_idx"] = draw_idx
        all_fold_rows.extend(fold_rows)

        fold_df = pd.DataFrame(fold_rows)
        per_draw_summary.append({
            "draw_idx": draw_idx,
            "n_folds": int(len(fold_rows)),
            "mean_rmse": float(fold_df["fold_rmse"].mean()),
            "median_rmse": float(fold_df["fold_rmse"].median()),
            "mean_mae": float(fold_df["fold_mae"].mean()),
            "median_mae": float(fold_df["fold_mae"].median()),
        })

    pd.DataFrame(all_fold_rows).to_csv(
        out_dir / "fold_metrics.csv", index=False
    )
    save_json(
        {
            "per_draw_summary": per_draw_summary,
            "subset_meta": subset_meta,
        },
        out_dir / "summary.json",
    )

    logger.info(
        "DONE — %d subset(s) × %d fold(s) = %d fits total. Outputs in %s",
        len(subsets), len(regions_use), len(all_fold_rows), out_dir,
    )


if __name__ == "__main__":
    main()
