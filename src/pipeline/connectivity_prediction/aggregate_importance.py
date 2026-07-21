#!/usr/bin/env python3
"""Aggregate feature importance across LOBO fold artifacts.

Reads trained fold pickles, computes per-fold XGBoost gain vectors, assigns
each edge to a unique fold for edge-weighted averaging, then writes the
importance CSV files that ``cross_target_biology.py`` and the figure
panels depend on.

Outputs (to the model directory's ``plots/`` subdirectory):
    feature_importance_weighted_gain.csv   - full per-feature table
    feature_importance_top20_abs.csv       - top-K by abs gain
    feature_importance_top20_rel.csv       - top-K by rel gain
    importance_by_lr_abs.csv               - LR-collapsed (abs)
    importance_by_lr_rel.csv               - LR-collapsed (rel)
    importance_by_sender_celltype_abs.csv  - sender CT-collapsed (abs)
    importance_by_sender_celltype_rel.csv  - sender CT-collapsed (rel)
    importance_by_receiver_celltype_abs.csv
    importance_by_receiver_celltype_rel.csv
    lr_importance_by_fold_matrix.csv       - LR x fold matrix (normalized)
    fold_weights.csv                       - per-fold edge assignment counts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm import tqdm

from clrc.core.io import find_repo_root, load_alignment_data, load_pickle, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.core.types import FoldArtifact
from clrc.prediction.evaluation import (
    assign_unique_edges,
    collect_all_candidates,
    compute_metrics,
    summarize_candidates,
)
from clrc.prediction.importance import (
    add_cell_class_column,
    aggregate_by_group,
    aggregate_full_importance_pipeline,
)
from clrc.prediction.lobo import select_features


def _load_all_folds(folds_dir: Path) -> Dict[str, FoldArtifact]:
    """Load every fold pickle under ``folds_dir`` keyed by holdout_region."""
    out: Dict[str, FoldArtifact] = {}
    fold_paths = sorted(folds_dir.glob("*.pkl"))
    if not fold_paths:
        raise FileNotFoundError(f"No fold pickles under {folds_dir}")
    for path in tqdm(fold_paths, desc="Loading folds"):
        fa = load_pickle(path)
        out[str(fa.holdout_region)] = fa
    return out


def _resolve_model_dir(
    cfg: dict, target: str, explicit: Path | None, repo_root: Path
) -> Path:
    """Find the model directory containing ``folds/`` for the given target.

    Search order:
        1. ``--model-dir`` argument if provided
        2. ``{out_base}/models/{target}`` or its timestamped subdirs
        3. ``data/expanded_ABC/model_GBM/*{version}_{metric}*`` (legacy)
    """
    if explicit is not None:
        return explicit

    out_base = Path(cfg["output"]["base_dir"])
    std = out_base / "models" / target
    if (std / "folds").exists():
        return std
    nested = sorted(std.glob("*/folds"), reverse=True)
    if nested:
        return nested[0].parent

    target_cfg = cfg["xgboost"][target]
    legacy_base = repo_root / "data" / "expanded_ABC" / "model_GBM"
    if legacy_base.exists():
        matches = sorted(
            legacy_base.glob(f"*_{target_cfg['version']}_{target_cfg['metric']}*"),
            reverse=True,
        )
        for m in matches:
            if (m / "folds").exists():
                return m

    raise FileNotFoundError(
        f"Could not locate model directory with folds/ for target={target}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate XGBoost feature importance across LOBO fold artifacts"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", choices=["sc", "fc"], required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("aggregate_importance", out_base)
    repo_root = find_repo_root()

    # --- Load alignment data and select features (aligned triple) ---
    target_cfg = cfg["xgboost"][args.target]
    data = load_alignment_data(
        cfg["data"]["alignment_pkl"],
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )
    # select_features returns (X, feature_names, meta) with all three aligned
    # row-for-row. It raises ValueError if data.meta or data.feature_names is
    # missing, so no defensive None-checks needed here.
    _, feature_names, meta = select_features(
        data, feature_mode=target_cfg["feature_mode"]
    )

    # --- Locate fold artifacts ---
    model_dir = _resolve_model_dir(cfg, args.target, args.model_dir, repo_root)
    folds_dir = model_dir / "folds"
    logger.info("Loading folds from %s", folds_dir)
    fold_artifacts = _load_all_folds(folds_dir)
    logger.info("Loaded %d fold artifacts", len(fold_artifacts))

    # --- Candidate collection + unique-edge assignment ---
    candidates = collect_all_candidates(fold_artifacts.values())
    logger.info("Candidate summary: %s", summarize_candidates(candidates))
    unique_df, fold_weights = assign_unique_edges(candidates)
    metrics = compute_metrics(unique_df)
    logger.info(
        "Unique-edge metrics: R²=%.4f, ρ=%.4f, RMSE=%.4f, MAE=%.4f (n=%d)",
        metrics["R2_ecdf"], metrics["spearman_rho"], metrics["RMSE"], metrics["MAE"],
        metrics["n_samples"],
    )

    # --- Full importance pipeline ---
    feat_df, lr_fold_matrix = aggregate_full_importance_pipeline(
        fold_artifacts, fold_weights, feature_names, meta
    )

    # --- Write outputs ---
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Fold weights
    weights_df = (
        pd.DataFrame(
            {
                "holdout_region": list(fold_weights.keys()),
                "unique_edge_count_assigned": list(fold_weights.values()),
            }
        )
        .sort_values("holdout_region")
        .reset_index(drop=True)
    )
    weights_df.to_csv(plots_dir / "fold_weights.csv", index=False)

    # Full per-feature table
    feat_df.to_csv(plots_dir / "feature_importance_weighted_gain.csv", index=False)

    # Top-K feature tables
    top_k = args.top_k
    feat_df.sort_values("weighted_mean_gain_abs", ascending=False).head(top_k).to_csv(
        plots_dir / f"feature_importance_top{top_k}_abs.csv", index=False
    )
    feat_df.sort_values("weighted_mean_gain_rel", ascending=False).head(top_k).to_csv(
        plots_dir / f"feature_importance_top{top_k}_rel.csv", index=False
    )

    # LR-collapsed
    lr_abs = aggregate_by_group(feat_df, "lr_name", sort_by="aggregated_importance_abs")
    lr_rel = aggregate_by_group(feat_df, "lr_name", sort_by="aggregated_importance_rel")
    lr_abs.to_csv(plots_dir / "importance_by_lr_abs.csv", index=False)
    lr_rel.to_csv(plots_dir / "importance_by_lr_rel.csv", index=False)

    # Sender cell type
    sender_abs = add_cell_class_column(
        aggregate_by_group(feat_df, "ct_L", sort_by="aggregated_importance_abs")
    )
    sender_rel = add_cell_class_column(
        aggregate_by_group(feat_df, "ct_L", sort_by="aggregated_importance_rel")
    )
    sender_abs.to_csv(plots_dir / "importance_by_sender_celltype_abs.csv", index=False)
    sender_rel.to_csv(plots_dir / "importance_by_sender_celltype_rel.csv", index=False)

    # Receiver cell type
    recv_abs = add_cell_class_column(
        aggregate_by_group(feat_df, "ct_R", sort_by="aggregated_importance_abs")
    )
    recv_rel = add_cell_class_column(
        aggregate_by_group(feat_df, "ct_R", sort_by="aggregated_importance_rel")
    )
    recv_abs.to_csv(plots_dir / "importance_by_receiver_celltype_abs.csv", index=False)
    recv_rel.to_csv(plots_dir / "importance_by_receiver_celltype_rel.csv", index=False)

    # LR x fold matrix
    lr_fold_matrix.to_csv(plots_dir / "lr_importance_by_fold_matrix.csv")

    logger.info("Wrote importance outputs to %s", plots_dir)


if __name__ == "__main__":
    main()
