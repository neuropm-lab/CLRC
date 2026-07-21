#!/usr/bin/env python3
"""Per-feature fold-level CI driver for XGBoost relative importance.

For each feature and each supported group (LR pair, sender cell-type,
receiver cell-type), compute the 2.5th / 97.5th percentile across the
LOBO folds of per-fold relative importance.

This is NOT a classical bootstrap -- no resampling. The LOBO folds
themselves are the independent units and percentiles are computed
directly on that empirical distribution. Group-level CIs sum features
within each group PER FOLD first, then take percentiles across the
per-fold sum distribution (see
:func:`clrc.prediction.bootstrap.compute_group_level_ci` docstring for
why this is the correct order).

Inputs
------
Same fold-artifact directory used by ``aggregate_importance.py``:
``<model_dir>/folds/*.pkl``. No retraining.

Outputs
-------
Written to ``<model_dir>/ci/``:

    feature_ci.csv                 — per-feature CIs.
    lr_ci.csv                      — per-LR-pair aggregated CIs.
    sender_celltype_ci.csv         — per-sender-cell-type aggregated CIs.
    receiver_celltype_ci.csv       — per-receiver-cell-type aggregated CIs.
    fold_importance_matrix.csv     — (fold × feature) relative-importance
                                     matrix used to construct the CIs.
                                     Rows = holdout_region (sorted).

Example::

    python src/pipeline/connectivity_prediction/bootstrap_ci.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --target sc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from clrc.core.io import (
    find_repo_root,
    load_alignment_data,
    load_pickle,
    load_yaml_config,
)
from clrc.core.logging import setup_logging
from clrc.core.types import FoldArtifact
from clrc.prediction.bootstrap import (
    compute_feature_level_ci,
    compute_group_level_ci,
)
from clrc.prediction.importance import compute_fold_importances
from clrc.prediction.lobo import select_features


def _load_all_folds(folds_dir: Path) -> Dict[str, FoldArtifact]:
    """Load every fold pickle under ``folds_dir`` keyed by holdout_region.

    Mirrors the loader in ``aggregate_importance.py`` — intentionally
    duplicated here rather than factored into a shared helper, since the
    two drivers are meant to stay independently runnable.
    """
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

    Same resolution logic as in ``aggregate_importance.py`` (again,
    intentionally duplicated rather than shared).
    """
    if explicit is not None:
        return explicit

    out_base = Path(cfg["output"]["base_dir"])
    std = out_base / "models" / target
    if (std / "folds").exists():
        return std
    # Exclude runs whose name carries a non-default feature-mode suffix
    # (e.g. *_distance_only has 1 feature, *_cci_distance has 3993); we want
    # the plain cci_only run as the baseline for per-feature CIs.
    nested = [
        p.parent for p in sorted(std.glob("*/folds"), reverse=True)
        if not p.parent.name.endswith("_distance_only")
        and not p.parent.name.endswith("_cci_distance")
    ]
    if nested:
        return nested[0]

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


def _build_relative_importance_matrix(
    fold_importances: Dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Stack per-fold gain vectors into an L1-normalized (fold × feature) matrix.

    Each row is one fold's gain vector, rescaled so it sums to 1 (the
    standard "relative importance" convention, matching the
    row-normalization applied in
    :func:`clrc.prediction.importance.aggregate_importances` when computing
    ``imp_rel``). Folds with zero total gain (degenerate) become rows of
    zeros rather than NaN.

    Returns ``(matrix, sorted_holdout_names)``.
    """
    holdouts = sorted(fold_importances.keys())
    if not holdouts:
        raise ValueError("fold_importances is empty")
    mat = np.vstack(
        [np.asarray(fold_importances[h], dtype=float) for h in holdouts]
    )
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mat_rel = np.where(row_sums > 0, mat / row_sums, 0.0)
    return mat_rel, holdouts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-feature and per-group fold-level percentile CIs "
            "from LOBO fold artifacts."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", choices=["sc", "fc"], required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided significance level (default 0.05 -> 95%% CI).",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("bootstrap_ci", out_base)
    repo_root = find_repo_root()

    # --- Load alignment data and select features (aligned triple) ---
    target_cfg = cfg["xgboost"][args.target]
    data = load_alignment_data(
        cfg["data"]["alignment_pkl"],
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )
    _, feature_names, meta = select_features(
        data, feature_mode=target_cfg["feature_mode"]
    )
    n_features = len(feature_names)
    logger.info("Selected %d features in mode %s", n_features, target_cfg["feature_mode"])

    # --- Locate and load fold artifacts ---
    model_dir = _resolve_model_dir(cfg, args.target, args.model_dir, repo_root)
    folds_dir = model_dir / "folds"
    logger.info("Loading folds from %s", folds_dir)
    fold_artifacts = _load_all_folds(folds_dir)
    logger.info("Loaded %d fold artifacts", len(fold_artifacts))

    # --- Per-fold gain vectors -> (n_folds, n_features) relative matrix ---
    fold_importances = compute_fold_importances(
        fold_artifacts, n_features=n_features
    )
    matrix, holdout_order = _build_relative_importance_matrix(fold_importances)
    logger.info(
        "Built fold importance matrix shape=%s (alpha=%.3f)",
        matrix.shape, args.alpha,
    )

    # --- Compute CIs ---
    feature_ci = compute_feature_level_ci(
        matrix, feature_names, meta, alpha=args.alpha
    )
    lr_ci = compute_group_level_ci(
        matrix, feature_names, meta, group_key="lr_name", alpha=args.alpha
    )
    sender_ci = compute_group_level_ci(
        matrix, feature_names, meta, group_key="ct_L", alpha=args.alpha
    )
    receiver_ci = compute_group_level_ci(
        matrix, feature_names, meta, group_key="ct_R", alpha=args.alpha
    )

    # --- Write outputs ---
    ci_dir = model_dir / "ci"
    ci_dir.mkdir(parents=True, exist_ok=True)

    feature_ci.to_csv(ci_dir / "feature_ci.csv", index=False)
    lr_ci.to_csv(ci_dir / "lr_ci.csv", index=False)
    sender_ci.to_csv(ci_dir / "sender_celltype_ci.csv", index=False)
    receiver_ci.to_csv(ci_dir / "receiver_celltype_ci.csv", index=False)

    # Persist the raw (fold × feature) relative-importance matrix for
    # downstream use (e.g., plotting error bars, per-fold rank plots).
    matrix_df = pd.DataFrame(matrix, index=holdout_order, columns=feature_names)
    matrix_df.index.name = "holdout_region"
    matrix_df.to_csv(ci_dir / "fold_importance_matrix.csv")

    logger.info(
        "Wrote CI outputs to %s (feature=%d rows, lr=%d, sender=%d, receiver=%d)",
        ci_dir, len(feature_ci), len(lr_ci), len(sender_ci), len(receiver_ci),
    )


if __name__ == "__main__":
    main()
