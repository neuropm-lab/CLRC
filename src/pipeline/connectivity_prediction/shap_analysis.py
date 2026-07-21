#!/usr/bin/env python3
"""SHAP analysis for XGBoost LOBO models. Calls clrc.prediction.shap."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from clrc.core.io import load_alignment_data, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.prediction.lobo import select_features
from clrc.prediction.shap import (
    analyze_feature_directionality,
    booster_from_fold,
    load_fold,
    run_shap_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP analysis for LOBO models")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", choices=["sc", "fc"], required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    target_cfg = cfg["xgboost"][args.target]
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("shap_analysis", out_base)

    # Load data for feature names
    data = load_alignment_data(
        cfg["data"]["alignment_pkl"],
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )
    _, feature_names, _ = select_features(data, feature_mode=target_cfg["feature_mode"])

    # Find model directory
    model_dir = args.model_dir or (out_base / "models" / args.target)
    folds_dir = model_dir / "folds"
    if not folds_dir.exists():
        # Try to find the latest timestamped directory
        candidates = sorted(model_dir.glob("**/folds"), reverse=True)
        if candidates:
            folds_dir = candidates[0]
    logger.info("Loading folds from %s", folds_dir)

    # Load all fold artifacts and run SHAP
    shap_dir = model_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    fold_files = sorted(folds_dir.glob("*.pkl"))
    logger.info("Found %d fold files", len(fold_files))

    all_shap_values = []
    all_X_samples = []

    for fold_path in fold_files:
        fold_obj = load_fold(fold_path)
        if fold_obj is None:
            continue

        booster = booster_from_fold(fold_obj)
        shap_vals, _, X_sample = run_shap_analysis(
            booster, data.X, feature_names, sample_size=500, seed=42
        )
        all_shap_values.append(shap_vals)
        all_X_samples.append(X_sample)

    if not all_shap_values:
        logger.error("No SHAP values computed.")
        return

    # Combine across folds
    combined_shap = np.mean(all_shap_values, axis=0)
    combined_X = all_X_samples[0]  # Same sample for all folds

    # Directionality analysis
    directionality_df = analyze_feature_directionality(
        combined_shap, combined_X, feature_names
    )
    directionality_df.to_csv(shap_dir / "shap_directionality.csv", index=False)

    logger.info("SHAP analysis complete. Results in %s", shap_dir)


if __name__ == "__main__":
    main()
