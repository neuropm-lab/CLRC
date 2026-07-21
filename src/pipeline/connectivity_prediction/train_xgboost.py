#!/usr/bin/env python3
"""Train XGBoost LOBO model (SC or FC) using clrc pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clrc.core.io import load_alignment_data, load_yaml_config, save_pickle, safe_filename, timestamp
from clrc.core.metrics import rmse, mae
from clrc.core.logging import setup_logging
from clrc.prediction.lobo import infer_regions, precompute_fold_masks, iter_lobo_folds, select_features
from clrc.prediction.xgboost import train_predict_xgb
from clrc.core.io import stable_hash_int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOBO XGBoost training. For HPO, use src/pipeline/connectivity_prediction/hpo.py.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", choices=["sc", "fc"], required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["cci_only", "distance_only", "cci_distance"],
        default=None,
        help=(
            "Override feature_mode from YAML (target_cfg['feature_mode']). "
            "When set, the output directory name also gets a feature_mode "
            "suffix so distance_only / cci_distance runs do not collide "
            "with the default cci_only outputs."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    target_cfg = cfg["xgboost"][args.target]
    if args.feature_mode is not None:
        target_cfg["feature_mode"] = args.feature_mode
    out_base = Path(cfg["output"]["base_dir"])

    logger = setup_logging("train_xgboost", out_base)

    # Load data
    data = load_alignment_data(
        target_cfg.get("alignment_pkl", cfg["data"]["alignment_pkl"]),
        version=target_cfg["version"],
        target_scale=target_cfg.get("target_scale", 1.0),
    )

    X_selected, feature_names, meta = select_features(
        data, feature_mode=target_cfg["feature_mode"]
    )
    # Replace data.X / feature_names / meta with the selected triple so all
    # downstream code (hpo, lobo, xgboost) sees consistent arrays.
    from clrc.core.types import AlignmentData
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

    # --- Training mode ---
    from clrc.core.io import find_repo_root
    params_path = Path(target_cfg["params_json"])
    if not params_path.is_absolute():
        params_path = find_repo_root() / params_path
    with params_path.open() as f:
        params_blob = json.load(f)
    best_params_xgb = params_blob["best_params_xgb"]

    # Include feature_mode in the output dir name so non-default modes
    # (distance_only, cci_distance) do not collide with cci_only outputs.
    # cci_only keeps the historical naming for backward compatibility.
    feature_mode = target_cfg["feature_mode"]
    feature_mode_suffix = "" if feature_mode == "cci_only" else f"_{feature_mode}"
    exp_dir = args.output_dir or (
        out_base / "models" / args.target /
        f"{timestamp()}_{target_cfg['version']}_{safe_filename(target_cfg['metric'])}{feature_mode_suffix}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = exp_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # LOBO evaluation
    metric_names = data.metric_names
    j = metric_names.index(target_cfg["metric"])
    SC = data.SC_voxel if target_cfg["version"] == "voxel" else data.SC_naive
    y_all = SC[:, j].astype(float)

    regions_all = infer_regions(data.edge_table)
    fold_masks = precompute_fold_masks(data.edge_table, regions_all)
    logger.info("Precomputed %d fold masks.", len(fold_masks))

    fold_artifacts = []
    for fold in iter_lobo_folds(
        data.X, data.edge_table, y_all, fold_masks,
        eps=0.0,
        y_transform=cfg["xgboost"]["y_transform"],
        data_type=target_cfg["data_type"],
        include_edge_tables=True,
        regions=regions_all,
    ):
        (holdout_region, X_train, y_train_t, X_test, y_test_t,
         _y_train_raw, y_test_raw, ecdf, edge_table_test, test_idx) = fold

        split_seed = int(stable_hash_int(f"{cfg['xgboost']['seed']}_{holdout_region}"))
        y_pred, best_iter, model_raw = train_predict_xgb(
            X_train, y_train_t, X_test,
            params=best_params_xgb,
            num_boost_round=cfg["xgboost"]["max_boost_rounds"],
            split_seed=split_seed,
            booster_seed=cfg["xgboost"]["seed"],
            device=cfg["xgboost"]["device"],
            valid_fraction=cfg["xgboost"]["valid_fraction"],
            early_stopping_rounds=cfg["xgboost"]["early_stopping_rounds"],
        )

        from clrc.core.types import FoldArtifact
        fa = FoldArtifact(
            holdout_region=holdout_region,
            metric=target_cfg["metric"],
            version=target_cfg["version"],
            n_train=int(X_train.shape[0]),
            n_test=int(X_test.shape[0]),
            n_features=int(X_train.shape[1]),
            eps=0.0,
            y_transform=cfg["xgboost"]["y_transform"],
            params=dict(best_params_xgb),
            best_iteration=best_iter,
            model_raw=model_raw,
            ecdf=ecdf,
            test_idx=test_idx,
            y_test_raw=y_test_raw,
            y_test_t=y_test_t,
            y_pred=y_pred,
            fold_rmse=rmse(y_test_t, y_pred),
            fold_mae=mae(y_test_t, y_pred),
            edge_table_test=edge_table_test,
            eval_metrics=list(best_params_xgb.get("eval_metric", [])),
        )
        fold_artifacts.append(fa)
        save_pickle(fa, folds_dir / f"{safe_filename(holdout_region)}.pkl")
        logger.info("[%s] n_test=%d, best_iter=%d, RMSE=%.6f, MAE=%.6f",
                    holdout_region, fa.n_test, best_iter, fa.fold_rmse, fa.fold_mae)

    # Summary
    fold_summary = pd.DataFrame({
        "holdout_region": [fa.holdout_region for fa in fold_artifacts],
        "n_train": [fa.n_train for fa in fold_artifacts],
        "n_test": [fa.n_test for fa in fold_artifacts],
        "fold_rmse": [fa.fold_rmse for fa in fold_artifacts],
        "fold_mae": [fa.fold_mae for fa in fold_artifacts],
        "best_iteration": [fa.best_iteration for fa in fold_artifacts],
    }).sort_values("holdout_region")
    fold_summary.to_csv(exp_dir / "full_lobo_fold_summary.csv", index=False)

    logger.info("Training complete. %d folds saved to %s", len(fold_artifacts), exp_dir)


if __name__ == "__main__":
    main()
