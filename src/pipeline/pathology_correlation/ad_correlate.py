#!/usr/bin/env python3
"""Partial Spearman correlation analysis for AD features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from clrc.core.io import load_pickle, load_yaml_config, save_pickle
from clrc.core.logging import setup_logging
from clrc.ad.correlation import partial_spearman_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial Spearman AD correlation analysis")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--aggregation-pkl", type=Path, default=None,
        help="Override path to aggregation pickle",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("ad_correlate", out_base)

    # Load aggregation output
    agg_path = args.aggregation_pkl or (out_base / "aggregation" / "ad_aggregation.pkl")
    agg = load_pickle(agg_path)

    X_collapsed = agg["X_collapsed"]
    X_region = agg["X_region"]
    feature_names_collapsed = agg["feature_names_collapsed"]
    feature_names_region = agg["feature_names_region"]
    region_names = agg["region_names"]
    Y_clinical = agg["Y_clinical"]
    covariates_arr = agg["covariates_arr"]
    clinical_vars = agg["clinical_vars"]
    subject_ids = agg["subject_ids"]

    n_subjects = len(subject_ids)
    logger.info("Loaded aggregation: %d subjects", n_subjects)

    corr_dir = out_base / "correlations"
    corr_dir.mkdir(parents=True, exist_ok=True)

    # --- Region-collapsed correlations ---
    logger.info("Computing region-collapsed correlations...")
    collapsed_results = {}

    for var_idx, var_name in enumerate(clinical_vars):
        y = Y_clinical[:, var_idx]
        corrs, pvals = partial_spearman_batch(X_collapsed, y, covariates_arr)
        collapsed_results[var_name] = {"correlations": corrs, "pvalues": pvals}

        df = pd.DataFrame({
            "feature": feature_names_collapsed,
            "correlation": corrs,
            "pvalue": pvals,
        })
        df.to_csv(corr_dir / f"collapsed_{var_name}.csv", index=False)

        valid = ~np.isnan(corrs)
        if valid.sum() > 0:
            top_idx = np.argsort(np.abs(corrs[valid]))[-5:][::-1]
            logger.info("  %s top |r|:", var_name)
            for idx in top_idx:
                orig = np.where(valid)[0][idx]
                logger.info(
                    "    %s: r=%.3f",
                    feature_names_collapsed[orig][:60], corrs[orig],
                )

    # --- Region-specific correlations ---
    logger.info("Computing region-specific correlations...")
    region_results = {}

    for var_idx, var_name in enumerate(clinical_vars):
        y = Y_clinical[:, var_idx]
        var_region_results = {}

        for reg_idx, region_name in enumerate(region_names):
            X_reg = X_region[:, reg_idx, :]
            corrs, pvals = partial_spearman_batch(X_reg, y, covariates_arr)
            var_region_results[region_name] = {
                "correlations": corrs,
                "pvalues": pvals,
            }

            df = pd.DataFrame({
                "feature": feature_names_region,
                "correlation": corrs,
                "pvalue": pvals,
            })
            df.to_csv(corr_dir / f"region_{var_name}_{region_name}.csv", index=False)

        region_results[var_name] = var_region_results
        logger.info("  %s: computed for %d regions", var_name, len(region_names))

    # Save combined results
    save_pickle(
        {
            "collapsed_results": collapsed_results,
            "region_results": region_results,
            "feature_names_collapsed": feature_names_collapsed,
            "feature_names_region": feature_names_region,
            "region_names": region_names,
            "clinical_vars": clinical_vars,
            "subject_ids": subject_ids,
        },
        corr_dir / "all_correlations.pkl",
    )

    logger.info("Correlations complete. Results saved to %s", corr_dir)


if __name__ == "__main__":
    main()
