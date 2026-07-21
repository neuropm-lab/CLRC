#!/usr/bin/env python3
"""Biological interpretation: categorization, enrichment, network. Calls clrc modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from clrc.biology.network import (
    build_celltype_network,
    compute_network_metrics,
    hub_permutation_test,
    hubs_contingency_test,
    identify_hubs,
)
from clrc.core.io import load_yaml_config
from clrc.core.logging import setup_logging
from clrc.prediction.interpretation import categorize_features, load_lr_importance


def _load_full_feature_importance(model_dir: Path) -> pd.DataFrame:
    """Load the per-feature importance table written by aggregate_importance.py.

    The file has columns ``ct_L``, ``ct_R``, ``weighted_mean_gain_rel`` (and
    others) which is the exact shape expected by
    :func:`clrc.biology.network.build_celltype_network`.
    """
    candidate = model_dir / "plots" / "feature_importance_weighted_gain.csv"
    if not candidate.exists():
        candidate = model_dir / "feature_importance_weighted_gain.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"feature_importance_weighted_gain.csv not found under {model_dir}/"
            " (checked plots/ and model dir root)"
        )
    return pd.read_csv(candidate)


def _run_hub_analysis(
    full_feature_df: pd.DataFrame,
    *,
    edge_percentile_threshold: float = 80.0,
    hub_percentile_threshold: float = 80.0,
):
    """Build network, compute metrics, and identify hubs for a single model.

    Pipeline: ``build_celltype_network`` -> ``compute_network_metrics`` ->
    ``identify_hubs`` with 80th-percentile thresholds on both
    ``total_strength`` and ``pagerank``.
    """
    G = build_celltype_network(
        full_feature_df, edge_percentile_threshold=edge_percentile_threshold
    )
    # Annotate cell_type so hub_permutation_test can derive cell_class via
    # the classification CELL_CLASS_MAP (node names are supercluster names).
    for node, data in G.nodes(data=True):
        data["cell_type"] = node
    metrics = compute_network_metrics(G)
    hubs = identify_hubs(metrics, percentile_threshold=hub_percentile_threshold)
    return G, metrics, hubs


def _find_model_with_plots(base: Path, target_cfg: dict, repo_root: Path) -> Path:
    """Locate model dir containing plots/, checking out/ and data/ locations."""
    # Check out/ base first
    if (base / "plots").exists():
        return base
    candidates = sorted(base.glob("*/plots"), reverse=True)
    if candidates:
        return candidates[0].parent
    # Fallback: data/ legacy dirs
    legacy_base = repo_root / "data" / "expanded_ABC" / "model_GBM"
    if legacy_base.exists():
        matches = sorted(
            legacy_base.glob(f"*_{target_cfg['version']}_{target_cfg['metric']}*"),
            reverse=True,
        )
        for m in matches:
            if (m / "plots").exists():
                return m
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Biological interpretation pipeline")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    logger = setup_logging("cross_target_biology", out_base)

    interp_dir = out_base / "interpretation"
    interp_dir.mkdir(parents=True, exist_ok=True)

    from clrc.core.io import find_repo_root
    repo_root = find_repo_root()

    sc_dir = _find_model_with_plots(
        out_base / "models" / "sc", cfg["xgboost"]["sc"], repo_root,
    )
    fc_dir = _find_model_with_plots(
        out_base / "models" / "fc", cfg["xgboost"]["fc"], repo_root,
    )

    sc_lr = load_lr_importance(sc_dir)
    fc_lr = load_lr_importance(fc_dir)
    logger.info("Loaded SC LR importance from %s (%d rows)", sc_dir, len(sc_lr))
    logger.info("Loaded FC LR importance from %s (%d rows)", fc_dir, len(fc_lr))

    # Categorize (no top_n_lr cutoff)
    categorized = categorize_features(sc_lr, fc_lr)
    categorized.to_csv(interp_dir / "feature_categories.csv", index=False)

    n_sc = (categorized["category"] == "SC-biased").sum()
    n_fc = (categorized["category"] == "FC-biased").sum()
    n_bal = (categorized["category"] == "Balanced").sum()
    logger.info(
        "Categorized %d LR pairs: %d SC-biased, %d FC-biased, %d Balanced",
        len(categorized), n_sc, n_fc, n_bal,
    )

    # -----------------------------------------------------------------
    # Hub permutation + SC-vs-FC contingency analysis
    # -----------------------------------------------------------------
    # Tests whether SC hub status is distributed across several cell classes
    # whereas FC is dominated by inhibitory neurons, via two complementary
    # tests:
    #   (1) ``hub_permutation_test`` run separately on the SC and FC
    #       HPO-best networks, reporting a per-class p-value under a
    #       class-label-shuffling null.
    #   (2) ``hubs_contingency_test`` directly comparing the SC vs FC hub
    #       supercluster distributions as a 2 × n_classes contingency
    #       table (chi-squared + Fisher exact when 2 × 2).
    # Hub definition: 80th-percentile on ``total_strength`` OR ``pagerank``
    # (see ``clrc.biology.network.identify_hubs``).
    networks_dir = interp_dir / "networks"
    networks_dir.mkdir(parents=True, exist_ok=True)

    try:
        sc_full = _load_full_feature_importance(sc_dir)
        fc_full = _load_full_feature_importance(fc_dir)
    except FileNotFoundError as e:
        logger.warning(
            "Skipping hub analysis: %s (run aggregate_importance.py first)", e
        )
    else:
        G_sc, metrics_sc, hubs_sc = _run_hub_analysis(sc_full)
        G_fc, metrics_fc, hubs_fc = _run_hub_analysis(fc_full)

        metrics_sc.to_csv(networks_dir / "sc_network_metrics.csv", index=False)
        metrics_fc.to_csv(networks_dir / "fc_network_metrics.csv", index=False)
        hubs_sc.to_csv(networks_dir / "hubs_sc.csv", index=False)
        hubs_fc.to_csv(networks_dir / "hubs_fc.csv", index=False)
        logger.info(
            "Identified %d SC hubs and %d FC hubs (80th pct total_strength OR"
            " pagerank)", len(hubs_sc), len(hubs_fc),
        )

        # (1) Per-model permutation test: shuffle cell-class labels under
        #     fixed topology and recompute per-class hub fraction. 1000
        #     permutations; random_state=42 for reproducibility.
        hubs_sc_for_perm = hubs_sc.rename(columns={"celltype": "group_name"})
        hubs_fc_for_perm = hubs_fc.rename(columns={"celltype": "group_name"})
        perm_sc = hub_permutation_test(
            G_sc, hubs_sc_for_perm, n_permutations=1000, random_state=42
        )
        perm_fc = hub_permutation_test(
            G_fc, hubs_fc_for_perm, n_permutations=1000, random_state=42
        )
        perm_sc.to_csv(networks_dir / "hub_permutation_sc.csv", index=False)
        perm_fc.to_csv(networks_dir / "hub_permutation_fc.csv", index=False)
        logger.info(
            "hub_permutation_test: SC rows=%d, FC rows=%d",
            len(perm_sc), len(perm_fc),
        )

        # (2) Direct SC-vs-FC contingency test at 31-supercluster
        #     granularity (not collapsed to 4 broad classes).
        #     ``identify_hubs`` output holds supercluster names in the
        #     ``celltype`` column, so we pass that as the class column.
        ct_result = hubs_contingency_test(
            hubs_sc, hubs_fc, cell_class_column="celltype"
        )
        ct_result["contingency_table"].to_csv(
            networks_dir / "hub_sc_vs_fc_contingency.csv",
            index=True,
        )
        chi2_summary = {
            "chi2": ct_result["chi2"],
            "p_chi2": ct_result["p_chi2"],
            "dof": ct_result["dof"],
            "p_fisher": ct_result["p_fisher"],
            "n_classes": ct_result["contingency_table"].shape[1],
        }
        with open(networks_dir / "hub_sc_vs_fc_chi2.json", "w") as f:
            json.dump(chi2_summary, f, indent=2)
        logger.info(
            "SC-vs-FC contingency: chi2=%.3f, dof=%d, p_chi2=%.3g, "
            "p_fisher=%s, n_classes=%d",
            ct_result["chi2"], ct_result["dof"], ct_result["p_chi2"],
            (f"{ct_result['p_fisher']:.3g}"
             if ct_result["p_fisher"] is not None else "NA (table not 2x2)"),
            chi2_summary["n_classes"],
        )


if __name__ == "__main__":
    main()
