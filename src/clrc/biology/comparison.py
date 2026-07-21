"""SC vs FC feature / network comparison utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Mapping, Optional

import numpy as np
import pandas as pd

from clrc.biology.classification import CELL_CLASS_MAP, classify_lr_pathway

logger = logging.getLogger(__name__)


# =============================================================================
# Printing / reporting helpers
# =============================================================================


def print_section_header(title: str) -> None:
    """Print a boxed section header to stdout."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80 + "\n")


def print_subsection_header(title: str) -> None:
    """Print a subsection header underlined with dashes."""
    print(f"\n{title}")
    print("-" * 80)


def get_top_k_features(
    df: pd.DataFrame,
    value_col: str,
    name_col: str,
    k: int = 20,
) -> List[str]:
    """Return the top-k feature names sorted by ``value_col`` descending.

    Parameters
    ----------
    df
        DataFrame containing both ``name_col`` and ``value_col``.
    value_col
        Numeric column to sort by.
    name_col
        Name column whose top-k values are returned.
    k
        Number of features to return.
    """
    return df.sort_values(value_col, ascending=False).head(k)[name_col].tolist()


def print_features_with_values(
    df: pd.DataFrame,
    features: List[str],
    name_col: str,
    value_col: str,
    modality: str,
) -> None:
    """Print a list of features and their values, labeled by modality."""
    print(f"\n{modality} Top-{len(features)} features:")
    for i, feat in enumerate(features, 1):
        val = df.loc[df[name_col] == feat, value_col].values[0]
        print(f"  {i:2d}. {feat:<70s} {val:.6f}")


# =============================================================================
# Feature-set intersection
# =============================================================================


def analyze_intersection(
    sc_features: List[str],
    fc_features: List[str],
    feature_type: str,
) -> None:
    """Print overlap + Jaccard similarity between two ranked feature lists."""
    intersection = set(sc_features) & set(fc_features)
    print_subsection_header(f"Intersecting {feature_type}")
    if intersection:
        print(f"\nFound {len(intersection)} intersecting features:")
        for i, feat in enumerate(sorted(intersection), 1):
            sc_rank = sc_features.index(feat) + 1
            fc_rank = fc_features.index(feat) + 1
            print(
                f"  {i:2d}. {feat:<70s} (SC rank: {sc_rank:2d}, FC rank: {fc_rank:2d})"
            )
        union = set(sc_features) | set(fc_features)
        jaccard = len(intersection) / len(union) if union else 0
        print(
            f"\nJaccard similarity: {jaccard:.4f} "
            f"({len(intersection)}/{len(union)})"
        )
    else:
        print("\nNo intersecting features found.")


# =============================================================================
# Cell-class interaction analysis
# =============================================================================


def analyze_celltype_class_interactions(
    full_feature_df: pd.DataFrame,
    modality: str = "SC",
    cell_class_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Aggregate feature importance by sender x receiver cell class.

    Parameters
    ----------
    full_feature_df
        Feature importance DataFrame with ``ct_L``, ``ct_R`` and
        ``weighted_mean_gain_rel`` columns.
    modality
        Label used to name the output importance column
        (``importance_{modality.lower()}``). Defaults to ``"SC"``.
    cell_class_map
        Cell type -> cell class mapping. Defaults to ``CELL_CLASS_MAP``.
    """
    if cell_class_map is None:
        cell_class_map = CELL_CLASS_MAP

    df = full_feature_df.copy()
    df["sender_class"] = df["ct_L"].map(cell_class_map).fillna("Other")
    df["receiver_class"] = df["ct_R"].map(cell_class_map).fillna("Other")
    class_interactions = (
        df.groupby(["sender_class", "receiver_class"])["weighted_mean_gain_rel"]
        .sum()
        .reset_index()
    )
    class_interactions.columns = [
        "sender_class",
        "receiver_class",
        f"importance_{modality.lower()}",
    ]
    return class_interactions


def compare_celltype_class_interactions(
    sc_class_df: pd.DataFrame,
    fc_class_df: pd.DataFrame,
    ratio_threshold: float = 1.2,
) -> pd.DataFrame:
    """Compare cell-class interaction importance between SC and FC.

    Parameters
    ----------
    sc_class_df
        Output of ``analyze_celltype_class_interactions(..., modality='SC')``.
    fc_class_df
        Output of ``analyze_celltype_class_interactions(..., modality='FC')``.
    ratio_threshold
        Ratio cutoff (default 1.2) for the SC-biased / FC-biased / Balanced
        category labeling.
    """
    merged = pd.merge(
        sc_class_df,
        fc_class_df,
        on=["sender_class", "receiver_class"],
        how="outer",
    ).fillna(0)

    eps = 1e-10
    merged["ratio_sc_fc"] = (merged["importance_sc"] + eps) / (
        merged["importance_fc"] + eps
    )
    merged["log2_ratio"] = np.log2(merged["ratio_sc_fc"])

    def _cat(ratio: float, thr: float = ratio_threshold) -> str:
        if ratio > thr:
            return "SC-biased"
        elif ratio < 1.0 / thr:
            return "FC-biased"
        return "Balanced"

    merged["category"] = merged["ratio_sc_fc"].apply(_cat)
    return merged.sort_values("ratio_sc_fc", ascending=False)


# =============================================================================
# Network-level comparison
# =============================================================================


def compare_networks(
    G_sc,
    G_fc,
    metrics_sc: pd.DataFrame,
    metrics_fc: pd.DataFrame,
) -> pd.DataFrame:
    """Compare SC and FC networks at the node level.

    Returns a merged DataFrame with ``total_strength`` and ``pagerank`` for
    both modalities plus log2 ratios, sorted by SC/FC strength ratio.

    The ``G_sc``/``G_fc`` arguments are kept for call-site compatibility but
    are not currently used by the comparison itself; callers typically
    supply them alongside the precomputed metrics so downstream code can
    reuse the same graph handles.
    """
    merged = pd.merge(
        metrics_sc[["celltype", "cell_class", "total_strength", "pagerank"]],
        metrics_fc[["celltype", "total_strength", "pagerank"]],
        on="celltype",
        how="outer",
        suffixes=("_sc", "_fc"),
    ).fillna(0)

    eps = 1e-10
    merged["strength_ratio"] = (merged["total_strength_sc"] + eps) / (
        merged["total_strength_fc"] + eps
    )
    merged["pagerank_ratio"] = (merged["pagerank_sc"] + eps) / (
        merged["pagerank_fc"] + eps
    )
    merged["log2_strength_ratio"] = np.log2(merged["strength_ratio"])
    merged["log2_pagerank_ratio"] = np.log2(merged["pagerank_ratio"])
    return merged.sort_values("strength_ratio", ascending=False)


# =============================================================================
# LR pathway class analysis
# =============================================================================


def analyze_lr_pathway_classes(categorized_df: pd.DataFrame) -> pd.DataFrame:
    """Pathway class breakdown for categorized LR interactions.

    Parameters
    ----------
    categorized_df
        Output of ``categorize_features`` with columns ``group_name``,
        ``category``, ``importance_sc``, ``importance_fc`` and
        ``importance_combined``.
    """
    df = categorized_df.copy()
    df["pathway_class"] = df["group_name"].apply(classify_lr_pathway)
    return (
        df.groupby(["pathway_class", "category"])
        .agg(
            {
                "group_name": "count",
                "importance_sc": "sum",
                "importance_fc": "sum",
                "importance_combined": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                "group_name": "n_interactions",
                "importance_sc": "total_importance_sc",
                "importance_fc": "total_importance_fc",
                "importance_combined": "total_importance_combined",
            }
        )
    )


def summarize_pathway_classes(
    categorized_df: pd.DataFrame,
    ratio_threshold: float = 1.2,
) -> pd.DataFrame:
    """Overall pathway class importance summary across SC and FC.

    Parameters
    ----------
    categorized_df
        As in :func:`analyze_lr_pathway_classes`.
    ratio_threshold
        Ratio cutoff for the ``overall_bias`` label (default 1.2).
    """
    df = categorized_df.copy()
    df["pathway_class"] = df["group_name"].apply(classify_lr_pathway)

    summary = (
        df.groupby("pathway_class")
        .agg(
            {
                "group_name": "count",
                "importance_sc": "sum",
                "importance_fc": "sum",
                "importance_combined": "sum",
            }
        )
        .reset_index()
    )
    summary.columns = [
        "pathway_class",
        "n_interactions",
        "importance_sc",
        "importance_fc",
        "importance_combined",
    ]
    eps = 1e-10
    summary["ratio_sc_fc"] = (summary["importance_sc"] + eps) / (
        summary["importance_fc"] + eps
    )
    summary["log2_ratio"] = np.log2(summary["ratio_sc_fc"])

    def _cat(r: float, thr: float = ratio_threshold) -> str:
        if r > thr:
            return "SC-biased"
        elif r < 1.0 / thr:
            return "FC-biased"
        return "Balanced"

    summary["overall_bias"] = summary["ratio_sc_fc"].apply(_cat)
    return summary.sort_values("importance_combined", ascending=False)


# =============================================================================
# Formatted report (SC vs FC) — thin I/O wrapper
# =============================================================================


def _find_file(directory: Path, filename: str) -> Path:
    """Search for *filename* first in ``directory/plots/``, then ``directory/``.

    Some experiment outputs store CSVs under a ``plots/`` subdirectory,
    others directly at the experiment root; this checks both.
    """
    candidate = directory / "plots" / filename
    if candidate.exists():
        return candidate
    candidate = directory / filename
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find {filename} in {directory}/plots/ or {directory}/"
    )


def _load_csv(directory: Path, filename: str) -> pd.DataFrame:
    """Load a CSV from the experiment directory (checking both locations)."""
    return pd.read_csv(_find_file(directory, filename))


def _load_lr_importance(directory: Path) -> pd.DataFrame:
    """Load LR-collapsed importance (relative)."""
    return _load_csv(directory, "importance_by_lr_rel.csv")


def _load_celltype_importance(directory: Path, role: str = "sender") -> pd.DataFrame:
    """Load celltype-aggregated importance (relative).

    Parameters
    ----------
    directory
        Experiment output directory.
    role
        ``"sender"`` or ``"receiver"``.
    """
    if role not in ("sender", "receiver"):
        raise ValueError(f"Invalid role: {role}. Must be 'sender' or 'receiver'.")
    return _load_csv(directory, f"importance_by_{role}_celltype_rel.csv")


def run_formatted_comparisons(
    sc_dir: Path,
    fc_dir: Path,
    top_k: int = 20,
) -> None:
    """Print formatted comparison tables for SC vs FC importance files.

    Reads ``feature_importance_top20_rel.csv``, ``importance_by_lr_rel.csv`` and
    ``importance_by_{sender,receiver}_celltype_rel.csv`` from each modality
    directory and prints top-k overlap tables. Missing files are logged and
    that section is skipped.
    """
    sc_dir = Path(sc_dir)
    fc_dir = Path(fc_dir)

    print_section_header(
        "RAW FEATURE IMPORTANCE (Ligand-Receptor | Sender-Receiver)"
    )
    try:
        sc_raw = _load_csv(sc_dir, "feature_importance_top20_rel.csv")
        fc_raw = _load_csv(fc_dir, "feature_importance_top20_rel.csv")
        sc_feats = get_top_k_features(
            sc_raw, "weighted_mean_gain_rel", "feature_name", top_k
        )
        fc_feats = get_top_k_features(
            fc_raw, "weighted_mean_gain_rel", "feature_name", top_k
        )
        print_features_with_values(
            sc_raw,
            sc_feats,
            "feature_name",
            "weighted_mean_gain_rel",
            "STRUCTURAL CONNECTIVITY",
        )
        print_features_with_values(
            fc_raw,
            fc_feats,
            "feature_name",
            "weighted_mean_gain_rel",
            "FUNCTIONAL CONNECTIVITY",
        )
        analyze_intersection(sc_feats, fc_feats, "raw features")
    except FileNotFoundError as e:
        print(f"  Skipping raw feature comparison: {e}")

    print_section_header("LIGAND-RECEPTOR INTERACTIONS (Celltype-collapsed)")
    try:
        sc_lr = _load_lr_importance(sc_dir)
        fc_lr = _load_lr_importance(fc_dir)
        sc_feats = get_top_k_features(
            sc_lr, "aggregated_importance_rel", "group_name", top_k
        )
        fc_feats = get_top_k_features(
            fc_lr, "aggregated_importance_rel", "group_name", top_k
        )
        print_features_with_values(
            sc_lr,
            sc_feats,
            "group_name",
            "aggregated_importance_rel",
            "STRUCTURAL CONNECTIVITY",
        )
        print_features_with_values(
            fc_lr,
            fc_feats,
            "group_name",
            "aggregated_importance_rel",
            "FUNCTIONAL CONNECTIVITY",
        )
        analyze_intersection(sc_feats, fc_feats, "L-R interactions")
    except FileNotFoundError as e:
        print(f"  Skipping LR comparison: {e}")

    for role in ("sender", "receiver"):
        print_section_header(f"{role.upper()} CELLTYPES (LR-collapsed)")
        try:
            sc_ct = _load_celltype_importance(sc_dir, role)
            fc_ct = _load_celltype_importance(fc_dir, role)
            sc_feats = get_top_k_features(
                sc_ct, "aggregated_importance_rel", "group_name", top_k
            )
            fc_feats = get_top_k_features(
                fc_ct, "aggregated_importance_rel", "group_name", top_k
            )
            print_features_with_values(
                sc_ct,
                sc_feats,
                "group_name",
                "aggregated_importance_rel",
                "STRUCTURAL CONNECTIVITY",
            )
            print_features_with_values(
                fc_ct,
                fc_feats,
                "group_name",
                "aggregated_importance_rel",
                "FUNCTIONAL CONNECTIVITY",
            )
            analyze_intersection(sc_feats, fc_feats, f"{role} celltypes")
        except FileNotFoundError as e:
            print(f"  Skipping {role} celltype comparison: {e}")
