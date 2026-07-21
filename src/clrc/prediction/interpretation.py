"""Feature categorization: SC-biased / FC-biased / Balanced via importance ratio."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_csv(directory: Path, filename: str) -> pd.DataFrame:
    path = directory / "plots" / filename
    if not path.exists():
        path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {directory / 'plots' / filename}")
    return pd.read_csv(path)


def load_lr_importance(directory: Path) -> pd.DataFrame:
    """Load LR-collapsed importance (relative)."""
    return _load_csv(directory, "importance_by_lr_rel.csv")


def load_celltype_importance(directory: Path, role: str = "sender") -> pd.DataFrame:
    """Load celltype-aggregated importance (relative).

    Parameters
    ----------
    role : str
        ``"sender"`` or ``"receiver"``.
    """
    if role not in ("sender", "receiver"):
        raise ValueError(f"Invalid role: {role}. Must be 'sender' or 'receiver'.")
    return _load_csv(directory, f"importance_by_{role}_celltype_rel.csv")


def categorize_features(
    sc_df: pd.DataFrame,
    fc_df: pd.DataFrame,
    ratio_threshold: float = 1.2,
    min_importance: float = 0.0,
    name_col: str = "group_name",
    value_col: str = "aggregated_importance_rel",
) -> pd.DataFrame:
    """Categorize L-R interactions as SC-biased / FC-biased / Balanced.

    All LR pairs with non-zero combined importance are categorized (no
    arbitrary top-N cutoff). Use *min_importance* to filter noise if needed.

    Parameters
    ----------
    sc_df, fc_df : DataFrame
        LR importance tables for SC and FC models.
    ratio_threshold : float
        Ratio above which a feature is SC-biased (below 1/threshold → FC-biased).
    min_importance : float
        Minimum combined importance to include (default 0.0 = keep all non-zero).
    """
    sc_sub = sc_df[[name_col, value_col]].copy()
    sc_sub.columns = [name_col, "importance_sc"]

    fc_sub = fc_df[[name_col, value_col]].copy()
    fc_sub.columns = [name_col, "importance_fc"]

    merged = pd.merge(sc_sub, fc_sub, on=name_col, how="outer").fillna(0)
    merged["importance_combined"] = merged["importance_sc"] + merged["importance_fc"]
    merged = merged[merged["importance_combined"] > min_importance]
    merged = merged.sort_values("importance_combined", ascending=False)

    eps = 1e-10
    merged["ratio_sc_fc"] = (merged["importance_sc"] + eps) / (
        merged["importance_fc"] + eps
    )
    merged["log2_ratio"] = np.log2(merged["ratio_sc_fc"])

    def _cat(ratio: float) -> str:
        if ratio > ratio_threshold:
            return "SC-biased"
        elif ratio < 1.0 / ratio_threshold:
            return "FC-biased"
        return "Balanced"

    merged["category"] = merged["ratio_sc_fc"].apply(_cat)
    merged["rank_sc"] = merged["importance_sc"].rank(ascending=False).astype(int)
    merged["rank_fc"] = merged["importance_fc"].rank(ascending=False).astype(int)
    merged["rank_combined"] = (
        merged["importance_combined"].rank(ascending=False).astype(int)
    )
    return merged.sort_values("importance_combined", ascending=False).reset_index(
        drop=True
    )
