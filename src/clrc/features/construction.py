"""CCI feature construction: edge table, vectorization, region mapping."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Edge table (vectorized, column-major to match R's as.vector())
# ---------------------------------------------------------------------------

def build_edge_table(regions: List[str]) -> pd.DataFrame:
    """Build (src, tgt, edge_idx) table for an n x n matrix, column-major order.

    Column-major means target (column) varies slowly and source (row) varies
    fast, matching R's ``as.vector()`` behaviour.
    """
    n = len(regions)
    src_idx = np.tile(np.arange(n), n)     # row varies fast
    tgt_idx = np.repeat(np.arange(n), n)   # column varies slow
    return pd.DataFrame(
        {
            "edge_idx": np.arange(n * n),
            "src_region": [regions[i] for i in src_idx],
            "tgt_region": [regions[i] for i in tgt_idx],
        }
    )


# ---------------------------------------------------------------------------
#  SC / FC vectorization
# ---------------------------------------------------------------------------

def vectorize_sc_block(
    sc_3d: np.ndarray, metric_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Vectorize (n_metrics, n, n) SC/FC block to (n*n, n_metrics), column-major."""
    if sc_3d.ndim == 2:
        n1, n2 = sc_3d.shape
        if n1 == n2:
            sc_3d = np.expand_dims(sc_3d, 0)
    if sc_3d.ndim != 3:
        raise ValueError(f"Expected (n_metrics, n, n), got {sc_3d.shape}")
    n_metrics, n1, n2 = sc_3d.shape
    if n1 != n2:
        raise ValueError("SC matrices must be square.")
    if len(metric_names) != n_metrics:
        raise ValueError("metric_names length does not match SC first dim.")

    SC_mat = np.empty((n1 * n2, n_metrics), dtype=float)
    for m in range(n_metrics):
        SC_mat[:, m] = sc_3d[m].reshape(-1, order="F")  # match R
    return SC_mat, metric_names


# ---------------------------------------------------------------------------
#  Group name parsing
# ---------------------------------------------------------------------------

def parse_group_names(
    group_names: List[str],
) -> Tuple[
    List[str],                          # node_regions (len 2133)
    List[str],                          # node_celltypes (len 2133)
    Dict[Tuple[str, str], int],         # (region, ct) -> node_index
    List[str],                          # unique sorted regions
    List[str],                          # unique sorted celltypes
]:
    """Parse 'Region::CellType' labels from NeuronChat H5 group_names."""
    node_regions: List[str] = []
    node_cts: List[str] = []
    node_lookup: Dict[Tuple[str, str], int] = {}

    for idx, lab in enumerate(group_names):
        if "::" not in lab:
            raise ValueError(f"Label '{lab}' does not look like 'Region::CellType'")
        region, ct = lab.split("::", 1)
        region = region.strip()
        ct = ct.strip()
        node_regions.append(region)
        node_cts.append(ct)
        node_lookup[(region, ct)] = idx

    regions_uniq = sorted(set(node_regions))
    cts_uniq = sorted(set(node_cts))
    return node_regions, node_cts, node_lookup, regions_uniq, cts_uniq


def build_ct_region_index(
    ct: str,
    regions_109: List[str],
    node_lookup: Dict[Tuple[str, str], int],
) -> np.ndarray:
    """For a celltype, return int array mapping region index → node index (-1 if absent)."""
    out = np.full(len(regions_109), -1, dtype=np.intp)
    for i, reg in enumerate(regions_109):
        idx = node_lookup.get((reg, ct), -1)
        out[i] = idx
    return out


# ---------------------------------------------------------------------------
#  109 → 101 ABC restriction
# ---------------------------------------------------------------------------

def restrict_to_ABC(
    regions_109: List[str],
    ABC_regions_struct: List[str],
    region_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build flat index map from CCI-region space to ABC regions.

    Parameters
    ----------
    regions_109 : list[str]
        All CCI regions (e.g. 109 for the ABC Atlas).
    ABC_regions_struct : list[str]
        ABC regions from the structural connectivity H5.
    region_aliases : dict[str, str], optional
        Remap structural region names to CCI names (e.g. {"A24": "ACC"}).
        Applied before matching. Defaults to {"A24": "ACC"} if None.

    Returns
    -------
    idx_abc_in_109 : int array
        Position of each ABC region in the CCI region list.
    flat_idx_map : int array
        Column-major CCI→ABC flat index mapping.
    ABC_regions_cci : list[str]
        ABC region names after alias remapping.
    """
    if region_aliases is None:
        region_aliases = {"A24": "ACC"}  # ABC Atlas default

    region_to_idx_109 = {r: i for i, r in enumerate(regions_109)}
    ABC_regions_cci = [region_aliases.get(r, r) for r in ABC_regions_struct]

    missing = [r for r in ABC_regions_cci if r not in region_to_idx_109]
    if missing:
        raise KeyError(f"ABC regions not found in CCI regions_109: {missing}")

    idx_abc_in_109 = np.array(
        [region_to_idx_109[r] for r in ABC_regions_cci], dtype=np.intp
    )

    n_abc = len(ABC_regions_struct)
    n_109 = len(regions_109)
    flat_idx_map = np.empty(n_abc * n_abc, dtype=np.intp)
    pos = 0
    for j_abc in range(n_abc):           # column-major: column outer
        j_109 = idx_abc_in_109[j_abc]
        for i_abc in range(n_abc):       # row inner
            i_109 = idx_abc_in_109[i_abc]
            flat_idx_map[pos] = i_109 + n_109 * j_109
            pos += 1

    logger.info(
        "ABC mapping: %d -> %d regions, flat_idx_map length %d",
        n_109, n_abc, len(flat_idx_map),
    )
    return idx_abc_in_109, flat_idx_map, ABC_regions_cci
