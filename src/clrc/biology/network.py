"""Cell type communication network analysis (networkx)."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from scipy import stats

from clrc.biology.classification import CELL_CLASS_MAP

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    nx = None  # ty: ignore[invalid-assignment]
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


def build_celltype_network(
    full_feature_df: pd.DataFrame,
    edge_percentile_threshold: float = 80.0,
    cell_class_map: Optional[Mapping[str, str]] = None,
) -> "nx.DiGraph":
    """Build a directed cell type communication network.

    Parameters
    ----------
    full_feature_df : DataFrame
        Must have columns: ct_L, ct_R, weighted_mean_gain_rel.
    edge_percentile_threshold : float
        Keep edges above this percentile of summed weight (default 80).
    cell_class_map : mapping, optional
        Cell type → class mapping (default: ABC Atlas CELL_CLASS_MAP).
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for network analysis.")
    if cell_class_map is None:
        cell_class_map = CELL_CLASS_MAP

    edge_weights = (
        full_feature_df.groupby(["ct_L", "ct_R"])["weighted_mean_gain_rel"]
        .sum()
        .reset_index()
    )
    edge_weights.columns = ["sender", "receiver", "weight"]

    threshold = np.percentile(
        edge_weights["weight"], 100 - edge_percentile_threshold
    )
    edge_weights = edge_weights[edge_weights["weight"] >= threshold]

    G = nx.DiGraph()
    all_celltypes = set(edge_weights["sender"]) | set(edge_weights["receiver"])
    for ct in all_celltypes:
        G.add_node(ct, cell_class=cell_class_map.get(ct, "Other"))
    for _, row in edge_weights.iterrows():
        G.add_edge(row["sender"], row["receiver"], weight=row["weight"])
    return G


def compute_network_metrics(G: "nx.DiGraph") -> pd.DataFrame:
    """Compute node-level network metrics (degree, betweenness, pagerank)."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for network analysis.")

    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))

    try:
        betweenness = nx.betweenness_centrality(G, weight="weight")
    except Exception:
        betweenness = {n: 0 for n in G.nodes()}
    try:
        pagerank = nx.pagerank(G, weight="weight")
    except Exception:
        pagerank = {n: 1.0 / len(G.nodes()) for n in G.nodes()}

    metrics = []
    for node in G.nodes():
        cell_class = G.nodes[node].get("cell_class", "Other")
        metrics.append(
            {
                "celltype": node,
                "cell_class": cell_class,
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
                "weighted_in_strength": in_degree.get(node, 0),
                "weighted_out_strength": out_degree.get(node, 0),
                "total_strength": in_degree.get(node, 0) + out_degree.get(node, 0),
                "betweenness_centrality": betweenness.get(node, 0),
                "pagerank": pagerank.get(node, 0),
            }
        )
    return pd.DataFrame(metrics).sort_values("total_strength", ascending=False)


def compute_global_network_metrics(G: "nx.DiGraph") -> Dict[str, float]:
    """Compute graph-level metrics."""
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for network analysis.")
    metrics: Dict[str, float] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
    }
    try:
        G_und = G.to_undirected()
        metrics["avg_clustering"] = nx.average_clustering(G_und, weight="weight")
    except Exception:
        metrics["avg_clustering"] = 0.0
    return metrics


def detect_communities(G: "nx.DiGraph") -> Dict[str, int]:
    """Detect communities using Louvain (networkx built-in)."""
    if not HAS_NETWORKX:
        return {}
    G_und = G.to_undirected()
    try:
        communities = nx.community.louvain_communities(G_und, weight="weight")
        partition: Dict[str, int] = {}
        for cid, nodes in enumerate(communities):
            for node in nodes:
                partition[node] = cid
        return partition
    except Exception as e:
        warnings.warn(f"Community detection failed: {e}")
        return {}


def identify_hubs(
    metrics_df: pd.DataFrame,
    percentile_threshold: float = 80.0,
) -> pd.DataFrame:
    """Identify hub cell types based on total_strength and pagerank."""
    str_thr = np.percentile(metrics_df["total_strength"], percentile_threshold)
    pr_thr = np.percentile(metrics_df["pagerank"], percentile_threshold)

    hubs = metrics_df[
        (metrics_df["total_strength"] >= str_thr)
        | (metrics_df["pagerank"] >= pr_thr)
    ].copy()
    hubs["is_strength_hub"] = hubs["total_strength"] >= str_thr
    hubs["is_pagerank_hub"] = hubs["pagerank"] >= pr_thr
    return hubs


def hub_permutation_test(
    network: "nx.DiGraph",
    hubs_df: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    random_state: Optional[int] = 42,
    cell_class_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Permutation test for hub concentration in cell classes.

    Null model: fix network topology, randomly shuffle cell-class labels
    across nodes, recompute which nodes would be labeled as hubs under the
    same threshold, then compute the fraction of hubs in each class. The
    observed hub-class distribution is compared against the null distribution
    to get a two-sided p-value per class.

    Because hub identification is topology-derived (``total_strength`` /
    ``pagerank``), shuffling node *class* labels does not change which nodes
    are hubs — only the class-label distribution within the hub set is
    randomized. Equivalently, this asks: given ``k`` observed hubs drawn from
    ``N`` nodes with an observed class distribution, is the observed
    per-class hub fraction more extreme than random?

    Parameters
    ----------
    network
        Directed graph. Nodes should carry a ``cell_type`` attribute that is
        mapped through ``cell_class_map`` to derive a cell class.
    hubs_df
        Output of ``identify_hubs()`` — DataFrame with a ``group_name``
        column listing the hub node names.
    n_permutations
        Number of permutation iterations (default 1000).
    random_state
        Seed for reproducibility. Default 42.
    cell_class_map
        Mapping ``cell_type -> cell_class``. Falls back to
        ``clrc.biology.classification.CELL_CLASS_MAP`` when ``None``.

    Returns
    -------
    DataFrame with columns
        ``cell_class``, ``observed_hub_frac``, ``null_mean``, ``null_std``,
        ``pval_two_sided`` (one row per class that appears in the observed
        node-class distribution).
    """
    if not HAS_NETWORKX:
        raise ImportError("NetworkX is required for hub_permutation_test.")
    if cell_class_map is None:
        cell_class_map = CELL_CLASS_MAP

    nodes = list(network.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return pd.DataFrame(
            columns=[
                "cell_class",
                "observed_hub_frac",
                "null_mean",
                "null_std",
                "pval_two_sided",
            ]
        )

    # Derive node -> class vector via ``cell_type`` attribute.
    node_classes = np.array(
        [
            cell_class_map.get(network.nodes[n].get("cell_type", n), "Other")
            for n in nodes
        ]
    )

    # Observed hub mask (over the full node list).
    hub_names = set(hubs_df["group_name"].astype(str).tolist())
    hub_mask = np.array([str(n) in hub_names for n in nodes])
    n_hubs = int(hub_mask.sum())

    # Determine the set of classes to report on — every class present among
    # all nodes (so callers always see a stable row set even if a class is
    # empty in the observed hub set).
    unique_classes = sorted(set(node_classes.tolist()))

    # Observed per-class hub fraction.
    observed_frac: Dict[str, float] = {}
    if n_hubs == 0:
        for cls in unique_classes:
            observed_frac[cls] = 0.0
    else:
        for cls in unique_classes:
            observed_frac[cls] = float(
                np.sum(node_classes[hub_mask] == cls) / n_hubs
            )

    # Null distribution: shuffle class labels across nodes, recompute the
    # per-class hub fraction under the fixed hub mask.
    rng = np.random.default_rng(random_state)
    null_fracs: Dict[str, List[float]] = {cls: [] for cls in unique_classes}

    for _ in range(n_permutations):
        permuted = rng.permutation(node_classes)
        if n_hubs == 0:
            for cls in unique_classes:
                null_fracs[cls].append(0.0)
            continue
        hub_labels = permuted[hub_mask]
        for cls in unique_classes:
            null_fracs[cls].append(float(np.sum(hub_labels == cls) / n_hubs))

    rows = []
    for cls in unique_classes:
        null_arr = np.asarray(null_fracs[cls], dtype=float)
        obs = observed_frac[cls]
        null_mean = float(null_arr.mean()) if null_arr.size else float("nan")
        null_std = float(null_arr.std(ddof=0)) if null_arr.size else float("nan")
        # Two-sided: fraction of null iterations at least as extreme as
        # |obs - null_mean|.
        if null_arr.size == 0:
            pval = float("nan")
        else:
            diff_obs = abs(obs - null_mean)
            diff_null = np.abs(null_arr - null_mean)
            # Add +1 to numerator and denominator to avoid p = 0 (standard
            # permutation-test convention, Phipson & Smyth 2010).
            pval = float((np.sum(diff_null >= diff_obs) + 1) / (null_arr.size + 1))
        rows.append(
            {
                "cell_class": cls,
                "observed_hub_frac": obs,
                "null_mean": null_mean,
                "null_std": null_std,
                "pval_two_sided": pval,
            }
        )

    return pd.DataFrame(rows)


def hubs_contingency_test(
    hubs_sc: pd.DataFrame,
    hubs_fc: pd.DataFrame,
    cell_class_column: str = "supercluster_name",
) -> Dict[str, Any]:
    """Direct SC-vs-FC contingency test over hub cell-class distributions.

    Tests whether the hub cell-class distribution under the structural-
    connectivity (SC) model differs from that under the functional-
    connectivity (FC) model. Whereas ``hub_permutation_test`` tests one
    model at a time (per-class p-values vs a class-shuffled null), this
    test directly compares the two observed hub distributions by
    building a 2 x n_classes contingency table of hub counts
    (row 0 = SC, row 1 = FC) and running a Pearson chi-squared test
    (``scipy.stats.chi2_contingency``). When the table collapses to 2 x 2
    we also report ``scipy.stats.fisher_exact``; for larger tables Fisher's
    exact is not available in vanilla SciPy so ``p_fisher`` is ``None`` and
    the caller should rely on the chi-squared result (optionally with Monte
    Carlo p-value downstream).

    Parameters
    ----------
    hubs_sc, hubs_fc
        Hub tables (e.g. the output of :func:`identify_hubs`) with a column
        named ``cell_class_column`` whose values define the contingency
        table categories. A "count" per category = number of rows in the
        hub table with that category label.
    cell_class_column
        Column name whose values define the contingency-table categories.
        Defaults to ``"supercluster_name"``; pass ``"celltype"`` for the
        current ``identify_hubs`` output, or ``"cell_class"`` for the
        collapsed Excitatory/Inhibitory/Glia/Other view.

    Returns
    -------
    dict with keys
        - ``contingency_table``: 2 × n_classes :class:`pandas.DataFrame`
          (rows ``["SC", "FC"]``, columns = union of class labels).
        - ``chi2``: Pearson chi-squared statistic.
        - ``p_chi2``: chi-squared p-value.
        - ``dof``: degrees of freedom.
        - ``p_fisher``: Fisher's exact p-value (two-sided) when the table
          is 2 × 2, else ``None``.
    """
    if cell_class_column not in hubs_sc.columns:
        raise KeyError(
            f"Column {cell_class_column!r} not in hubs_sc "
            f"(columns={list(hubs_sc.columns)})"
        )
    if cell_class_column not in hubs_fc.columns:
        raise KeyError(
            f"Column {cell_class_column!r} not in hubs_fc "
            f"(columns={list(hubs_fc.columns)})"
        )

    sc_counts = hubs_sc[cell_class_column].value_counts()
    fc_counts = hubs_fc[cell_class_column].value_counts()

    all_classes = sorted(set(sc_counts.index) | set(fc_counts.index))
    table = pd.DataFrame(
        [
            [int(sc_counts.get(c, 0)) for c in all_classes],
            [int(fc_counts.get(c, 0)) for c in all_classes],
        ],
        index=["SC", "FC"],
        columns=all_classes,
    )

    # scipy.stats.chi2_contingency handles zero-rows/columns by trimming.
    chi2, p_chi2, dof, _expected = stats.chi2_contingency(table.values)

    p_fisher: Optional[float]
    if table.shape == (2, 2):
        _odds, p_fisher = stats.fisher_exact(table.values, alternative="two-sided")
        p_fisher = float(p_fisher)
    else:
        p_fisher = None

    return {
        "contingency_table": table,
        "chi2": float(chi2),
        "p_chi2": float(p_chi2),
        "dof": int(dof),
        "p_fisher": p_fisher,
    }
