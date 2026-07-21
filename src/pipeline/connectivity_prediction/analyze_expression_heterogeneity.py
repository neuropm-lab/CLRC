#!/usr/bin/env python3
"""Per-LR-pair expression-heterogeneity test (max-capacity vs
realized-signaling).

For each ligand-receptor pair, quantify how *concentrated* its ligand
and receptor gene expression is within each region, then ask whether
LR pairs with higher CLRC feature importance tend to be the more
concentrated (low-entropy / high-Gini) ones. Concentrated expression in
specific cell-types within a region is suggestive of targeted cell-cell
contacts rather than diffuse uniform expression, which in turn is
consistent (but not dispositive) with the LR pair's "maximum
communication capacity" being plausibly realized in vivo.

Two heterogeneity metrics per (gene, region):

  - Shannon entropy across cell-types
      Using ``supercluster_name`` (31 ABC cell classes), compute per
      cell-type mean expression of the gene within the region, normalize
      to a probability distribution, then
      ``H = -sum(p_ct * log2(p_ct))`` over non-zero cell-types.
      Low entropy means expression is concentrated in few cell-types.
      Range ``[0, log2(n_nonzero_celltypes)]``.

  - Gini coefficient across cells
      Per region, per gene, compute the standard Gini coefficient of
      the per-cell expression distribution (all cells in the region,
      across cell-types). Range ``[0, 1]``. High Gini means a small
      fraction of cells carries most of the expression.

The two metrics look at complementary axes: entropy is cell-type-level
and robust to per-cell dropout (cell-type means smooth out noise);
Gini is single-cell-level and captures within-cell-type expression
sparsity as well.

Per LR pair, entropy and Gini are averaged across (i) the pair's
ligand + receptor genes and (ii) the 109 brain regions, giving one
scalar per LR pair per metric. These are then Spearman-correlated
against aggregated CLRC feature importance (from
``<out>/interpretation/feature_categories.csv``), with a one-sided
alternative:

  - Concentrated ligand / receptor expression (low entropy, high Gini)
    is hypothesized to be enriched among high-importance LR pairs.
  - Null: permute the LR-importance vector 1000 times against the
    heterogeneity vector, recompute Spearman, empirical p via add-one
    smoothing.

This is an indirect test of whether importance tracks realized signaling
capacity. A positive correlation would provide weak evidence that CLRC-important LR pairs
tend to be expressed in the spatially concentrated patterns
characteristic of targeted cell-cell signaling; a null result would
not falsify the concern, only this proxy for it.

Outputs to ``<out>/lr_heterogeneity/``:
  - ``gene_heterogeneity.csv`` -- per (gene, region) entropy + Gini
    plus per-gene cross-region means.
  - ``lr_heterogeneity.csv`` -- per LR pair: ligand / receptor gene
    list, mean entropy, mean Gini, aggregated importance, category.
  - ``heterogeneity_vs_importance.json`` -- Spearman correlation and
    one-sided permutation p for entropy (negated) and Gini.

Run::

  uv run python src/pipeline/connectivity_prediction/analyze_expression_heterogeneity.py \\
      --config configs/abc_expanded_hpobest.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from clrc.core.io import find_repo_root, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.core.parallel import tqdm_joblib

logger = logging.getLogger(__name__)

_REGION_PREFIX = "Human "
_CELLTYPE_OBS_COL = "supercluster_name"
_REGION_OBS_COL = "region_of_interest_label"
_PERM_SEED = 42
_PERM_N = 1000


# ---------------------------------------------------------------------------
#  Heterogeneity metrics
# ---------------------------------------------------------------------------


def _shannon_entropy_matrix(X_celltype_mean: np.ndarray) -> np.ndarray:
    """Shannon entropy across cell-types, per gene, within a region.

    ``X_celltype_mean`` has shape ``(n_celltypes, n_genes)`` with the
    mean expression of each gene in each cell-type group within a
    region. Rows with a total of zero (gene not expressed in any
    cell-type in this region) yield ``nan`` entropy.
    """
    totals = X_celltype_mean.sum(axis=0)
    safe = np.where(totals > 0, totals, 1.0)
    p = X_celltype_mean / safe
    # Shannon entropy; 0 * log(0) := 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    H = -(p * logp).sum(axis=0)
    H = np.where(totals > 0, H, np.nan)
    return H


def _gini_columns(X_cells: np.ndarray) -> np.ndarray:
    """Vectorized per-column Gini coefficient.

    ``X_cells`` has shape ``(n_cells, n_genes)``; returns shape
    ``(n_genes,)``. Columns with zero total yield ``nan`` Gini. The
    computation sorts each column independently via ``np.sort(axis=0)``
    and applies the standard formula
    ``G = sum_i (2i - n - 1) * x_sorted[i] / (n * sum(x))`` with
    ``i`` indexing from 1.
    """
    if X_cells.size == 0:
        return np.full(X_cells.shape[1], np.nan)
    n = X_cells.shape[0]
    if n < 2:
        return np.full(X_cells.shape[1], np.nan)
    x_sorted = np.sort(X_cells, axis=0)
    i = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    weights = 2.0 * i - n - 1.0
    numer = (weights * x_sorted).sum(axis=0)
    totals = X_cells.sum(axis=0)
    safe = np.where(totals > 0, totals, 1.0)
    G = numer / (n * safe)
    G = np.where(totals > 0, G, np.nan)
    return G


# ---------------------------------------------------------------------------
#  Per-region aggregation
# ---------------------------------------------------------------------------


def _region_celltype_mean_matrix(
    X_sub: np.ndarray, celltype_ids: np.ndarray, n_celltypes: int,
) -> np.ndarray:
    """Return ``(n_celltypes, n_genes)`` mean expression matrix.

    ``X_sub`` is the dense expression submatrix for cells in one region.
    ``celltype_ids`` are integer codes [0, n_celltypes) for each cell's
    supercluster. Cell-types with no cells in this region get a row of
    zeros.
    """
    n_cells, n_genes = X_sub.shape
    out = np.zeros((n_celltypes, n_genes), dtype=np.float64)
    counts = np.zeros(n_celltypes, dtype=np.int64)
    np.add.at(out, celltype_ids, X_sub)
    np.add.at(counts, celltype_ids, 1)
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out


def _process_region(
    region: str, X_region: np.ndarray, celltype_ids: np.ndarray,
    n_celltypes: int, gene_names: List[str],
) -> pd.DataFrame:
    """Compute per-gene entropy + Gini for one region."""
    ct_mean = _region_celltype_mean_matrix(X_region, celltype_ids, n_celltypes)
    H = _shannon_entropy_matrix(ct_mean)
    G = _gini_columns(X_region)
    return pd.DataFrame({
        "region": region,
        "gene": gene_names,
        "entropy_celltype": H,
        "gini_cell": G,
        "n_cells_in_region": X_region.shape[0],
    })


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _resolve_out_root(cfg: dict) -> Path:
    base = Path(cfg["output"]["base_dir"])
    if not base.is_absolute():
        base = find_repo_root() / base
    return base


def _load_lr_pair_table(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "src/neuronchat/data/merged_interactionDB_human_1092LR.json"
    with path.open() as f:
        db = json.load(f)
    rows = []
    for name in sorted(db.keys()):
        entry = db[name]
        rows.append({
            "lr_name": name,
            "ligand_genes": list(entry["lig_contributor"]),
            "receptor_genes": list(entry["receptor_subunit"]),
        })
    return pd.DataFrame(rows)


def _load_feature_importance(out_root: Path) -> pd.DataFrame:
    """Load the per-LR-pair feature categorization with aggregated importance.

    ``feature_categories.csv`` from ``cross_target_biology.py`` uses
    ``group_name`` as the LR-pair identifier and exposes
    ``importance_sc / importance_fc / importance_combined`` columns alongside
    a ``category`` (SC-biased / FC-biased / Balanced) label. We normalize
    ``group_name`` to ``lr_name`` here so downstream merges with the LR-DB
    table (which uses ``lr_name``) are direct.
    """
    path = out_root / "interpretation" / "feature_categories.csv"
    df = pd.read_csv(path)
    required = {"group_name", "category", "importance_combined"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path} missing required columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )
    df = df.rename(columns={"group_name": "lr_name"})
    return df


def _resolve_h5ad_path(cfg: dict, repo_root: Path) -> Path:
    rel = cfg["data"]["abc_expression_h5ad"]
    p = Path(rel)
    if not p.is_absolute():
        p = repo_root / p
    if not p.is_file():
        raise FileNotFoundError(f"abc_expression_h5ad not found: {p}")
    return p


def _one_worker(args) -> pd.DataFrame:
    """loky-safe per-region worker: opens its own anndata, reads its region slice.

    Accepts a tuple of picklable primitives ``(h5ad_path, region, row_idx_list,
    celltype_ids_list, n_celltypes, gene_names)``; returns the per-(gene,region)
    heterogeneity DataFrame produced by :func:`_process_region`.
    """
    import anndata as ad
    (h5ad_path_s, region, row_idx_list, celltype_ids_list,
     n_celltypes, gene_names) = args
    adata = ad.read_h5ad(h5ad_path_s, backed="r")
    idx = np.asarray(row_idx_list, dtype=np.int64)
    X_sub = adata.X[idx, :]
    if sparse.issparse(X_sub):
        X_sub = X_sub.toarray()
    X_sub = np.asarray(X_sub, dtype=np.float32)
    return _process_region(
        region, X_sub,
        np.asarray(celltype_ids_list, dtype=np.int64),
        n_celltypes, gene_names,
    )


def compute_gene_heterogeneity(
    h5ad_path: Path, expected_regions: List[str], *, n_jobs: int = 1,
) -> pd.DataFrame:
    """Return a long-format frame of per (region, gene) entropy + Gini.

    Aggregates the 3.37M-cell h5ad to per (region, cell-type) means for
    the entropy metric, and computes the Gini from the full per-cell
    expression within each region.
    """
    import anndata as ad

    logger.info("Opening %s in backed mode...", h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    gene_names = adata.var["gene_symbol"].astype(str).tolist()
    region_series = (
        adata.obs[_REGION_OBS_COL].astype(str)
        .str.replace(f"^{_REGION_PREFIX}", "", regex=True)
    )
    celltype_series = adata.obs[_CELLTYPE_OBS_COL].astype(str)

    celltype_categories = sorted(celltype_series.unique())
    celltype_to_id = {ct: i for i, ct in enumerate(celltype_categories)}
    n_celltypes = len(celltype_categories)
    logger.info("Cell-types: %d", n_celltypes)

    all_regions = sorted(region_series.unique())
    missing = [r for r in expected_regions if r not in all_regions]
    if missing:
        raise KeyError(
            f"h5ad missing regions required by the alignment pickle: {missing}"
        )

    # Pre-compute per-region row indices once (fast).
    region_rows: Dict[str, np.ndarray] = {}
    celltype_ids: Dict[str, np.ndarray] = {}
    for r in expected_regions:
        mask = (region_series == r).to_numpy()
        idx = np.flatnonzero(mask)
        region_rows[r] = idx
        celltype_ids[r] = celltype_series.iloc[idx].map(celltype_to_id).to_numpy(dtype=np.int64)

    def _one_serial(region: str) -> pd.DataFrame:
        idx = region_rows[region]
        X_sub = adata.X[idx, :]
        if sparse.issparse(X_sub):
            X_sub = X_sub.toarray()
        X_sub = np.asarray(X_sub, dtype=np.float32)
        return _process_region(
            region, X_sub, celltype_ids[region], n_celltypes, gene_names,
        )

    t0 = time.perf_counter()
    if n_jobs == 1:
        frames = []
        for r in expected_regions:
            t_r = time.perf_counter()
            frames.append(_one_serial(r))
            logger.info(
                "region %s: %d cells in %.1fs",
                r, region_rows[r].size, time.perf_counter() - t_r,
            )
    else:
        # loky workers cannot pickle the backed-mode anndata handle
        # (h5py objects are not picklable), so each worker opens its own
        # anndata and reads only its region. The inner function is a
        # module-level callable (defined below) so it survives pickling.
        from joblib import Parallel, delayed
        from tqdm.auto import tqdm
        worker_args = [
            (
                str(h5ad_path),
                region,
                region_rows[region].tolist(),
                celltype_ids[region].tolist(),
                n_celltypes,
                gene_names,
            )
            for region in expected_regions
        ]
        with tqdm_joblib(
            tqdm(total=len(expected_regions), desc="heterogeneity per region")
        ):
            frames = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one_worker)(args) for args in worker_args
            )
    logger.info(
        "Per-region heterogeneity complete (%d regions, %.1fs)",
        len(expected_regions), time.perf_counter() - t0,
    )
    return pd.concat(frames, ignore_index=True)


def aggregate_to_lr_pair(
    gene_het: pd.DataFrame, lr_table: pd.DataFrame,
) -> pd.DataFrame:
    """Average per (gene, region) heterogeneity to per LR pair."""
    per_gene = gene_het.groupby("gene").agg(
        entropy_gene_mean=("entropy_celltype", "mean"),
        gini_gene_mean=("gini_cell", "mean"),
        n_regions=("region", "nunique"),
    ).reset_index()
    lookup_ent = dict(zip(per_gene["gene"], per_gene["entropy_gene_mean"]))
    lookup_gini = dict(zip(per_gene["gene"], per_gene["gini_gene_mean"]))

    rows = []
    for _, row in lr_table.iterrows():
        genes = list(row["ligand_genes"]) + list(row["receptor_genes"])
        ents = [lookup_ent[g] for g in genes if g in lookup_ent and np.isfinite(lookup_ent[g])]
        ginis = [lookup_gini[g] for g in genes if g in lookup_gini and np.isfinite(lookup_gini[g])]
        rows.append({
            "lr_name": row["lr_name"],
            "n_ligand_genes": len(row["ligand_genes"]),
            "n_receptor_genes": len(row["receptor_genes"]),
            "n_genes_used": len(ents),
            "entropy_lr_mean": float(np.mean(ents)) if ents else float("nan"),
            "gini_lr_mean": float(np.mean(ginis)) if ginis else float("nan"),
        })
    return pd.DataFrame(rows)


def importance_vs_heterogeneity(
    lr_het: pd.DataFrame, feature_cat: pd.DataFrame,
    importance_col: str = "importance_combined",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Merge heterogeneity with importance and compute Spearman + permutation.

    Uses ``importance_col`` from ``feature_cat``; falls back to
    ``importance_combined`` if present, else tries
    ``aggregated_importance_combined``, else raises.
    """
    candidate_cols = [
        importance_col,
        "importance_combined",
        "aggregated_importance_combined",
        "combined_importance",
    ]
    picked = next((c for c in candidate_cols if c in feature_cat.columns), None)
    if picked is None:
        raise KeyError(
            f"feature_categories.csv has no importance column among "
            f"{candidate_cols}. Columns: {list(feature_cat.columns)}"
        )
    merged = lr_het.merge(
        feature_cat[["lr_name", "category", picked]].rename(
            columns={picked: "importance_combined"}
        ),
        on="lr_name", how="inner",
    )
    ok = merged.dropna(subset=["entropy_lr_mean", "gini_lr_mean", "importance_combined"])
    imp = ok["importance_combined"].to_numpy()
    ent = ok["entropy_lr_mean"].to_numpy()
    gini = ok["gini_lr_mean"].to_numpy()
    n = len(ok)

    rho_ent = float(stats.spearmanr(imp, -ent).statistic) if n >= 10 else float("nan")
    rho_gini = float(stats.spearmanr(imp, gini).statistic) if n >= 10 else float("nan")

    rng = np.random.default_rng(_PERM_SEED)
    ge_ent = 0
    ge_gini = 0
    for _ in range(_PERM_N):
        perm = rng.permutation(imp)
        if stats.spearmanr(perm, -ent).statistic >= rho_ent:
            ge_ent += 1
        if stats.spearmanr(perm, gini).statistic >= rho_gini:
            ge_gini += 1
    p_ent = (1 + ge_ent) / (1 + _PERM_N)
    p_gini = (1 + ge_gini) / (1 + _PERM_N)

    return merged, {
        "n_lr_pairs": int(n),
        "importance_col_used": picked,
        "spearman_importance_vs_neg_entropy": rho_ent,
        "spearman_importance_vs_gini": rho_gini,
        "permutation_p_neg_entropy_one_sided": p_ent,
        "permutation_p_gini_one_sided": p_gini,
        "n_permutations": _PERM_N,
        "permutation_seed": _PERM_SEED,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help="joblib worker count for per-region aggregation.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml_config(args.config)
    repo_root = find_repo_root()
    out_root = _resolve_out_root(cfg)

    analysis_dir = out_root / "lr_heterogeneity"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("analyze_expression_heterogeneity", output_dir=analysis_dir)

    # Get the 109 expected ABC regions from the alignment pickle so we
    # only aggregate the region subset used by the main pipeline.
    import pickle
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = repo_root / align_pkl
    with align_pkl.open("rb") as f:
        align = pickle.load(f)
    expected_regions = list(align["ABC_regions_cci"])
    logger.info("Expected regions: %d", len(expected_regions))

    h5ad_path = _resolve_h5ad_path(cfg, repo_root)
    lr_table = _load_lr_pair_table(repo_root)
    feature_cat = _load_feature_importance(out_root)

    gene_het = compute_gene_heterogeneity(
        h5ad_path, expected_regions, n_jobs=args.n_jobs,
    )
    gene_het_path = analysis_dir / "gene_heterogeneity.csv"
    gene_het.to_csv(gene_het_path, index=False)
    logger.info("wrote per (gene, region) -> %s (%d rows)", gene_het_path, len(gene_het))

    lr_het = aggregate_to_lr_pair(gene_het, lr_table)
    lr_het_path = analysis_dir / "lr_heterogeneity.csv"
    lr_het.to_csv(lr_het_path, index=False)
    logger.info("wrote per LR pair -> %s (%d rows)", lr_het_path, len(lr_het))

    merged, stats_summary = importance_vs_heterogeneity(lr_het, feature_cat)
    merged_path = analysis_dir / "lr_heterogeneity_with_importance.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("wrote merged with importance -> %s", merged_path)

    summary_path = analysis_dir / "heterogeneity_vs_importance.json"
    with summary_path.open("w") as f:
        json.dump(stats_summary, f, indent=2)
    logger.info("wrote summary -> %s", summary_path)

    print("\n" + "=" * 80)
    print("Expression heterogeneity vs CLRC importance (per LR pair)")
    print("=" * 80)
    for k, v in stats_summary.items():
        print(f"  {k:45s} {v}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
