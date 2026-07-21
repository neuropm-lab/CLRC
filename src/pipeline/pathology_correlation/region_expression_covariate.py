#!/usr/bin/env python3
"""Region-expression covariate extension for AD partial Spearman.

Adds a **per-subject per-region per-gene** expression covariate derived
from the ROSMAP snRNA-seq atlas to the partial Spearman test of CLRC
features against clinical variables. A subject-invariant broadcast
would be absorbed into the intercept and produce results identical to
baseline; varying the covariate across both axes of the test is
required for it to have statistical bite.

Analysis unit
-------------
The unit of the partial Spearman test is the **(subject, region) pair**.
We pool across all ROSMAP subjects x AD regions (fewer if a subject is
missing a region in either the CLRC aggregation or the snRNA subset).
Both the CLRC feature matrix and the expression covariate therefore
vary along the same axis -- no subject-constant broadcast.

Covariate variants
------------------
- ``baseline``: no extra covariate. Region-stacked partial Spearman
  with just ``age_death, educ, msex``. Useful as a reference.
- ``lr_pair_expression``: add per-LR-pair mean(log1p raw counts) of
  ligand + receptor genes, per (subject, region). Each LR feature gets
  its own pair-specific covariate column.
- ``global_pc1``: add PC1 of the per-(subject, region) gene-expression
  matrix. Shared across all features.
- ``lr_pair_expression+global_pc1``: both added together.

Expression covariate construction
---------------------------------
1. Load the LR-gene subset h5ad (produced by
   ``preprocess_rosmap_lr_subset.py``).
2. Per (subject, region), mean log1p(raw counts) across cells, per gene.
3. Reshape into a ``(n_subjects * n_regions, n_genes)`` long matrix
   (NaN rows for missing subject-region combos).
4. ``lr_pair_expression``: for each LR pair, mean across the ligand +
   receptor genes that exist in the panel.
5. ``global_pc1``: PCA on the long matrix (rows with any NaN dropped
   from fit; scores scattered back). Standardised columns.

Regression guard
----------------
The ``lr_pair_expression`` output must differ from the ``baseline``
output for at least some features -- this catches a silent regression
to the old no-op behaviour.

Usage
-----
    uv run python src/pipeline/pathology_correlation/region_expression_covariate.py \
        --config configs/rosmap_expanded.yaml \
        --covariate-variant lr_pair_expression \
        --rosmap-subset-h5ad data/AD_Multiomic_MultiRegion/rosmap_lr_subset_536.h5ad

Output
------
``<output_base>/covariates/<variant>/partial_spearman.csv`` -- long-form
table. One row per (clinical_var, feature_name) with the partial
Spearman correlation and p-value at the (subject, region)-stacked scope.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from clrc.ad.correlation import (
    correlate_one_lr_group,
    partial_spearman_batch,
)
from clrc.core.parallel import tqdm_joblib
from clrc.core.io import (
    find_repo_root,
    load_pickle,
    load_yaml_config,
    save_pickle,
)
from clrc.core.logging import setup_logging


logger = logging.getLogger("clrc.pipeline.region_expression_covariate")


VARIANT_CHOICES = ("baseline", "lr_pair_expression", "global_pc1", "lr_pair_expression+global_pc1")

# Feature names use the form '{LR} | {sender}->{receiver}' where the
# separator is U+2192 (RIGHTWARDS ARROW). See
# ``clrc.ad.aggregation.build_feature_name``.
_FEATURE_SPLIT_RE = re.compile(r"\s\|\s")


# ---------------------------------------------------------------------------
#  Feature parsing
# ---------------------------------------------------------------------------

def parse_feature(feature_name: str) -> Tuple[str, str, str]:
    """Return (lr_name, sender_ct, receiver_ct) for a feature name.

    Raises ValueError if the format is unexpected.
    """
    parts = _FEATURE_SPLIT_RE.split(feature_name, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected feature-name format: {feature_name!r}")
    lr = parts[0]
    sr = parts[1]
    if "\u2192" in sr:
        sender, receiver = sr.split("\u2192", 1)
    elif "->" in sr:
        sender, receiver = sr.split("->", 1)
    else:
        raise ValueError(f"Unexpected sender/receiver delim: {feature_name!r}")
    return lr, sender.strip(), receiver.strip()


def feature_lr_indices(
    feature_names: Sequence[str], lr_labels: Sequence[str]
) -> np.ndarray:
    """Map each feature name to its LR index. Raises KeyError on miss."""
    lr_to_idx = {name: i for i, name in enumerate(lr_labels)}
    out = np.empty(len(feature_names), dtype=np.intp)
    missing: List[str] = []
    for i, f in enumerate(feature_names):
        lr, _s, _r = parse_feature(f)
        if lr not in lr_to_idx:
            missing.append(lr)
            continue
        out[i] = lr_to_idx[lr]
    if missing:
        uniq = sorted(set(missing))[:10]
        raise KeyError(
            f"{len(set(missing))} feature LR tokens not found in lr_labels. "
            f"First 10: {uniq}"
        )
    return out


# ---------------------------------------------------------------------------
#  Expression aggregation: per (subject, region, gene) mean log1p counts
# ---------------------------------------------------------------------------

def aggregate_subject_region_expression(
    h5ad_path: Path,
    subject_ids: Sequence[str],
    region_names: Sequence[str],
    *,
    subject_col: str = "ROSMAP_IndividualID",
    region_col: str = "BrainRegion",
) -> Tuple[np.ndarray, List[str]]:
    """Return per-(subject, region, gene) mean log1p raw counts.

    Loads the *subset* h5ad (not the 17 GB panel) and aggregates cells by
    ``(subject_col, region_col)``. Missing combos (subject-region pairs
    with no cells) are written as NaN rows so they can be filtered
    downstream.

    Parameters
    ----------
    h5ad_path
        Path to the LR-gene subset h5ad produced by
        ``preprocess_rosmap_lr_subset.py``.
    subject_ids
        Ordering of subject rows in the output. Exactly the subject IDs
        used by the AD aggregation pickle so rows align.
    region_names
        Ordering of region columns.
    subject_col, region_col
        ``obs`` column names to group by.

    Returns
    -------
    expr : np.ndarray of shape ``(n_subjects, n_regions, n_genes)``
        Cells without data (no cells in that subject-region) are NaN.
    gene_names : list[str]
        Column ordering.
    """
    import anndata as ad

    logger.info("Loading subset h5ad: %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    gene_names = list(adata.var_names)
    logger.info(
        "Subset adata: %d cells x %d genes", adata.n_obs, adata.n_vars
    )

    obs = adata.obs[[subject_col, region_col]].copy()
    obs[subject_col] = obs[subject_col].astype(str)
    obs[region_col] = obs[region_col].astype(str)

    # Build integer keys for vectorised aggregation.
    subj_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    reg_to_idx = {rid: i for i, rid in enumerate(region_names)}

    # Drop cells whose subject or region is not in the target sets.
    subj_idx = obs[subject_col].map(subj_to_idx)
    reg_idx = obs[region_col].map(reg_to_idx)
    keep = subj_idx.notna() & reg_idx.notna()
    dropped = int((~keep).sum())
    if dropped > 0:
        logger.info(
            "Dropping %d cells whose subject or region is not in target sets.",
            dropped,
        )
    subj_idx = subj_idx[keep].to_numpy(dtype=np.int64)
    reg_idx = reg_idx[keep].to_numpy(dtype=np.int64)

    # Slice X accordingly. adata.X is sparse (csr); keep sparse for mean.
    X = adata.X[keep.to_numpy()]  # (n_cells_kept, n_genes)
    import scipy.sparse as sp
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    n_subjects = len(subject_ids)
    n_regions = len(region_names)
    n_genes = len(gene_names)

    # Per-cell log1p of raw counts, then mean across cells within a
    # (subject, region) group. We aggregate by converting each group's
    # cells to log1p on the fly, summing, and dividing by count.
    expr = np.full((n_subjects, n_regions, n_genes), np.nan, dtype=np.float64)

    # Combined key for grouping.
    combo = subj_idx * n_regions + reg_idx  # (n_cells,)
    n_combos = n_subjects * n_regions
    counts = np.bincount(combo, minlength=n_combos)

    # log1p of raw counts; csr_matrix.log1p returns a csr_matrix.
    logger.info(
        "Computing log1p and group-sum over %d cells x %d genes (%d combos)...",
        X.shape[0], X.shape[1], n_combos,
    )
    Xlog = X.log1p()

    # Group sum: for each combo, sum rows. We do this by building a
    # sparse 'group-indicator' matrix G of shape (n_combos, n_cells)
    # with G[c, i] = 1 iff combo[i] == c; then sums = G @ Xlog.
    n_cells = Xlog.shape[0]
    G = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float64),
         (combo, np.arange(n_cells, dtype=np.int64))),
        shape=(n_combos, n_cells),
    )
    sums = (G @ Xlog).toarray()  # (n_combos, n_genes)

    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(
            counts[:, None] > 0,
            sums / np.maximum(counts[:, None], 1),
            np.nan,
        )

    # Reshape back to (n_subjects, n_regions, n_genes).
    expr = means.reshape(n_subjects, n_regions, n_genes)

    n_missing = int(np.sum(counts == 0))
    logger.info(
        "Per-(subject, region) aggregation: %d combos total, %d missing (no cells).",
        n_combos, n_missing,
    )
    return expr, gene_names


# ---------------------------------------------------------------------------
#  Covariate builders: lr_pair_expression (per-LR pair) and global_pc1 (global PC1)
# ---------------------------------------------------------------------------

def build_lr_pair_expression_covariate(
    expr_long: np.ndarray,              # (n_obs, n_genes)
    gene_names: Sequence[str],
    lr_labels: Sequence[str],
    lr_gene_lookup: Mapping[str, Tuple[Sequence[str], Sequence[str]]],
) -> np.ndarray:
    """Return the per-observation, per-LR-pair lr_pair_expression covariate matrix.

    Shape: ``(n_obs, n_lr_pairs)``. For each LR pair, the covariate in
    observation ``i`` is the mean of the ligand + receptor genes (deduped)
    that intersect ``gene_names``. LR pairs with zero overlap contribute
    an all-NaN column — the partial Spearman helper propagates NaNs via
    its missing-mask logic.
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_obs = expr_long.shape[0]
    n_lr = len(lr_labels)
    cov = np.full((n_obs, n_lr), np.nan, dtype=np.float64)

    all_nan_cols = 0
    for j, lr_name in enumerate(lr_labels):
        lig, rec = lr_gene_lookup.get(lr_name, ([], []))
        unique_genes = []
        seen = set()
        for g in list(lig) + list(rec):
            if g in gene_to_idx and g not in seen:
                seen.add(g)
                unique_genes.append(g)
        if not unique_genes:
            all_nan_cols += 1
            continue
        idxs = np.asarray([gene_to_idx[g] for g in unique_genes], dtype=np.intp)
        cov[:, j] = np.nanmean(expr_long[:, idxs], axis=1)

    logger.info(
        "lr_pair_expression covariate: shape=%s, all-NaN columns=%d / %d",
        cov.shape, all_nan_cols, n_lr,
    )
    return cov


def build_global_pc1_covariate(
    expr_long: np.ndarray,              # (n_obs, n_genes)
    *,
    standardize: bool = True,
) -> np.ndarray:
    """Return PC1 scores for each observation, length ``n_obs``.

    Fits PCA on the rows without any NaN; scatters the PC1 scores back
    into a length-``n_obs`` vector (NaN for dropped rows).
    """
    from sklearn.decomposition import PCA

    n_obs, n_genes = expr_long.shape
    valid = ~np.any(np.isnan(expr_long), axis=1)
    X = expr_long[valid]
    if X.shape[0] < 3:
        raise ValueError(
            f"Only {X.shape[0]} non-NaN rows; PCA requires at least 3."
        )

    if standardize:
        col_mean = X.mean(axis=0, keepdims=True)
        col_std = X.std(axis=0, keepdims=True, ddof=0)
        nonzero = col_std.ravel() > 0
        if not nonzero.any():
            raise ValueError(
                "All gene columns have zero variance; cannot compute PC1."
            )
        Xz = (X[:, nonzero] - col_mean[:, nonzero]) / col_std[:, nonzero]
    else:
        Xz = X

    pca = PCA(n_components=1)
    scores_valid = pca.fit_transform(Xz)[:, 0]
    # z-score the scores so downstream partial-Spearman projection matrix
    # is well-conditioned.
    sd = scores_valid.std(ddof=0)
    if sd > 0:
        scores_valid = (scores_valid - scores_valid.mean()) / sd

    out = np.full(n_obs, np.nan, dtype=np.float64)
    out[valid] = scores_valid
    logger.info(
        "global_pc1 covariate: n_valid=%d / %d, explained_variance_ratio=%.4f",
        int(valid.sum()), n_obs, float(pca.explained_variance_ratio_[0]),
    )
    return out


# ---------------------------------------------------------------------------
#  Long-form reshape: (n_subj, n_regions, ...) -> (n_subj * n_regions, ...)
# ---------------------------------------------------------------------------

def stack_long(
    X_region: np.ndarray,              # (n_subj, n_regions, n_features)
    Y_clinical: np.ndarray,            # (n_subj, n_vars)
    covariates_arr: np.ndarray,        # (n_subj, k_base)
    subject_ids: Sequence[str],
    region_names: Sequence[str],
) -> Tuple[
    np.ndarray,            # X_long (n_obs, n_features)
    np.ndarray,            # Y_long (n_obs, n_vars)
    np.ndarray,            # cov_long (n_obs, k_base)
    np.ndarray,            # subj_idx (n_obs,) int
    np.ndarray,            # reg_idx  (n_obs,) int
]:
    """Broadcast subject-level arrays across regions and flatten.

    The partial Spearman unit is the (subject, region) pair, so Y and
    covariates are replicated along the region axis.
    """
    n_subj, n_regions, n_features = X_region.shape
    n_obs = n_subj * n_regions

    X_long = X_region.reshape(n_obs, n_features)

    Y_long = np.repeat(Y_clinical, n_regions, axis=0)
    assert Y_long.shape == (n_obs, Y_clinical.shape[1])

    cov_long = np.repeat(covariates_arr, n_regions, axis=0)
    assert cov_long.shape == (n_obs, covariates_arr.shape[1])

    subj_idx = np.repeat(np.arange(n_subj), n_regions)
    reg_idx = np.tile(np.arange(n_regions), n_subj)
    return X_long, Y_long, cov_long, subj_idx, reg_idx


# ---------------------------------------------------------------------------
#  Partial-Spearman driver (stacked subject-region long-form)
# ---------------------------------------------------------------------------

def _group_features_by_lr(
    feature_lr_idx: np.ndarray,
) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[int]] = {}
    for j, lr_idx in enumerate(feature_lr_idx):
        groups.setdefault(int(lr_idx), []).append(j)
    return {k: np.asarray(v, dtype=np.intp) for k, v in groups.items()}


def correlate_long(
    X_long: np.ndarray,
    Y_long: np.ndarray,
    base_cov_long: np.ndarray,
    feature_names: Sequence[str],
    feature_lr_idx: np.ndarray,
    clinical_vars: Sequence[str],
    variant: str,
    cov_6a: Optional[np.ndarray],       # (n_obs, n_lr) or None
    cov_6c: Optional[np.ndarray],       # (n_obs,) or None
    *,
    n_jobs: int = 1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run partial Spearman for every (clinical var, feature) pair.

    For non-baseline variants the per-LR-group calls are independent
    (each has its own lr_pair_expression covariate column + a shared global_pc1) and are
    dispatched across a joblib ``Parallel(n_jobs=n_jobs)`` pool with a
    tqdm progress bar per clinical variable.
    """
    n_obs, n_features = X_long.shape
    groups = _group_features_by_lr(feature_lr_idx)

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for var_idx, var_name in enumerate(clinical_vars):
        y = Y_long[:, var_idx]
        corrs = np.full(n_features, np.nan)
        pvals = np.full(n_features, np.nan)

        if variant == "baseline":
            r, p = partial_spearman_batch(X_long, y, base_cov_long)
            corrs[:] = r
            pvals[:] = p
        else:
            group_items = list(groups.items())
            logger.info(
                "  %s (variant=%s): dispatching %d LR groups across n_jobs=%d",
                var_name, variant, len(group_items), n_jobs,
            )
            # Precompute per-group inputs (cheap views, no copy) so the
            # parallel dispatcher does not need to know about cov_6a shape.
            tasks = (
                delayed(correlate_one_lr_group)(
                    lr_idx,
                    cols,
                    X_long[:, cols],
                    y,
                    base_cov_long,
                    cov_6a[:, lr_idx] if cov_6a is not None else None,
                    cov_6c,
                )
                for lr_idx, cols in group_items
            )
            with tqdm_joblib(
                tqdm(total=len(group_items), desc=f"  {var_name}", unit="LR")
            ):
                out = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

            for cols, r, p in out:
                corrs[cols] = r
                pvals[cols] = p

        results[var_name] = {"correlations": corrs, "pvalues": pvals}
        valid = ~np.isnan(corrs)
        mean_abs = (
            float(np.nanmean(np.abs(corrs))) if valid.any() else float("nan")
        )
        logger.info(
            "  %s (variant=%s): n_valid=%d/%d mean|r|=%.3f",
            var_name, variant, int(valid.sum()), len(corrs), mean_abs,
        )
    return results


# ---------------------------------------------------------------------------
#  LR gene lookup (expanded DB)
# ---------------------------------------------------------------------------

def load_lr_gene_lookup(
    db_path: Path, lr_labels: Sequence[str]
) -> Dict[str, Tuple[List[str], List[str]]]:
    import json
    with db_path.open() as f:
        db = json.load(f)
    lookup: Dict[str, Tuple[List[str], List[str]]] = {}
    missing: List[str] = []
    for name in lr_labels:
        entry = db.get(name)
        if entry is None:
            missing.append(name)
            lookup[name] = ([], [])
            continue
        lookup[name] = (
            [str(g) for g in entry.get("lig_contributor", [])],
            [str(g) for g in entry.get("receptor_subunit", [])],
        )
    if missing:
        logger.warning(
            "%d LR pairs missing from interaction DB; lr_pair_expression column is NaN. "
            "First 5: %s", len(missing), missing[:5],
        )
    return lookup


# ---------------------------------------------------------------------------
#  Archive-before-overwrite
# ---------------------------------------------------------------------------

def _archive_dir_if_exists(target: Path) -> Optional[Path]:
    if not target.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = target.with_name(f"{target.name}.archive.{stamp}")
    shutil.move(str(target), str(archive))
    logger.info("Archived existing output: %s -> %s", target, archive)
    return archive


# ---------------------------------------------------------------------------
#  Subset helpers (for the --dry-run sub-slicing)
# ---------------------------------------------------------------------------

def _subset_aggregation(
    agg: Dict,
    *,
    n_lr: Optional[int] = None,
    n_clinical: Optional[int] = None,
) -> Dict:
    """Subset first ``n_lr`` LR pairs and first ``n_clinical`` clinical vars."""
    out = dict(agg)
    cts = agg["unique_celltypes"]
    n_ct = len(cts)
    full_n_lr = len(agg["lr_labels"])

    if n_lr is not None and n_lr < full_n_lr:
        feature_mask = np.zeros(full_n_lr * n_ct * n_ct, dtype=bool)
        feature_mask[: n_lr * n_ct * n_ct] = True
        out["lr_labels"] = agg["lr_labels"][:n_lr]
        out["X_collapsed"] = agg["X_collapsed"][:, feature_mask]
        out["feature_names_collapsed"] = [
            f for f, k in zip(agg["feature_names_collapsed"], feature_mask) if k
        ]
        out["X_region"] = agg["X_region"][:, :, feature_mask]
        out["feature_names_region"] = [
            f for f, k in zip(agg["feature_names_region"], feature_mask) if k
        ]

    if n_clinical is not None and n_clinical < len(agg["clinical_vars"]):
        out["Y_clinical"] = agg["Y_clinical"][:, :n_clinical]
        out["clinical_vars"] = agg["clinical_vars"][:n_clinical]
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-subject per-region expression covariate "
        "extension for AD partial Spearman.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--covariate-variant",
        choices=list(VARIANT_CHOICES),
        required=True,
    )
    parser.add_argument(
        "--rosmap-subset-h5ad", type=Path, required=True,
        help="Gene-subsetted ROSMAP snRNA-seq h5ad "
        "(produced by preprocess_rosmap_lr_subset.py).",
    )
    parser.add_argument(
        "--aggregation-pkl", type=Path, default=None,
        help="Override path to aggregation pickle "
        "(default: <out_base>/aggregation/ad_aggregation.pkl).",
    )
    parser.add_argument(
        "--lr-db", type=Path, default=None,
        help="Interaction DB JSON for LR gene lookup. Default: "
        "src/neuronchat/data/merged_interactionDB_human_1092LR.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: "
        "<out_base>/covariates/<variant>).",
    )
    parser.add_argument(
        "--dry-run-lr", type=int, default=None,
        help="Restrict to the first K LR pairs (smoke test).",
    )
    parser.add_argument(
        "--dry-run-clinical", type=int, default=None,
        help="Restrict to the first K clinical variables.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=16,
        help=(
            "Parallel joblib workers for the per-LR-group partial-Spearman "
            "loop (non-baseline variants only). Default 16. Set lower when "
            "stacking against another CPU-saturating job."
        ),
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    setup_logging("region_expression_covariate", out_base)

    variant: str = args.covariate_variant
    variant_dir = (
        args.output_dir
        or (out_base / "covariates" / variant.replace("+", "_plus_"))
    )
    _archive_dir_if_exists(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Region-expression covariate driver: variant=%s, out=%s", variant, variant_dir,
    )

    repo_root = find_repo_root()

    # --- Load aggregation ---
    agg_path = args.aggregation_pkl or (
        out_base / "aggregation" / "ad_aggregation.pkl"
    )
    agg = load_pickle(agg_path)
    if args.dry_run_lr or args.dry_run_clinical:
        agg = _subset_aggregation(
            agg, n_lr=args.dry_run_lr, n_clinical=args.dry_run_clinical,
        )
        logger.info(
            "DRY RUN: %d LR pairs, %d clinical vars",
            len(agg["lr_labels"]), len(agg["clinical_vars"]),
        )

    subject_ids = list(agg["subject_ids"])
    region_names = list(agg["region_names"])
    X_region = agg["X_region"]
    feature_names_region = agg["feature_names_region"]
    Y_clinical = agg["Y_clinical"]
    covariates_arr = agg["covariates_arr"]
    clinical_vars = list(agg["clinical_vars"])
    lr_labels = list(agg["lr_labels"])

    n_subj = len(subject_ids)
    n_regions = len(region_names)
    logger.info(
        "Aggregation loaded: %d subjects, %d LR pairs, %d regions, "
        "%d features per region",
        n_subj, len(lr_labels), n_regions, X_region.shape[-1],
    )

    # --- Build per-(subject, region, gene) expression matrix ---
    expr_3d, gene_names = aggregate_subject_region_expression(
        args.rosmap_subset_h5ad,
        subject_ids=subject_ids,
        region_names=region_names,
    )
    expr_long = expr_3d.reshape(n_subj * n_regions, len(gene_names))
    logger.info(
        "Expression long matrix: shape=%s, non-NaN rows=%d",
        expr_long.shape, int(np.sum(~np.any(np.isnan(expr_long), axis=1))),
    )

    # --- Build lr_pair_expression / global_pc1 covariates (as needed) ---
    cov_6a: Optional[np.ndarray] = None
    cov_6c: Optional[np.ndarray] = None

    if variant in ("lr_pair_expression", "lr_pair_expression+global_pc1"):
        db_path = args.lr_db or (
            repo_root / "src/neuronchat/data/merged_interactionDB_human_1092LR.json"
        )
        lr_lookup = load_lr_gene_lookup(db_path, lr_labels)
        cov_6a = build_lr_pair_expression_covariate(expr_long, gene_names, lr_labels, lr_lookup)

    if variant in ("global_pc1", "lr_pair_expression+global_pc1"):
        cov_6c = build_global_pc1_covariate(expr_long, standardize=True)

    # --- Stack CLRC feature matrix to long form ---
    X_long, Y_long, base_cov_long, subj_idx, reg_idx = stack_long(
        X_region, Y_clinical, covariates_arr, subject_ids, region_names,
    )
    logger.info(
        "Long-form stacked: X=%s Y=%s base_cov=%s",
        X_long.shape, Y_long.shape, base_cov_long.shape,
    )

    # --- Compute partial Spearman ---
    feat_lr_idx = feature_lr_indices(feature_names_region, lr_labels)
    results = correlate_long(
        X_long=X_long,
        Y_long=Y_long,
        base_cov_long=base_cov_long,
        feature_names=feature_names_region,
        feature_lr_idx=feat_lr_idx,
        clinical_vars=clinical_vars,
        variant=variant,
        cov_6a=cov_6a,
        cov_6c=cov_6c,
        n_jobs=args.n_jobs,
    )

    # --- Emit long-form CSV ---
    # Parse features once so we can emit lr_name, ct_L, ct_R columns.
    parsed = [parse_feature(f) for f in feature_names_region]
    lr_col = [p[0] for p in parsed]
    ctL_col = [p[1] for p in parsed]
    ctR_col = [p[2] for p in parsed]

    rows: List[pd.DataFrame] = []
    for var_name, res in results.items():
        df = pd.DataFrame({
            "feature_name": feature_names_region,
            "lr_name": lr_col,
            "ct_L": ctL_col,
            "ct_R": ctR_col,
            "clinical_var": var_name,
            "correlation": res["correlations"],
            "pvalue": res["pvalues"],
            "n_valid": np.full(
                len(feature_names_region),
                int(np.sum(~np.any(np.isnan(base_cov_long), axis=1)
                           & ~np.isnan(Y_long[:, clinical_vars.index(var_name)]))),
            ),
            "covariate_variant": variant,
            "scope": "subject_region_stacked",
        })
        rows.append(df)
    long_df = pd.concat(rows, ignore_index=True)

    out_csv = variant_dir / "partial_spearman.csv"
    long_df.to_csv(out_csv, index=False)
    logger.info("Wrote %s (rows=%d)", out_csv, len(long_df))

    save_pickle(
        {
            "results": results,
            "feature_names": feature_names_region,
            "lr_labels": lr_labels,
            "clinical_vars": clinical_vars,
            "subject_ids": subject_ids,
            "region_names": region_names,
            "covariate_variant": variant,
            "gene_names": gene_names,
            "subject_region_subj_idx": subj_idx,
            "subject_region_reg_idx": reg_idx,
        },
        variant_dir / "all_correlations.pkl",
    )

    logger.info("Done. Outputs in %s", variant_dir)


if __name__ == "__main__":
    main()
