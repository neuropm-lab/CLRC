#!/usr/bin/env python3
"""Co-expression baseline driver.

Three baselines test whether CLRC's predictive performance recapitulates
simple gene co-expression. Each baseline produces an alternative
``(n_edges, n_features)`` matrix aligned to the same ``edge_table`` as
the main analysis, wrapped in the same alignment-pickle schema, then
fed to the same XGBoost LOBO pipeline with the HPO-best parameters of
the target.

Baselines (semantic identifiers also used as ``--baseline`` CLI values
and as subdirectory names under ``<out>/coexpression_baseline/``):

  ``region_collapsed_nc``
      Per LR pair, collapse NeuronChat's cell-type-resolved communication
      matrix by mean across cell-type pairs, yielding one scalar per edge.
      Tests whether CLRC's cell-type resolution contributes beyond the
      region-level NeuronChat output alone.

  ``lr_expression_product``
      Per LR pair and per edge, the product of the sender region's mean
      ``log1p`` expression of the ligand genes and the receiver region's
      mean ``log1p`` expression of the receptor genes. Arithmetic
      mean-then-product is algebraically equivalent to product-then-mean
      under arithmetic means.
      Tests whether NeuronChat's interaction model contributes beyond a
      raw sender-by-receiver expression product.

  ``spatial_gene_coexpression``
      Per edge, a single scalar Pearson correlation between the two
      regions' per-gene expression profiles across the ligand-receptor
      gene panel. Single-column feature matrix. Direct implementation of
      the ``spatial gene co-expression`` baseline: no LR-pair or
      cell-type information.

Usage
-----
Build the baseline alignment pickle (shared across SC and FC targets)
and exit without training:

    python src/pipeline/connectivity_prediction/coexpression_baseline.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --baseline lr_expression_product \\
        --target sc \\
        --build-only

Build + train a single target (reuses the cached baseline pickle if
one already exists under
``<out>/coexpression_baseline/<baseline>/aligned_<baseline>.pkl``):

    python src/pipeline/connectivity_prediction/coexpression_baseline.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --baseline lr_expression_product \\
        --target sc

Build + train **both** targets in one invocation. The baseline
alignment pickle is built once; SC and FC training loops share it in
memory without re-reading from disk between targets:

    python src/pipeline/connectivity_prediction/coexpression_baseline.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --baseline region_collapsed_nc \\
        --target sc,fc

Force a rebuild of the cached alignment pickle (e.g., after editing
feature-construction code):

    python src/pipeline/connectivity_prediction/coexpression_baseline.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --baseline region_collapsed_nc --target sc,fc --rebuild

Dry run (build features, then 3 LOBO folds per target only):

    python src/pipeline/connectivity_prediction/coexpression_baseline.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --baseline lr_expression_product --target sc --dry-run

Required inputs
---------------
``region_collapsed_nc``:
      cfg.data.nc_h5 (existing), cfg.data.alignment_pkl (existing,
      edge_table).
``lr_expression_product`` and ``spatial_gene_coexpression``:
      per-region per-gene expression. Read from
      ``cfg.data.abc_expression_h5ad`` -- the annotated ABC snRNA-seq
      anndata scoped to the expanded DB's ligand + receptor genes (447
      genes covering every L / R subunit of the 1092-LR DB). Region
      labels in ``.obs`` use ``region_of_interest_label`` (``Human ``
      prefix stripped). Cells are grouped by region and aggregated to
      ``log1p(mean(raw counts))`` per region x gene. The LR-pair table
      (with ligand_genes / receptor_genes) is read from
      ``src/neuronchat/data/merged_interactionDB_human_1092LR.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from clrc.core.io import (
    find_repo_root,
    load_alignment_data,
    load_yaml_config,
    safe_filename,
    save_json,
    save_pickle,
    stable_hash_int,
    timestamp,
)
from clrc.core.logging import setup_logging
from clrc.core.metrics import mae, rmse
from clrc.core.types import AlignmentData, FeatureMeta, FoldArtifact
from clrc.features.coexpression import (
    build_lr_expression_product,
    build_region_collapsed_nc,
    build_spatial_gene_coexpression,
)
from clrc.prediction.lobo import infer_regions, iter_lobo_folds, precompute_fold_masks
from clrc.prediction.xgboost import train_predict_xgb


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Per-region gene expression (inputs for lr_expression_product and
#  spatial_gene_coexpression)
# ---------------------------------------------------------------------------

# ABC.h5ad region labels are prefixed "Human " — the main pipeline's region
# names (from the SC H5) are the un-prefixed canonical labels. Strip here
# rather than carrying a dedicated alias dict, matching how restrict_to_ABC
# already handles the A24 -> ACC rename.
_REGION_PREFIX = "Human "


def _resolve_expression_cache_path(cfg: dict, repo_root: Path) -> Path:
    """Shared parquet cache path for the per-region log1p expression frame.

    Always ``<cfg.output.base_dir>/coexpression_baseline/abc_region_expression_log1p.parquet``,
    independent of ``--output-dir`` overrides so every (baseline, target)
    invocation reads and writes the same cache. The parquet is computed
    once (~3.5 min on the 3.37 M-cell h5ad) and reused by both
    ``lr_expression_product`` and ``spatial_gene_coexpression`` for both
    SC and FC without re-aggregation.
    """
    base = Path(cfg["output"]["base_dir"])
    if not base.is_absolute():
        base = repo_root / base
    return base / "coexpression_baseline" / "abc_region_expression_log1p.parquet"


def _resolve_h5ad_path(cfg: dict, repo_root: Path) -> Path:
    """Resolve the per-cell ABC expression h5ad path from the config.

    The expanded-DB pipeline requires an anndata whose gene set covers every
    ligand + receptor subunit of the 1092-LR database (447 unique genes).
    Fails loud if the config key is missing so we never silently fall back to
    a narrower gene panel that would invalidate the lr_expression_product
    and spatial_gene_coexpression baselines.
    """
    try:
        h5ad_rel = cfg["data"]["abc_expression_h5ad"]
    except KeyError as exc:
        raise KeyError(
            "Config missing 'data.abc_expression_h5ad' — point this at the "
            "annotated ABC snRNA-seq anndata scoped to the expanded-DB L+R "
            "genes (447 genes, 100% coverage of the 1092-LR database). "
            "Example: data/ABC_neuronchat/ABC_neuronchat_annotated.h5ad."
        ) from exc
    h5ad_path = Path(h5ad_rel)
    if not h5ad_path.is_absolute():
        h5ad_path = repo_root / h5ad_path
    if not h5ad_path.is_file():
        raise FileNotFoundError(
            f"cfg.data.abc_expression_h5ad does not exist: {h5ad_path}. "
            f"Verify the path and gene-set coverage before running."
        )
    return h5ad_path


def build_abc_region_expression(
    h5ad_path: Path,
    expected_regions: List[str],
    cache_path: Optional[Path] = None,
    *,
    log1p: bool = True,
) -> pd.DataFrame:
    """Aggregate per-cell ABC Atlas expression to per-region means.

    Groups cells by ``obs.region_of_interest_label`` (with 'Human ' prefix
    stripped), computes the arithmetic mean of raw counts across cells in
    each region, then optionally applies log1p. Columns are gene symbols
    (from ``var.gene_symbol``).

    Parameters
    ----------
    h5ad_path
        Path to ``data/UMAP_ANNDATA/ABC.h5ad``.
    expected_regions
        ABC_regions_cci from the main alignment pickle. Used to verify the
        aggregation covers every edge-table region. Raises if any region is
        missing from the h5ad (do not silently drop -- we must not
        fabricate expression data to cover gaps in the panel).
    cache_path
        Optional parquet path for caching the aggregated DataFrame. Reused
        on subsequent calls if present.
    log1p
        When True (default), apply ``np.log1p`` after mean
        aggregation. Set False to get raw-count means.
    """
    if cache_path is not None and cache_path.exists():
        logger.info("Loading cached per-region expression from %s", cache_path)
        df = pd.read_parquet(cache_path)
        missing = [r for r in expected_regions if r not in df.index]
        if missing:
            raise KeyError(
                f"Cached expression DataFrame missing regions: {missing}. "
                f"Delete the cache at {cache_path} to rebuild."
            )
        return df

    import anndata as ad  # imported lazily — heavy dep

    logger.info("Aggregating per-region expression from %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    # Strip "Human " prefix
    region_series = (
        adata.obs["region_of_interest_label"]
        .astype(str)
        .str.replace(f"^{_REGION_PREFIX}", "", regex=True)
    )
    gene_symbols = adata.var["gene_symbol"].astype(str).tolist()

    unique_regions = sorted(region_series.unique())
    missing = [r for r in expected_regions if r not in unique_regions]
    if missing:
        raise KeyError(
            f"ABC.h5ad does not contain regions required by the alignment "
            f"pickle: {missing}. Halting -- do NOT fabricate per-region "
            f"expression."
        )

    # Aggregate in chunks to avoid materialising the full sparse matrix.
    n_genes = adata.shape[1]
    agg = pd.DataFrame(
        np.zeros((len(unique_regions), n_genes), dtype=np.float64),
        index=unique_regions,
        columns=gene_symbols,
    )
    counts = pd.Series(0, index=unique_regions, dtype=np.int64)

    region_to_rows: dict[str, np.ndarray] = {}
    for r in unique_regions:
        region_to_rows[r] = np.flatnonzero((region_series == r).to_numpy())

    for r, rows in region_to_rows.items():
        # adata.X is sparse in backed mode — slice efficiently
        X_r = adata.X[rows, :]
        # Convert to dense for aggregation (n_cells_region × n_genes; manageable
        # on ABC: max ~100k cells × 285 genes ≈ 220 MB as float64).
        if hasattr(X_r, "toarray"):
            X_r_dense = X_r.toarray()
        else:
            X_r_dense = np.asarray(X_r)
        agg.loc[r, :] = X_r_dense.mean(axis=0)
        counts[r] = X_r_dense.shape[0]
        logger.debug("region %s: %d cells", r, counts[r])

    logger.info(
        "Aggregated %d regions, %d genes. Per-region cell counts: min=%d, max=%d",
        agg.shape[0], agg.shape[1], int(counts.min()), int(counts.max()),
    )

    if log1p:
        agg.loc[:, :] = np.log1p(agg.values)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(cache_path)
        logger.info("Cached per-region expression to %s", cache_path)
    return agg


def load_lr_pair_table(json_path: Path) -> pd.DataFrame:
    """Load the LR pair table with ligand / receptor gene lists.

    Returns a DataFrame with columns ``lr_name``, ``ligand_genes``,
    ``receptor_genes`` where the gene columns are list[str]. Source is the
    merged interaction DB used to build the expanded NC H5 (1092 LR pairs).
    """
    with json_path.open() as f:
        db = json.load(f)
    rows = []
    for lr_name in sorted(db.keys()):
        entry = db[lr_name]
        rows.append(
            {
                "lr_name": lr_name,
                "ligand_genes": list(entry["lig_contributor"]),
                "receptor_genes": list(entry["receptor_subunit"]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Feature matrix assembly
# ---------------------------------------------------------------------------

def _meta_for_lr(
    feature_names: List[str],
    lr_pair_table: Optional[pd.DataFrame],
) -> List[FeatureMeta]:
    """Build FeatureMeta list for an LR-indexed feature matrix.

    Used by the ``region_collapsed_nc`` and ``lr_expression_product``
    baselines: each column corresponds to exactly one LR pair (no
    cell-type resolution). ct_L / ct_R are None by design.
    """
    lr_to_genes = {}
    if lr_pair_table is not None:
        for _, row in lr_pair_table.iterrows():
            lg = row["ligand_genes"]
            rg = row["receptor_genes"]
            lr_to_genes[row["lr_name"]] = (
                "+".join(lg) if isinstance(lg, (list, tuple)) else str(lg),
                "+".join(rg) if isinstance(rg, (list, tuple)) else str(rg),
            )
    meta: List[FeatureMeta] = []
    for i, name in enumerate(feature_names):
        lg_str, rg_str = lr_to_genes.get(name, (None, None))
        meta.append(
            {
                "feature_name": name,
                "lr_name": name,
                "ct_L": None,
                "ct_R": None,
                "lr_index": i,
                "ligand_genes": lg_str,
                "receptor_genes": rg_str,
            }
        )
    return meta


def _meta_for_spatial_gene_coexpression() -> List[FeatureMeta]:
    """Single pooled scalar feature: no LR or cell-type resolution at all."""
    return [
        {
            "feature_name": "spatial_gene_coexpression",
            "lr_name": None,
            "ct_L": None,
            "ct_R": None,
            "lr_index": None,
            "ligand_genes": None,
            "receptor_genes": None,
        }
    ]


def drop_all_nan_columns(
    X: np.ndarray,
    feature_names: List[str],
    meta: List[FeatureMeta],
) -> Tuple[np.ndarray, List[str], List[FeatureMeta]]:
    """Drop columns that are entirely NaN.

    For ``lr_expression_product`` this handles any LR pair whose ligand or
    receptor gene set has zero overlap with the expression panel; for
    ``region_collapsed_nc`` a NaN column means the underlying NC matrix
    had no non-zero values for that pair.
    """
    keep = ~np.all(np.isnan(X), axis=0)
    if keep.all():
        return X, feature_names, meta
    n_dropped = int((~keep).sum())
    logger.info("Dropping %d all-NaN feature columns.", n_dropped)
    return (
        X[:, keep],
        [n for n, k in zip(feature_names, keep) if k],
        [m for m, k in zip(meta, keep) if k],
    )


def build_baseline_payload(
    baseline: str,
    cfg: dict,
) -> dict:
    """Build the alignment-pickle payload for the requested baseline.

    Reuses the edge_table, SC/FC target matrices, distance vectors, and
    metric_names from the main alignment pickle — only ``X_kept_np``,
    ``feature_names_kept`` and ``meta_ABC_kept`` are replaced. The
    payload is target-agnostic (SC and FC target matrices are both
    inside ``align``), so a single build can serve training for every
    target via :func:`clrc.core.io.load_alignment_data`.
    """
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    with align_pkl.open("rb") as f:
        align = pickle.load(f)

    edge_table = align["edge_table"]
    if not isinstance(edge_table, pd.DataFrame):
        edge_table = pd.DataFrame(edge_table)
    abc_regions_cci = list(align["ABC_regions_cci"])

    repo_root = find_repo_root()
    lr_db_path = repo_root / "src/neuronchat/data/merged_interactionDB_human_1092LR.json"

    if baseline == "region_collapsed_nc":
        X_new, names_new = build_region_collapsed_nc(
            cfg["data"]["nc_h5"], edge_table
        )
        lr_tbl = load_lr_pair_table(lr_db_path)
        meta_new = _meta_for_lr(names_new, lr_tbl)

    elif baseline == "lr_expression_product":
        lr_tbl = load_lr_pair_table(lr_db_path)
        h5ad_path = _resolve_h5ad_path(cfg, repo_root)
        cache_path = _resolve_expression_cache_path(cfg, repo_root)
        expr = build_abc_region_expression(
            h5ad_path, abc_regions_cci, cache_path=cache_path, log1p=True
        )
        # Restrict expression to the ABC regions in the edge table
        expr = expr.loc[abc_regions_cci, :]
        X_new, names_new = build_lr_expression_product(expr, lr_tbl, edge_table)
        meta_new = _meta_for_lr(names_new, lr_tbl)

    elif baseline == "spatial_gene_coexpression":
        lr_tbl = load_lr_pair_table(lr_db_path)
        h5ad_path = _resolve_h5ad_path(cfg, repo_root)
        cache_path = _resolve_expression_cache_path(cfg, repo_root)
        expr = build_abc_region_expression(
            h5ad_path, abc_regions_cci, cache_path=cache_path, log1p=True
        )
        expr = expr.loc[abc_regions_cci, :]
        # Use all genes in the panel that also appear as LR genes in the DB.
        # The expanded-DB ABC anndata is already scoped to the 447 L+R genes
        # of the 1092-LR database, so this intersection is just the full
        # panel; we keep it explicit so the baseline's gene scope is
        # self-documenting.
        all_lr_genes = set()
        for _, row in lr_tbl.iterrows():
            all_lr_genes.update(row["ligand_genes"])
            all_lr_genes.update(row["receptor_genes"])
        subset = sorted(all_lr_genes & set(expr.columns))
        X_new = build_spatial_gene_coexpression(expr, subset, edge_table)
        names_new = ["spatial_gene_coexpression"]
        meta_new = _meta_for_spatial_gene_coexpression()

    else:
        raise ValueError(
            f"Unknown baseline {baseline!r}; expected one of "
            f"region_collapsed_nc, lr_expression_product, "
            f"spatial_gene_coexpression."
        )

    X_new, names_new, meta_new = drop_all_nan_columns(X_new, names_new, meta_new)

    payload = dict(align)
    payload["X_kept_np"] = X_new
    payload["feature_names_kept"] = names_new
    payload["meta_ABC_kept"] = meta_new
    payload["baseline"] = baseline
    return payload


# ---------------------------------------------------------------------------
#  Training wrapper (re-uses the LOBO + XGBoost loop from train_xgboost.py)
# ---------------------------------------------------------------------------

def run_training(
    data: AlignmentData,
    cfg: dict,
    target: str,
    exp_dir: Path,
    *,
    dry_run: bool,
) -> None:
    target_cfg = cfg["xgboost"][target]
    folds_dir = exp_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    params_path = Path(target_cfg["params_json"])
    if not params_path.is_absolute():
        params_path = find_repo_root() / params_path
    with params_path.open() as f:
        params_blob = json.load(f)
    best_params_xgb = params_blob["best_params_xgb"]

    metric_names = data.metric_names
    j = metric_names.index(target_cfg["metric"])
    SC = data.SC_voxel if target_cfg["version"] == "voxel" else data.SC_naive
    y_all = SC[:, j].astype(float)

    regions_all = infer_regions(data.edge_table)
    if dry_run:
        regions_all = regions_all[:3]
        logger.info("DRY RUN: restricting LOBO to 3 holdout regions: %s", regions_all)
    fold_masks = precompute_fold_masks(data.edge_table, regions_all)

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

        split_seed = int(stable_hash_int(
            f"{cfg['xgboost']['seed']}_{holdout_region}"
        ))
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
        logger.info(
            "[%s] n_test=%d, best_iter=%d, RMSE=%.6f, MAE=%.6f",
            holdout_region, fa.n_test, best_iter, fa.fold_rmse, fa.fold_mae,
        )

    fold_summary = pd.DataFrame(
        {
            "holdout_region": [fa.holdout_region for fa in fold_artifacts],
            "n_train": [fa.n_train for fa in fold_artifacts],
            "n_test": [fa.n_test for fa in fold_artifacts],
            "fold_rmse": [fa.fold_rmse for fa in fold_artifacts],
            "fold_mae": [fa.fold_mae for fa in fold_artifacts],
            "best_iteration": [fa.best_iteration for fa in fold_artifacts],
        }
    ).sort_values("holdout_region")
    fold_summary.to_csv(exp_dir / "full_lobo_fold_summary.csv", index=False)
    logger.info(
        "Training complete. %d folds saved to %s (dry_run=%s)",
        len(fold_artifacts), exp_dir, dry_run,
    )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

_VALID_TARGETS = ("sc", "fc")


def _parse_targets(raw: str) -> List[str]:
    """Parse a ``--target sc``, ``--target sc,fc``, or ``--target fc,sc``
    argument into a validated list. Preserves input order, de-duplicates.
    """
    parts = [t.strip().lower() for t in raw.split(",") if t.strip()]
    seen: List[str] = []
    for t in parts:
        if t not in _VALID_TARGETS:
            raise ValueError(
                f"Unknown target {t!r}; expected one of {_VALID_TARGETS}."
            )
        if t not in seen:
            seen.append(t)
    if not seen:
        raise ValueError("--target must name at least one of sc, fc.")
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Co-expression baseline driver"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=[
            "region_collapsed_nc",
            "lr_expression_product",
            "spatial_gene_coexpression",
        ],
        required=True,
        help="Which baseline to run (see module docstring).",
    )
    parser.add_argument(
        "--target", required=True,
        help=(
            "Connectivity target(s) for training. Accepts a single value "
            "(``sc`` or ``fc``) or a comma-separated list (``sc,fc``) to "
            "train both targets from the same cached baseline alignment "
            "pickle in one invocation."
        ),
    )
    parser.add_argument(
        "--build-only", action="store_true",
        help=(
            "Build and save the baseline alignment pickle, then exit. "
            "Target-level arguments are ignored after the pickle is cached."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build features, then run 3 LOBO folds only (validation).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Override the shared baseline directory "
            "(``<out>/coexpression_baseline/<baseline>/``). Per-target "
            "training outputs are written under this directory."
        ),
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help=(
            "Force rebuild of the baseline alignment pickle even if a "
            "cached copy exists. Use after changing feature-construction "
            "code or underlying inputs."
        ),
    )
    args = parser.parse_args()

    targets = _parse_targets(args.target)

    cfg = load_yaml_config(args.config)
    out_base = Path(cfg["output"]["base_dir"])
    # The alignment pickle is baseline-level (shared across targets);
    # per-target training outputs go under the target subdirectory.
    baseline_shared_dir = (
        args.output_dir
        or (out_base / "coexpression_baseline" / args.baseline)
    )
    baseline_shared_dir.mkdir(parents=True, exist_ok=True)

    _ = setup_logging("coexpression_baseline", out_base)
    logger.info(
        "Coexpression baseline: baseline=%s targets=%s dry_run=%s build_only=%s rebuild=%s",
        args.baseline, targets, args.dry_run, args.build_only, args.rebuild,
    )

    baseline_pkl = baseline_shared_dir / f"aligned_{args.baseline}.pkl"

    # 1) Build or reuse the baseline feature matrix (alignment payload).
    if baseline_pkl.exists() and not args.rebuild:
        logger.info("Reusing cached baseline alignment pickle: %s", baseline_pkl)
    else:
        payload = build_baseline_payload(args.baseline, cfg)
        save_pickle(payload, baseline_pkl)
        logger.info(
            "Saved baseline alignment pickle: %s (X shape=%s)",
            baseline_pkl, payload["X_kept_np"].shape,
        )
        save_json(
            {
                "baseline": args.baseline,
                "n_edges": int(payload["X_kept_np"].shape[0]),
                "n_features": int(payload["X_kept_np"].shape[1]),
                "alignment_pkl": str(baseline_pkl),
            },
            baseline_shared_dir / "baseline_info.json",
        )

    if args.build_only:
        logger.info("--build-only set; exiting before training.")
        return

    # 2) Train once per requested target, reusing the cached pickle.
    for target in targets:
        target_cfg = cfg["xgboost"][target]
        target_output_dir = baseline_shared_dir / target
        target_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "--- Training target=%s (version=%s, metric=%s) ---",
            target, target_cfg["version"], target_cfg["metric"],
        )
        data = load_alignment_data(
            baseline_pkl,
            version=target_cfg["version"],
            target_scale=target_cfg.get("target_scale", 1.0),
        )
        exp_dir = target_output_dir / (
            f"{timestamp()}_{target_cfg['version']}_{safe_filename(target_cfg['metric'])}"
            + ("_dryrun" if args.dry_run else "")
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        run_training(data, cfg, target, exp_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
