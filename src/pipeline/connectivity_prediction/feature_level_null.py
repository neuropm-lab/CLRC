#!/usr/bin/env python3
"""Feature-level variogram-matched spatial null driver.

Scrambles the output X matrix directly in log-space with a
sender/receiver decomposition. Produces variogram-matched null X
matrices per surrogate in minutes (no NeuronChat rerun required).

Design: joint L/R seed, exclude self-loops, epsilon floor 1e-3,
log-space additive (= multiplicative in linear scale). First-order
log-linear model -- any non-separable region-pair interaction is
intentionally ignored (that is the spatial null hypothesis).

Pipeline
--------
1. Load alignment pickle (real X, edge_table, SC/FC targets, region
   list).
2. Build or load the 86-node unique spatial distance matrix + mapping
   artifacts. Reuses ``spatial_null.py`` outputs when available:
   ``<out>/spatial_null/unique_distance_matrix_86.npy`` and its
   sidecar CSVs. Otherwise runs ``run_build_distances`` fresh.
3. Compute ``logL``, ``logR``, ``log_baseline`` from real X.
4. For each surrogate ``s`` in ``{0, ..., N-1}``:
   - Generate ``logL_surr[s]`` and ``logR_surr[s]`` via brainSMASH
     (joint seed).
   - Reconstruct ``X_null[s]`` by log-additive combination then
     ``exp``.
   - Save to ``<out>/feature_null/null_features/{s}.npy``.
5. Train LOBO XGBoost with HPO-best params on each cached X_null for
   each target (sc, fc). Writes per-surrogate metrics to
   ``<out>/feature_null/null_metrics/{target}/{s}.csv`` and
   appends to ``combined_null_metrics_{target}.csv``.

Outputs
-------
- ``<out>/feature_null/null_features/{s}.npy`` (shared across targets)
- ``<out>/feature_null/null_metrics/{sc,fc}/{s}.csv``
- ``<out>/feature_null/combined_null_metrics_{sc,fc}.csv``
- ``<out>/feature_null/logL_real.npy``, ``logR_real.npy``,
  ``log_baseline_real.npy`` (the decomposed real maps, for reference).

CLI examples
------------
Full path, N=100, single GPU::

    uv run python src/pipeline/connectivity_prediction/feature_level_null.py \\
        --config configs/abc_expanded_hpobest.yaml \\
        --n-surrogates 100 --seed 0

Split across two GPUs by surrogate-indices (parallel)::

    CUDA_VISIBLE_DEVICES=0 uv run python src/pipeline/connectivity_prediction/feature_level_null.py \\
        --config ... --surrogate-indices 0..49
    CUDA_VISIBLE_DEVICES=1 uv run python src/pipeline/connectivity_prediction/feature_level_null.py \\
        --config ... --surrogate-indices 50..99

Generate features only (skip LOBO; useful for a quick sanity pass)::

    uv run python src/pipeline/connectivity_prediction/feature_level_null.py \\
        --config ... --n-surrogates 100 --stage features_only
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from clrc.core.io import find_repo_root, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.spatial.atlas import build_unique_spatial_nodes
from clrc.spatial.feature_null import (
    compute_sender_receiver_log_maps,
    reconstruct_null_X_surrogate,
    surrogate_log_maps_variogram_matched,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module loader for spatial_null.py
#
# Loads by file path rather than a normal import, since pipeline/
# subdirs are not set up as an importable package.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPATIAL_NULL_PATH = _REPO_ROOT / "src/pipeline/connectivity_prediction/spatial_null.py"


def _load_spatial_null_module():
    spec = importlib.util.spec_from_file_location(
        "pipeline_spatial_null_for_feature_level_null", _SPATIAL_NULL_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Distance / mapping setup
# ---------------------------------------------------------------------------


def _resolve_out_root(cfg: dict) -> Path:
    out_root = Path(cfg["output"]["base_dir"])
    if not out_root.is_absolute():
        out_root = find_repo_root() / out_root
    return out_root


def _load_or_build_unique_spatial_artifacts(
    cfg: dict, stage_out_dir: Path
):
    """Reuse <out>/spatial_null/ artifacts if present; otherwise delegate
    to spatial_null.run_build_distances which has the canonical
    mapping + alignment logic."""
    spatial_null_mod = _load_spatial_null_module()

    # Build or reuse the distance matrix under <out>/spatial_null/,
    # which is the canonical location shared with spatial_null.py.
    sn_dir = stage_out_dir.parent / "spatial_null"
    dist_path = sn_dir / "unique_distance_matrix_86.npy"

    abc_regions_cci = spatial_null_mod._load_abc_regions_cci(cfg)
    mapping, voxel_counts = spatial_null_mod._load_and_align_mapping(cfg, abc_regions_cci)
    unique_nodes_df = build_unique_spatial_nodes(mapping)

    if dist_path.is_file():
        logger.info(
            "Reusing existing 86-node distance matrix at %s", dist_path,
        )
        distance_86 = np.load(dist_path)
    else:
        logger.info(
            "No cached 86-node distance matrix; rebuilding via "
            "spatial_null.run_build_distances -> %s", sn_dir,
        )
        sn_dir.mkdir(parents=True, exist_ok=True)
        spatial_null_mod.run_build_distances(cfg, sn_dir)
        distance_86 = np.load(dist_path)

    return {
        "abc_regions": abc_regions_cci,
        "distance_86": distance_86,
        "unique_nodes_df": unique_nodes_df,
        "voxel_counts": voxel_counts,
        "abc_to_allen_idx": mapping["abc_to_allen_idx"],
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_feature_level_null(
    cfg: dict,
    *,
    n_surrogates: int,
    surrogate_indices: Sequence[int] | None,
    seed_base: int,
    eps: float,
    stage: str,
    lobo_region_subset: Sequence[str] | None,
    target: str | None = None,
    n_jobs: int = 1,
) -> None:
    from clrc.core.io import load_alignment_data

    out_root = _resolve_out_root(cfg)
    stage_out_dir = out_root / "feature_null"
    stage_out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("feature_level_null", output_dir=stage_out_dir)

    # Fast-path: --stage lobo_only just trains on existing cached
    # null_features/{s}.npy for the selected target. Used by the 2-GPU
    # parallel launch pattern (one target per GPU, one tmux each).
    if stage == "lobo_only":
        if target is None:
            raise ValueError("--stage lobo_only requires --target sc|fc.")
        spatial_null_mod = _load_spatial_null_module()
        logger.info("[%s] Running LOBO on cached feature_null files...",
                    target.upper())
        spatial_null_mod.run_lobo_only(
            cfg=cfg,
            stage_out_dir=stage_out_dir,
            target=target,
            surrogate_indices=surrogate_indices,
            lobo_region_subset=lobo_region_subset,
        )
        return

    # 1. Real alignment data.
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    data_align = load_alignment_data(
        align_pkl,
        version=cfg["xgboost"]["sc"]["version"],
        target_scale=1.0,  # per-target scaling happens inside the LOBO helper
    )
    n_edges, n_features = data_align.X.shape
    logger.info("Real X: %d edges, %d features.", n_edges, n_features)

    # 2. Spatial artifacts (reuse or build).
    spatial = _load_or_build_unique_spatial_artifacts(cfg, stage_out_dir)
    abc_regions = spatial["abc_regions"]
    # edge_table uses src_region / tgt_region in the real alignment pickle;
    # the feature_null helpers accept either naming.
    _et_regions = set()
    for _col in ("sender_region", "src_region", "receiver_region", "tgt_region"):
        if _col in data_align.edge_table.columns:
            _et_regions.update(data_align.edge_table[_col].unique().tolist())
    if set(abc_regions) != _et_regions:
        # Not necessarily fatal if edge_table uses renamed aliases, but log.
        logger.warning(
            "abc_regions from spatial mapping (n=%d) does not match "
            "edge_table regions exactly. Proceeding with mapping's ordering.",
            len(abc_regions),
        )

    # 3. Decompose real X.
    t0 = time.perf_counter()
    logL, logR, log_baseline = compute_sender_receiver_log_maps(
        data_align.X,
        data_align.edge_table,
        regions=abc_regions,
        eps=eps,
        exclude_self_loops=True,
    )
    logger.info(
        "Decomposed real X in %.2fs. logL/logR shape=%s.",
        time.perf_counter() - t0, logL.shape,
    )
    np.save(stage_out_dir / "logL_real.npy", logL)
    np.save(stage_out_dir / "logR_real.npy", logR)
    np.save(stage_out_dir / "log_baseline_real.npy", log_baseline)

    # 4. Generate surrogates.
    if surrogate_indices is None:
        surrogate_indices = list(range(int(n_surrogates)))
    else:
        surrogate_indices = list(surrogate_indices)
        if max(surrogate_indices) >= n_surrogates:
            # Allow surrogate_indices to drive the requested count if it
            # goes higher than the --n-surrogates flag; informative log.
            logger.info(
                "surrogate_indices max=%d >= n_surrogates=%d; generating "
                "at the union size.",
                max(surrogate_indices), n_surrogates,
            )
            n_surrogates = max(n_surrogates, max(surrogate_indices) + 1)

    features_dir = stage_out_dir / "null_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Compute per-feature surrogates. Memory: (n_surr_max, n_features, 101) at
    # float64 = 100*3992*101*8 = ~320 MB per matrix; two of them => ~640 MB.
    # Feasible.
    logger.info(
        "Generating variogram-matched surrogates: n_surrogates=%d, "
        "n_features=%d, joint seed, seed_base=%d.",
        n_surrogates, n_features, seed_base,
    )
    t0 = time.perf_counter()
    logL_surr, logR_surr = surrogate_log_maps_variogram_matched(
        logL, logR,
        distance_86=spatial["distance_86"],
        unique_nodes_df=spatial["unique_nodes_df"],
        voxel_counts=spatial["voxel_counts"],
        abc_to_allen_idx=spatial["abc_to_allen_idx"],
        abc_regions_ordering=abc_regions,
        n_surrogates=n_surrogates,
        seed_base=seed_base,
        joint=True,
        n_jobs=n_jobs,
    )
    logger.info(
        "Surrogate generation done in %.1fs (%.2f s/feature avg).",
        time.perf_counter() - t0,
        (time.perf_counter() - t0) / max(1, n_features),
    )

    # Reconstruct and save X_null per selected surrogate index.
    for s_idx in surrogate_indices:
        t_s = time.perf_counter()
        X_null = reconstruct_null_X_surrogate(
            logL_surr[s_idx], logR_surr[s_idx], log_baseline,
            data_align.edge_table, abc_regions,
        )
        if X_null.shape != data_align.X.shape:
            raise RuntimeError(
                f"X_null shape {X_null.shape} != real X shape "
                f"{data_align.X.shape}. Aborting."
            )
        feat_path = features_dir / f"{s_idx}.npy"
        np.save(feat_path, X_null.astype(np.float32))
        logger.info(
            "[surrogate %d] X_null saved (shape=%s, range=[%.3g, %.3g], "
            "n_zero=%d) in %.2fs -> %s",
            s_idx, X_null.shape, float(X_null.min()), float(X_null.max()),
            int((X_null == 0).sum()),
            time.perf_counter() - t_s, feat_path,
        )

    if stage == "features_only":
        logger.info("--stage features_only: skipping LOBO.")
        return

    # 5. Train LOBO per target using the existing lobo_only machinery.
    spatial_null_mod = _load_spatial_null_module()
    for target in ("sc", "fc"):
        logger.info("[%s] Running LOBO on cached feature files...", target.upper())
        spatial_null_mod.run_lobo_only(
            cfg=cfg,
            stage_out_dir=stage_out_dir,
            target=target,
            surrogate_indices=surrogate_indices,
            lobo_region_subset=lobo_region_subset,
        )
    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_surrogate_indices(spec: str | None) -> List[int] | None:
    """Accept forms: '0,1,2', '0..49', '0-49', None."""
    if spec is None:
        return None
    spec = spec.strip()
    if ".." in spec:
        a, b = spec.split("..", 1)
        return list(range(int(a), int(b) + 1))
    if "-" in spec and spec.count("-") == 1:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="YAML config (e.g. abc_expanded_hpobest.yaml).")
    p.add_argument("--n-surrogates", type=int, default=100,
                   help="Number of surrogate draws (default 100).")
    p.add_argument("--surrogate-indices", type=str, default=None,
                   help="Restrict to specific indices: '0,1,2' or '0..49'.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base brainSMASH seed (default 0).")
    p.add_argument("--eps", type=float, default=1e-3,
                   help="Epsilon floor before log (default 1e-3).")
    p.add_argument(
        "--stage",
        choices=["all", "features_only", "lobo_only"],
        default="all",
        help=(
            "'all' generates surrogates + trains LOBO for SC and FC; "
            "'features_only' generates X_null.npy files only (skip LOBO); "
            "'lobo_only' trains LOBO for a single --target on existing "
            "null_features/{s}.npy (fast path for 2-GPU parallel launches)."
        ),
    )
    p.add_argument(
        "--target",
        choices=["sc", "fc"],
        default=None,
        help="Required with --stage lobo_only. Ignored otherwise.",
    )
    p.add_argument("--lobo-regions", type=str, default=None,
                   help="Comma-separated region subset for LOBO (dry-runs).")
    p.add_argument(
        "--n-jobs", type=int, default=16,
        help=(
            "Per-delta parallelism passed to brainSMASH during surrogate "
            "generation (default 16). Has no effect on the lobo_only stage. "
            "Set lower if stacking against another CPU-saturating job."
        ),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml_config(args.config)

    surr_idx = _parse_surrogate_indices(args.surrogate_indices)
    lobo_regions = None
    if args.lobo_regions:
        lobo_regions = [s.strip() for s in args.lobo_regions.split(",") if s.strip()]

    run_feature_level_null(
        cfg=cfg,
        n_surrogates=args.n_surrogates,
        surrogate_indices=surr_idx,
        seed_base=args.seed,
        eps=args.eps,
        stage=args.stage,
        lobo_region_subset=lobo_regions,
        target=args.target,
        n_jobs=args.n_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
