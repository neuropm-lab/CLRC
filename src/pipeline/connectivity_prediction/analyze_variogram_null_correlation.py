#!/usr/bin/env python3
"""Test whether CCI feature-target alignment exceeds a variogram-matched null.

For SC and FC targets, compute the mean absolute distance-partialled
Spearman correlation (|rho|) between each CCI feature column and the
connectivity target, aggregated across features. Repeat for each
cached null X matrix produced by ``feature_level_null.py`` (stored
at ``<out>/feature_null/null_features/{s}.npy``). Output the one-sided
empirical p-value (real vs null distribution) with add-one smoothing.

This is a standalone alignment test independent of any downstream ML
training. Its scientific content: does the real CCI feature matrix
align with connectivity targets more than variogram-matched spatial
scrambles do, once fiber distance is partialled out? This provides a
stringent control for spatial autocorrelation without requiring
per-surrogate LOBO model retraining.

Outputs written to ``<out>/feature_null/variogram_null_correlation/``:
  - ``real_vs_null_alignment.csv`` — one row per target with real_value,
    null distribution summary stats, empirical p, z-score.
  - ``null_distribution_{sc,fc}.csv`` — per-surrogate null |rho| values.

Surrogate-major loop: X is identical across SC and FC targets, so we
rank each surrogate X once and apply both per-target contexts. Per-target
state (ranked + centered y, ranked covariate, r_yc) is built once and
reused across all N surrogates via
:class:`clrc.spatial.variogram_null_correlation.PartialSpearmanContext`.

Run::

  uv run python src/pipeline/connectivity_prediction/analyze_variogram_null_correlation.py \\
      --config configs/abc_expanded_hpobest.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from clrc.core.io import find_repo_root, load_alignment_data, load_yaml_config
from clrc.core.logging import setup_logging
from clrc.spatial.variogram_null_correlation import (
    PartialSpearmanContext,
    _partial_r_from_centered,
    _rank_center_norms,
    build_partial_spearman_context,
    summarize_null_test,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------


def _resolve_out_root(cfg: dict) -> Path:
    out_root = Path(cfg["output"]["base_dir"])
    if not out_root.is_absolute():
        out_root = find_repo_root() / out_root
    return out_root


def _iter_null_feature_files(
    feature_null_dir: Path, max_surrogates: int | None = None,
) -> List[Path]:
    files = sorted(
        (feature_null_dir / "null_features").glob("*.npy"),
        key=lambda p: int(p.stem),
    )
    if not files:
        raise FileNotFoundError(
            f"No null_features/*.npy under {feature_null_dir}. "
            f"Run feature_level_null.py --stage features_only first."
        )
    if max_surrogates is not None:
        files = files[:max_surrogates]
    return files


# ---------------------------------------------------------------------------
#  Per-target data loading (y + covariates)
# ---------------------------------------------------------------------------


def _load_target_y_and_distance(
    cfg: dict, target: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(y, distance_vec, metric_name)`` for ``target``."""
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    tcfg = cfg["xgboost"][target]
    data = load_alignment_data(
        align_pkl,
        version=tcfg["version"],
        target_scale=tcfg.get("target_scale", 1.0),
    )
    metric_idx = list(data.metric_names).index(tcfg["metric"])
    SC = data.SC_voxel if tcfg["version"] == "voxel" else data.SC_naive
    y = SC[:, metric_idx].astype(np.float64)
    if data.distance_vec is None:
        raise RuntimeError(
            "Alignment pickle missing distance_vec_voxel/naive — required "
            "for distance-partialled alignment statistic."
        )
    distance_vec = np.asarray(data.distance_vec, dtype=np.float64)
    return y, distance_vec, tcfg["metric"]


def _load_shared_real_X(cfg: dict, any_target: str) -> np.ndarray:
    """Load X from the alignment pickle (identical across SC/FC targets)."""
    align_pkl = Path(cfg["data"]["alignment_pkl"])
    if not align_pkl.is_absolute():
        align_pkl = find_repo_root() / align_pkl
    tcfg = cfg["xgboost"][any_target]
    data = load_alignment_data(
        align_pkl,
        version=tcfg["version"],
        target_scale=tcfg.get("target_scale", 1.0),
    )
    return np.asarray(data.X, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Mean-|rho| for real + null, surrogate-major
# ---------------------------------------------------------------------------


def _mean_abs_from_centered(
    X_centered: np.ndarray, X_std: np.ndarray, ctx: PartialSpearmanContext,
) -> float:
    r = _partial_r_from_centered(X_centered, X_std, ctx)
    valid = np.isfinite(r)
    if not valid.any():
        return float("nan")
    return float(np.abs(r[valid]).mean())


def analyze_all_targets(
    cfg: dict,
    targets: Sequence[str],
    feature_null_dir: Path,
    max_surrogates: int | None = None,
) -> tuple[List[dict], Dict[str, pd.DataFrame]]:
    """Surrogate-major analysis across all requested targets.

    Builds one :class:`PartialSpearmanContext` per target, then iterates
    surrogate files once -- ranking each surrogate X a single time and
    applying every target context to the ranked matrix.
    """
    contexts: Dict[str, PartialSpearmanContext] = {}
    metrics: Dict[str, str] = {}
    for target in targets:
        y, distance_vec, metric = _load_target_y_and_distance(cfg, target)
        contexts[target] = build_partial_spearman_context(
            y, covariates=distance_vec.reshape(-1, 1),
        )
        metrics[target] = metric
        logger.info(
            "[%s] built partial-Spearman context: n=%d, r_yc=%.4f",
            target.upper(), contexts[target].n, contexts[target].r_yc,
        )

    # --- Real X --------------------------------------------------------
    X_real = _load_shared_real_X(cfg, targets[0])
    logger.info("Loaded real X, shape=%s", X_real.shape)

    t0 = time.perf_counter()
    X_centered, X_std = _rank_center_norms(X_real)
    logger.info(
        "Ranked + centered real X in %.2fs (shape=%s)",
        time.perf_counter() - t0, X_centered.shape,
    )

    real_values: Dict[str, float] = {}
    for target in targets:
        v = _mean_abs_from_centered(X_centered, X_std, contexts[target])
        real_values[target] = v
        logger.info(
            "[%s] real mean|rho| (distance-partialled) = %.6f",
            target.upper(), v,
        )
    del X_centered, X_std, X_real

    # --- Null surrogates ----------------------------------------------
    files = _iter_null_feature_files(feature_null_dir, max_surrogates=max_surrogates)
    n_surr = len(files)
    null_values: Dict[str, List[float]] = {t: [] for t in targets}
    t_loop_start = time.perf_counter()

    for i, path in enumerate(files):
        ts = time.perf_counter()
        X_s = np.load(path, mmap_mode=None)
        t_load = time.perf_counter() - ts

        tr = time.perf_counter()
        Xc, Xs = _rank_center_norms(X_s)
        t_rank = time.perf_counter() - tr

        per_target_vals = []
        for target in targets:
            v = _mean_abs_from_centered(Xc, Xs, contexts[target])
            null_values[target].append(v)
            per_target_vals.append(f"{target}={v:.6f}")

        if (i + 1) % 10 == 0 or i == 0 or i == n_surr - 1:
            elapsed = time.perf_counter() - t_loop_start
            rate = (i + 1) / elapsed if elapsed > 0 else float("nan")
            eta = (n_surr - (i + 1)) / rate if rate > 0 else float("nan")
            logger.info(
                "surrogate %d/%d: load=%.2fs rank=%.2fs [%s] "
                "(%.2f surr/s, ETA %.0fs)",
                i + 1, n_surr, t_load, t_rank,
                ", ".join(per_target_vals), rate, eta,
            )

    # --- Summaries ----------------------------------------------------
    summaries: List[dict] = []
    dist_dfs: Dict[str, pd.DataFrame] = {}
    for target in targets:
        null_arr = np.array(null_values[target], dtype=np.float64)
        p, z, mu, sd, med = summarize_null_test(real_values[target], null_arr)
        logger.info(
            "[%s] null distribution: n=%d, mean=%.6f, median=%.6f, std=%.6f",
            target.upper(), len(null_arr), mu, med, sd,
        )
        logger.info(
            "[%s] p (one-sided, real > null) = %.4g; z = %.3f",
            target.upper(), p, z,
        )
        summaries.append({
            "target": target,
            "metric": metrics[target],
            "n_surrogates": len(null_arr),
            "real_mean_abs_rho": real_values[target],
            "null_mean": mu,
            "null_median": med,
            "null_std": sd,
            "null_min": float(null_arr.min()),
            "null_max": float(null_arr.max()),
            "delta_real_minus_null_mean": real_values[target] - mu,
            "empirical_p_1sided": p,
            "z_score_vs_null": z,
        })
        dist_dfs[target] = pd.DataFrame({
            "surrogate_idx": np.arange(len(null_arr)),
            "target": target,
            "null_mean_abs_rho": null_arr,
        })

    return summaries, dist_dfs


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--targets", type=str, default="sc,fc",
        help="Comma-separated targets (default 'sc,fc').",
    )
    p.add_argument(
        "--feature-null-dir", type=Path, default=None,
        help=(
            "Override the feature_null directory (default "
            "<output.base_dir>/feature_null)."
        ),
    )
    p.add_argument(
        "--max-surrogates", type=int, default=None,
        help="Limit to the first N surrogates (sanity testing).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml_config(args.config)
    out_root = _resolve_out_root(cfg)
    feature_null_dir = args.feature_null_dir or (out_root / "feature_null")

    analysis_dir = feature_null_dir / "variogram_null_correlation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("analyze_variogram_null_correlation", output_dir=analysis_dir)
    logger.info("Config: %s", args.config)
    logger.info("feature_null_dir: %s", feature_null_dir)
    logger.info("analysis output dir: %s", analysis_dir)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    summaries, dist_dfs = analyze_all_targets(
        cfg, targets, feature_null_dir, max_surrogates=args.max_surrogates,
    )

    for target, dist_df in dist_dfs.items():
        dist_path = analysis_dir / f"null_distribution_{target}.csv"
        dist_df.to_csv(dist_path, index=False)
        logger.info("[%s] wrote null distribution -> %s", target.upper(), dist_path)

    summary_df = pd.DataFrame(summaries)
    summary_path = analysis_dir / "real_vs_null_alignment.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Summary -> %s", summary_path)

    print("\n" + "=" * 80)
    print("Variogram-matched spatial null test (distance-partialled mean |rho|)")
    print("=" * 80)
    show_cols = [
        "target", "n_surrogates", "real_mean_abs_rho", "null_mean",
        "null_std", "delta_real_minus_null_mean", "z_score_vs_null",
        "empirical_p_1sided",
    ]
    print(summary_df[show_cols].to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
