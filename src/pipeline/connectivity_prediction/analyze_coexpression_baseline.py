#!/usr/bin/env python3
"""Co-expression baseline comparison: does CLRC explain variance beyond
simpler transcriptomic models?

Three transcriptomic baselines are each compared against the full CLRC
model on the same connectivity target via leave-one-brain-region-out
cross-validation:

  - ``region_collapsed_nc``: per LR pair, collapse NeuronChat's
    cell-type-resolved communication to a single scalar per edge by
    taking the mean across cell-type pairs. Tests whether CLRC's
    cell-type resolution contributes predictive signal beyond the
    region-level NeuronChat output alone.

  - ``lr_expression_product``: per LR pair and per edge, the product of
    the sender region's mean ``log1p`` expression of the ligand genes
    and the receiver region's mean ``log1p`` expression of the receptor
    genes. Tests whether NeuronChat's interaction model contributes
    beyond a raw sender-x-receiver expression product.

  - ``spatial_gene_coexpression``: per edge, a single Pearson
    correlation of the two regions' per-gene expression profiles across
    the ligand-receptor gene panel. Direct implementation of the
    ``spatial gene co-expression`` baseline: a single scalar per edge,
    no LR-pair or cell-type information.

For each (baseline, target) pair, per-region fold metrics are aligned
(same LOBO splits across runs) and compared against CLRC via Wilcoxon
signed-rank on the paired fold-level differences. Reports four metrics:

  - ``rmse``: root mean squared error (lower is better).
  - ``mae``: mean absolute error (lower is better).
  - ``r2``: coefficient of determination (higher is better).
  - ``pearson_r`` / ``spearman_r``: correlation between per-edge
    predicted and observed values within a fold (higher is better).

For lower-is-better metrics, the headline one-sided p tests
``p_clrc_better`` = ``Wilcoxon(CLRC - baseline, alternative="less")``.
For higher-is-better metrics it tests
``p_clrc_better`` = ``Wilcoxon(CLRC - baseline, alternative="greater")``.
Every comparison is within a single target's own scale (no cross-target
contrast).

All metrics are computed on the same transformed scale the model was
trained on (``y_test_t`` vs ``y_pred``); per-fold predictions are
re-loaded from the fold artifact pickles for R^2 / Pearson / Spearman
because the summary CSVs only persist RMSE / MAE.

Outputs to ``<out>/coexpression_baseline/analysis/``:
  - ``fold_metrics_merged.csv`` -- long-format per (target, baseline,
    region) fold rmse / mae / r2 / pearson_r / spearman_r for CLRC and
    each baseline.
  - ``coexpression_wilcoxon.csv`` -- one row per (target, baseline,
    metric) with paired medians, Wilcoxon W, two-sided p, both
    one-sided p's, and a direction-aware ``p_clrc_better`` column.

Run::

  uv run python src/pipeline/connectivity_prediction/analyze_coexpression_baseline.py \\
      --config configs/abc_expanded_hpobest.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from clrc.core.io import find_repo_root, load_yaml_config
from clrc.core.logging import setup_logging

logger = logging.getLogger(__name__)

TARGETS = ("sc", "fc")
BASELINES = (
    "region_collapsed_nc",
    "lr_expression_product",
    "spatial_gene_coexpression",
)

# Direction for each metric: ``"lower"`` means the smaller value is better
# (CLRC wins when its per-fold metric is less than the baseline's);
# ``"higher"`` means the larger value is better.
METRIC_DIRECTIONS: Dict[str, str] = {
    "rmse": "lower",
    "mae": "lower",
    "r2": "higher",
    "pearson_r": "higher",
    "spearman_r": "higher",
}


def _resolve_out_root(cfg: dict) -> Path:
    out_root = Path(cfg["output"]["base_dir"])
    if not out_root.is_absolute():
        out_root = find_repo_root() / out_root
    return out_root


def _find_clrc_run_dir(out_root: Path, target: str) -> Path:
    """Locate the main CLRC run directory for ``target`` (not a distance variant)."""
    model_dir = out_root / "models" / target
    candidates = [
        p for p in model_dir.iterdir()
        if p.is_dir()
        and (p / "full_lobo_fold_summary.csv").is_file()
        and (p / "folds").is_dir()
        and not any(suffix in p.name for suffix in ("distance_only", "cci_distance"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No full CLRC run dir with folds/ found under {model_dir}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        logger.warning(
            "Multiple CLRC runs under %s; using newest: %s (others: %s)",
            model_dir, candidates[0].name, [p.name for p in candidates[1:]],
        )
    return candidates[0]


def _find_baseline_run_dir(
    out_root: Path, baseline: str, target: str,
) -> Path:
    parent = out_root / "coexpression_baseline" / baseline / target
    if not parent.is_dir():
        raise FileNotFoundError(f"Expected baseline run parent: {parent}")
    runs = [
        p for p in parent.iterdir()
        if p.is_dir()
        and (p / "full_lobo_fold_summary.csv").is_file()
        and (p / "folds").is_dir()
    ]
    if not runs:
        raise FileNotFoundError(f"No baseline run dir under {parent}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _fold_metrics_from_pickle(pkl_path: Path) -> Dict[str, float]:
    """Compute rmse / mae / r2 / pearson_r / spearman_r for one fold.

    Metrics are on the transformed scale ``y_test_t`` vs ``y_pred`` (same
    scale RMSE / MAE in the fold summary CSV use), so all five metrics
    are mutually comparable.
    """
    with pkl_path.open("rb") as f:
        fa = pickle.load(f)
    y = np.asarray(fa.y_test_t, dtype=np.float64)
    p = np.asarray(fa.y_pred, dtype=np.float64)
    n = int(y.size)
    if n < 2:
        return {
            "rmse": float(fa.fold_rmse),
            "mae": float(fa.fold_mae),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "n_test": n,
        }
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(invalid="ignore"):
        pear = stats.pearsonr(y, p)
        spear = stats.spearmanr(y, p)
    return {
        "rmse": float(fa.fold_rmse),
        "mae": float(fa.fold_mae),
        "r2": float(r2),
        "pearson_r": float(pear.statistic) if np.isfinite(pear.statistic) else float("nan"),
        "spearman_r": float(spear.statistic) if np.isfinite(spear.statistic) else float("nan"),
        "n_test": n,
    }


def _load_run_fold_metrics(run_dir: Path) -> pd.DataFrame:
    """Return per-region fold metrics (rmse, mae, r2, pearson_r, spearman_r)."""
    folds_dir = run_dir / "folds"
    pkls = sorted(folds_dir.glob("*.pkl"))
    rows = []
    for pkl in pkls:
        region = pkl.stem
        m = _fold_metrics_from_pickle(pkl)
        m["holdout_region"] = region
        rows.append(m)
    if not rows:
        raise RuntimeError(f"No fold pickles found under {folds_dir}")
    df = pd.DataFrame(rows)
    return df


def _load_run_fold_arrays(run_dir: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of ``(y_test_t, y_pred)`` arrays, one tuple per fold."""
    folds_dir = run_dir / "folds"
    pkls = sorted(folds_dir.glob("*.pkl"))
    if not pkls:
        raise RuntimeError(f"No fold pickles found under {folds_dir}")
    arrays: List[Tuple[np.ndarray, np.ndarray]] = []
    for pkl in pkls:
        with pkl.open("rb") as f:
            fa = pickle.load(f)
        arrays.append(
            (np.asarray(fa.y_test_t, dtype=np.float64),
             np.asarray(fa.y_pred, dtype=np.float64))
        )
    return arrays


def _pooled_metrics_from_arrays(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    n = int(y.size)
    if n < 2:
        return {
            "rmse": float("nan"), "mae": float("nan"),
            "r2": float("nan"), "pearson_r": float("nan"),
            "spearman_r": float("nan"),
        }
    resid = y - p
    rmse = float(np.sqrt(np.mean(resid * resid)))
    mae = float(np.mean(np.abs(resid)))
    ss_res = float(np.sum(resid * resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with np.errstate(invalid="ignore"):
        pear = stats.pearsonr(y, p)
        spear = stats.spearmanr(y, p)
    return {
        "rmse": rmse, "mae": mae, "r2": float(r2),
        "pearson_r": float(pear.statistic),
        "spearman_r": float(spear.statistic),
    }


def _pool_run_metrics(run_dir: Path) -> Dict[str, float]:
    """Concatenate (y_test_t, y_pred) across all folds of a run, then
    compute each metric once on the pooled vector.

    Pooled metrics are the canonical single-number LOBO summary per run
    (what most connectome-prediction papers report as the headline
    cross-validated R^2 or Pearson). Distinct from mean-across-folds
    because pooling weights regions by their test-set size (SC folds
    can differ in n_test; FC folds are balanced) and it is not a
    statistic-of-statistic.

    Returns a dict with rmse, mae, r2, pearson_r, spearman_r, plus the
    pooled sample sizes (``n_edges_pooled``, ``n_folds_pooled``).
    """
    fold_arrays = _load_run_fold_arrays(run_dir)
    y = np.concatenate([a[0] for a in fold_arrays])
    p = np.concatenate([a[1] for a in fold_arrays])
    out = _pooled_metrics_from_arrays(y, p)
    out["n_edges_pooled"] = int(y.size)
    out["n_folds_pooled"] = len(fold_arrays)
    return out


def fold_level_bootstrap(
    fold_arrays: List[Tuple[np.ndarray, np.ndarray]],
    *, n_boot: int = 1000, seed: int = 42,
) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """Fold-level bootstrap: resample held-out regions (with replacement)
    and recompute the pooled metric each iteration.

    This is the uncertainty measure consistent with the LOBO question
    ``how well does CLRC generalize to an unseen brain region?``
    because each bootstrap resample represents a hypothetical different
    set of held-out regions drawn from the same distribution. Edge-level
    bootstrap would instead ask the narrower question ``how precisely
    is the pooled metric estimated from the particular 20 k edges we
    have?``, which shrinks with n and is less meaningful here.

    Returns
    -------
    ci : dict
        ``metric -> (ci_lo_95, ci_hi_95)`` from 2.5 / 97.5 percentiles of
        the bootstrap distribution.
    dist_df : pd.DataFrame
        Long-format ``(metric, iter, value)`` bootstrap distribution, so
        callers can re-percentile or plot.
    """
    rng = np.random.default_rng(seed)
    n_folds = len(fold_arrays)
    metric_names = ("rmse", "mae", "r2", "pearson_r", "spearman_r")
    samples: Dict[str, List[float]] = {m: [] for m in metric_names}

    for _ in range(n_boot):
        idx = rng.integers(0, n_folds, size=n_folds)
        y_cat = np.concatenate([fold_arrays[i][0] for i in idx])
        p_cat = np.concatenate([fold_arrays[i][1] for i in idx])
        m = _pooled_metrics_from_arrays(y_cat, p_cat)
        for name in metric_names:
            samples[name].append(m[name])

    ci = {}
    dist_rows = []
    for name in metric_names:
        arr = np.asarray(samples[name], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size < 10:
            ci[name] = (float("nan"), float("nan"))
        else:
            ci[name] = (
                float(np.percentile(finite, 2.5)),
                float(np.percentile(finite, 97.5)),
            )
        for i, v in enumerate(arr):
            dist_rows.append({"metric": name, "iter": i, "value": float(v)})
    return ci, pd.DataFrame(dist_rows)


def _load_and_align(
    clrc_run_dir: Path, baseline_run_dir: Path,
) -> pd.DataFrame:
    c = _load_run_fold_metrics(clrc_run_dir).rename(
        columns={m: f"clrc_{m}" for m in METRIC_DIRECTIONS}
    ).rename(columns={"n_test": "clrc_n_test"})
    b = _load_run_fold_metrics(baseline_run_dir).rename(
        columns={m: f"baseline_{m}" for m in METRIC_DIRECTIONS}
    ).rename(columns={"n_test": "baseline_n_test"})
    merged = c.merge(b, on="holdout_region", how="inner")
    return merged


def _wilcoxon_paired(
    clrc_vals: np.ndarray, baseline_vals: np.ndarray, *,
    direction: str,
) -> Dict[str, float]:
    """Paired-Wilcoxon summary with direction-aware ``p_clrc_better``.

    ``direction='lower'`` means lower is better for this metric, so
    ``p_clrc_better`` is the one-sided p for ``CLRC < baseline``.
    ``direction='higher'`` means higher is better, so ``p_clrc_better``
    is the one-sided p for ``CLRC > baseline``.
    """
    diffs = np.asarray(clrc_vals, dtype=np.float64) - np.asarray(baseline_vals, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    non_zero = diffs[diffs != 0.0]
    n_eff = int(non_zero.size)
    result = {
        "n_eff_wilcoxon": n_eff,
        "wilcoxon_W": float("nan"),
        "p_two_sided": float("nan"),
        "p_clrc_less_baseline": float("nan"),
        "p_clrc_greater_baseline": float("nan"),
        "p_clrc_better": float("nan"),
    }
    if n_eff < 3:
        return result
    ts = stats.wilcoxon(non_zero, alternative="two-sided")
    lt = stats.wilcoxon(non_zero, alternative="less")
    gt = stats.wilcoxon(non_zero, alternative="greater")
    result.update({
        "wilcoxon_W": float(ts.statistic),
        "p_two_sided": float(ts.pvalue),
        "p_clrc_less_baseline": float(lt.pvalue),
        "p_clrc_greater_baseline": float(gt.pvalue),
    })
    if direction == "lower":
        result["p_clrc_better"] = float(lt.pvalue)
    elif direction == "higher":
        result["p_clrc_better"] = float(gt.pvalue)
    else:
        raise ValueError(f"Unknown direction {direction!r}")
    return result


def _augment_with_bootstrap(
    run_dir: Path, row: dict, *, n_boot: int, seed: int,
) -> pd.DataFrame:
    """Add fold-level bootstrap 95% CI columns to ``row`` and return the
    bootstrap distribution DataFrame for logging / plotting downstream.
    """
    fold_arrays = _load_run_fold_arrays(run_dir)
    ci, dist_df = fold_level_bootstrap(
        fold_arrays, n_boot=n_boot, seed=seed,
    )
    for metric, (lo, hi) in ci.items():
        row[f"{metric}_ci_lo_95"] = lo
        row[f"{metric}_ci_hi_95"] = hi
    dist_df = dist_df.assign(target=row["target"], source=row["source"])
    return dist_df


def run_analysis(
    cfg: dict, *, n_bootstrap: int = 1000, bootstrap_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_root = _resolve_out_root(cfg)
    rows: List[dict] = []
    merged_frames: List[pd.DataFrame] = []
    pooled_rows: List[dict] = []
    bootstrap_dist_frames: List[pd.DataFrame] = []

    # Pool CLRC metrics once per target (independent of which baseline we
    # contrast against -- CLRC is the same run for all three contrasts).
    pooled_clrc: Dict[str, Dict[str, float]] = {}
    clrc_dirs: Dict[str, Path] = {}
    for target in TARGETS:
        clrc_dir = _find_clrc_run_dir(out_root, target)
        clrc_dirs[target] = clrc_dir
        logger.info("[%s] CLRC run dir: %s", target.upper(), clrc_dir)
        pooled_clrc[target] = _pool_run_metrics(clrc_dir)
        row = {
            "target": target.upper(), "source": "clrc",
            "run_dir": str(clrc_dir), **pooled_clrc[target],
        }
        logger.info("[%s] bootstrapping CLRC (n=%d)...", target.upper(), n_bootstrap)
        bootstrap_dist_frames.append(
            _augment_with_bootstrap(
                clrc_dir, row, n_boot=n_bootstrap, seed=bootstrap_seed,
            )
        )
        pooled_rows.append(row)

    for target in TARGETS:
        clrc_dir = clrc_dirs[target]
        for baseline in BASELINES:
            baseline_dir = _find_baseline_run_dir(out_root, baseline, target)
            logger.info(
                "[%s] %s run dir: %s", target.upper(), baseline, baseline_dir,
            )
            merged = _load_and_align(clrc_dir, baseline_dir)
            merged = merged.assign(target=target, baseline=baseline)
            merged_frames.append(merged)
            pooled_base = _pool_run_metrics(baseline_dir)
            row = {
                "target": target.upper(), "source": baseline,
                "run_dir": str(baseline_dir), **pooled_base,
            }
            logger.info(
                "[%s] bootstrapping %s (n=%d)...",
                target.upper(), baseline, n_bootstrap,
            )
            bootstrap_dist_frames.append(
                _augment_with_bootstrap(
                    baseline_dir, row, n_boot=n_bootstrap, seed=bootstrap_seed,
                )
            )
            pooled_rows.append(row)
            n_folds = len(merged)
            for metric, direction in METRIC_DIRECTIONS.items():
                clrc_col = f"clrc_{metric}"
                base_col = f"baseline_{metric}"
                clrc_vals = merged[clrc_col].to_numpy(dtype=np.float64)
                base_vals = merged[base_col].to_numpy(dtype=np.float64)
                w = _wilcoxon_paired(clrc_vals, base_vals, direction=direction)
                clrc_valid = clrc_vals[~np.isnan(clrc_vals)]
                base_valid = base_vals[~np.isnan(base_vals)]
                if direction == "lower":
                    folds_better = int(np.sum(clrc_vals < base_vals))
                else:
                    folds_better = int(np.sum(clrc_vals > base_vals))
                rows.append({
                    "target": target.upper(),
                    "baseline": baseline,
                    "metric": metric,
                    "direction": direction,
                    "n_folds": n_folds,
                    "clrc_mean": float(np.mean(clrc_valid)) if clrc_valid.size else float("nan"),
                    "clrc_median": float(np.median(clrc_valid)) if clrc_valid.size else float("nan"),
                    "baseline_mean": float(np.mean(base_valid)) if base_valid.size else float("nan"),
                    "baseline_median": float(np.median(base_valid)) if base_valid.size else float("nan"),
                    "delta_mean_clrc_minus_baseline": float(np.mean(clrc_vals - base_vals)),
                    "delta_median_clrc_minus_baseline": float(np.median(clrc_vals - base_vals)),
                    "folds_clrc_better": folds_better,
                    **w,
                })

    summary = pd.DataFrame(rows)
    merged_long = pd.concat(merged_frames, ignore_index=True)
    pooled = pd.DataFrame(pooled_rows)
    bootstrap_dist = (
        pd.concat(bootstrap_dist_frames, ignore_index=True)
        if bootstrap_dist_frames else pd.DataFrame()
    )
    return summary, merged_long, pooled, bootstrap_dist


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Fold-level bootstrap iterations for 95%% CIs on pooled metrics.",
    )
    p.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="Seed for the bootstrap RNG.",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml_config(args.config)
    out_root = _resolve_out_root(cfg)

    analysis_dir = out_root / "coexpression_baseline" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("analyze_coexpression_baseline", output_dir=analysis_dir)
    logger.info("Analysis output dir: %s", analysis_dir)
    logger.info(
        "Fold-level bootstrap: n=%d, seed=%d",
        args.n_bootstrap, args.bootstrap_seed,
    )

    summary, merged_long, pooled, bootstrap_dist = run_analysis(
        cfg, n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed,
    )

    merged_path = analysis_dir / "fold_metrics_merged.csv"
    merged_long.to_csv(merged_path, index=False)
    logger.info(
        "wrote merged fold metrics -> %s (%d rows)", merged_path, len(merged_long),
    )

    summary_path = analysis_dir / "coexpression_wilcoxon.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(
        "wrote Wilcoxon summary -> %s (%d rows)", summary_path, len(summary),
    )

    pooled_path = analysis_dir / "pooled_metrics.csv"
    pooled.to_csv(pooled_path, index=False)
    logger.info(
        "wrote pooled metrics (with fold-level bootstrap 95%% CIs) -> %s (%d rows)",
        pooled_path, len(pooled),
    )

    bootstrap_dist_path = analysis_dir / "pooled_metrics_bootstrap_distribution.csv"
    bootstrap_dist.to_csv(bootstrap_dist_path, index=False)
    logger.info(
        "wrote bootstrap distribution -> %s (%d rows)",
        bootstrap_dist_path, len(bootstrap_dist),
    )

    print("\n" + "=" * 80)
    print("Co-expression baselines vs CLRC (paired-fold Wilcoxon)")
    print("Direction-aware p_clrc_better: one-sided test in the 'better' direction")
    print("  (lower for rmse/mae; higher for r2/pearson/spearman)")
    print("=" * 80)
    show = summary[[
        "target", "baseline", "metric", "direction", "n_folds",
        "clrc_median", "baseline_median",
        "delta_median_clrc_minus_baseline",
        "folds_clrc_better", "p_two_sided", "p_clrc_better",
    ]]
    print(show.to_string(index=False))
    print()

    print("=" * 80)
    print(
        "Pooled held-out metrics per run (concat y_test_t, y_pred across folds) "
        "with fold-level bootstrap 95% CIs"
    )
    print("=" * 80)
    # Compose a human-readable display with point [lo, hi] per metric.
    human_rows = []
    for _, r in pooled.iterrows():
        human = {
            "target": r["target"], "source": r["source"],
            "n_edges_pooled": int(r["n_edges_pooled"]),
            "n_folds_pooled": int(r["n_folds_pooled"]),
        }
        for metric in ("rmse", "mae", "r2", "pearson_r", "spearman_r"):
            human[metric] = (
                f"{r[metric]:.4f} [{r[f'{metric}_ci_lo_95']:.4f}, "
                f"{r[f'{metric}_ci_hi_95']:.4f}]"
            )
        human_rows.append(human)
    print(pd.DataFrame(human_rows).to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
