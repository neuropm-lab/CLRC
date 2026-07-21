#!/usr/bin/env python3
"""Cross-prediction / bias-validation analysis.

Reads the six LOBO bias-validation runs produced by
``cross_prediction.py`` --- for each target T ∈ {SC, FC} and each LR
subset ∈ {sc_biased, fc_biased, balanced}, one run with per-region fold
metrics (RMSE, MAE). This driver runs the statistical comparison that
tests whether predictive performance is **bias-specific** (i.e. whether
a subset biased toward target T actually predicts T more accurately
than a subset biased toward the other target).

Two complementary contrasts are emitted:

(A) Within-target, across-subset (scale-safe primary).
    For each target T, paired Wilcoxon signed-rank tests on per-region
    fold RMSE / MAE comparing every pair of subsets (sc_biased vs
    fc_biased, sc_biased vs balanced, fc_biased vs balanced). This is
    the direct test of bias specificity: within a target's own scale,
    does the "right" subset beat the "wrong" one?

(B) Cross-target, within-subset (scale-normalized secondary).
    Direct cross-target RMSE comparison is **meaningless** because SC
    and FC have different target scales (target_scale=100 on SC, 1 on
    FC by config). To make cross-target comparable we first divide
    each (subset, target, region) fold RMSE by the balanced-subset
    fold RMSE on the same (target, region). The normalized quantity
    "rel_rmse" is a per-region multiple of the balanced benchmark;
    rel_rmse = 1.0 means a subset matches balanced. We then paired-
    Wilcoxon rel_rmse(subset, T=SC, r) vs rel_rmse(subset, T=FC, r)
    per-subset, testing whether a subset's degradation vs balanced is
    the same on both targets. The balanced subset itself is degenerate
    (rel_rmse == 1 by construction) and is excluded from (B).

Outputs to ``<out>/bias_validation/bias_validation_analysis/``:
  - ``fold_metrics_merged.csv`` -- long-format merged fold metrics with
    columns [target, subset, region, fold_rmse, fold_mae, draw_idx, ...].
  - ``bias_validation_wilcoxon.csv`` -- one row per (contrast_type,
    target_or_subset, metric, contrast_label) with Wilcoxon W, two-sided
    and one-sided (same < other) p-values, and paired-fold medians.

Run::

  uv run python src/pipeline/connectivity_prediction/analyze_cross_prediction.py \\
      --config configs/abc_expanded_hpobest.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from clrc.core.io import find_repo_root, load_yaml_config
from clrc.core.logging import setup_logging

logger = logging.getLogger(__name__)

TARGETS = ("sc", "fc")
SUBSETS = ("sc_biased", "fc_biased", "balanced")


# ---------------------------------------------------------------------------
#  I/O
# ---------------------------------------------------------------------------


def _resolve_out_root(cfg: dict) -> Path:
    out_root = Path(cfg["output"]["base_dir"])
    if not out_root.is_absolute():
        out_root = find_repo_root() / out_root
    return out_root


def _locate_run_dir(bias_root: Path, target: str, subset: str) -> Path:
    """Return the unique timestamped run dir under ``bias_root/target/subset``.

    Raises if zero or multiple run directories are present.
    """
    parent = bias_root / target / subset
    if not parent.is_dir():
        raise FileNotFoundError(f"Expected bias_validation run parent: {parent}")
    runs = [p for p in parent.iterdir() if p.is_dir()]
    runs = [p for p in runs if (p / "fold_metrics.csv").is_file()]
    if len(runs) == 0:
        raise FileNotFoundError(
            f"No run dir with fold_metrics.csv under {parent}"
        )
    if len(runs) > 1:
        # Pick the newest by mtime and warn; mirror the robustness of the
        # cross-prediction driver which handles repeated relaunches.
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        logger.warning(
            "Multiple run dirs under %s; using newest: %s (others: %s)",
            parent, runs[0].name, [p.name for p in runs[1:]],
        )
    return runs[0]


def _load_merged_fold_metrics(bias_root: Path) -> pd.DataFrame:
    """Concatenate the six fold_metrics.csv files into a long-format frame."""
    frames = []
    for target in TARGETS:
        for subset in SUBSETS:
            run_dir = _locate_run_dir(bias_root, target, subset)
            csv_path = run_dir / "fold_metrics.csv"
            df = pd.read_csv(csv_path)
            df = df.assign(target=target, subset=subset, run_dir=str(run_dir))
            df = df.rename(columns={"holdout_region": "region"})
            frames.append(df)
            logger.info(
                "loaded target=%s subset=%s -> %d rows from %s",
                target, subset, len(df), run_dir.name,
            )
    merged = pd.concat(frames, ignore_index=True)
    return merged


# ---------------------------------------------------------------------------
#  Contrast A: within-target, across-subset
# ---------------------------------------------------------------------------


def _pair_by_region(
    df: pd.DataFrame, target: str, subset_a: str, subset_b: str, metric: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Align fold metrics between two subsets on the shared region set."""
    a = df.query("target == @target and subset == @subset_a")[
        ["region", metric]
    ].set_index("region")
    b = df.query("target == @target and subset == @subset_b")[
        ["region", metric]
    ].set_index("region")
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return np.array([]), np.array([]), []
    a_vals = a.loc[common, metric].to_numpy(dtype=np.float64)
    b_vals = b.loc[common, metric].to_numpy(dtype=np.float64)
    return a_vals, b_vals, common


def _wilcoxon_paired(
    a: np.ndarray, b: np.ndarray,
) -> Tuple[float, float, float, float, int]:
    """Paired-Wilcoxon summary: (W, p_two_sided, p_a_less_b, p_a_greater_b, n).

    Zero-difference ties are handled by scipy's ``zero_method='wilcox'``
    (default), which drops them; we report ``n`` as the number of non-
    zero-difference pairs. ``nan`` is returned for all p's when n < 3.
    """
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diffs = diffs[~np.isnan(diffs)]
    non_zero = diffs[diffs != 0.0]
    n_eff = int(non_zero.size)
    if n_eff < 3:
        return (float("nan"), float("nan"), float("nan"), float("nan"), n_eff)
    w_ts = stats.wilcoxon(non_zero, alternative="two-sided")
    w_less = stats.wilcoxon(non_zero, alternative="less")
    w_greater = stats.wilcoxon(non_zero, alternative="greater")
    return (
        float(w_ts.statistic),
        float(w_ts.pvalue),
        float(w_less.pvalue),
        float(w_greater.pvalue),
        n_eff,
    )


def contrast_within_target(df: pd.DataFrame) -> pd.DataFrame:
    """Within-target paired Wilcoxon across every ordered subset pair.

    Reports both RMSE and MAE. One row per (target, subset_A, subset_B,
    metric). ``p_less`` tests ``subset_A < subset_B``; ``p_greater`` tests
    the opposite one-sided direction.
    """
    rows = []
    subset_pairs = [
        (a, b) for i, a in enumerate(SUBSETS) for b in SUBSETS[i + 1 :]
    ]
    for target in TARGETS:
        for subset_a, subset_b in subset_pairs:
            for metric in ("fold_rmse", "fold_mae"):
                a_vals, b_vals, common = _pair_by_region(
                    df, target, subset_a, subset_b, metric,
                )
                W, p_ts, p_less, p_greater, n_eff = _wilcoxon_paired(a_vals, b_vals)
                rows.append({
                    "contrast_type": "within_target_across_subset",
                    "target": target,
                    "subset_A": subset_a,
                    "subset_B": subset_b,
                    "metric": metric,
                    "n_paired_regions": len(common),
                    "n_eff_wilcoxon": n_eff,
                    "median_A": float(np.median(a_vals)) if len(a_vals) else float("nan"),
                    "median_B": float(np.median(b_vals)) if len(b_vals) else float("nan"),
                    "median_diff_A_minus_B": (
                        float(np.median(a_vals - b_vals)) if len(a_vals) else float("nan")
                    ),
                    "wilcoxon_W": W,
                    "p_two_sided": p_ts,
                    "p_A_less_B": p_less,
                    "p_A_greater_B": p_greater,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Contrast B: cross-target within-subset, balanced-normalized
# ---------------------------------------------------------------------------


def _balanced_ref_map(
    df: pd.DataFrame, target: str, metric: str,
) -> Dict[str, float]:
    """Return ``{region: balanced-subset metric}`` for a target.

    Used as the per-region normalizer for the cross-target contrast so
    RMSEs on SC (scale ~1) and FC (scale ~0.1) are first put on a
    comparable "multiples-of-balanced" axis before pairing.
    """
    sub = df.query("target == @target and subset == 'balanced'")
    return dict(zip(sub["region"].to_list(), sub[metric].to_numpy(dtype=np.float64)))


def _relative_metric(
    df: pd.DataFrame, target: str, subset: str, metric: str,
    balanced_ref: Dict[str, float],
) -> pd.DataFrame:
    sub = df.query("target == @target and subset == @subset")[
        ["region", metric]
    ].copy()
    sub["balanced_ref"] = sub["region"].map(balanced_ref)
    sub["rel_metric"] = sub[metric] / sub["balanced_ref"]
    return sub[["region", "rel_metric"]].set_index("region")


def contrast_cross_target_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-target paired Wilcoxon on balanced-normalized fold metrics.

    Pairs per region the ratio ``rel_metric = (fold_metric / balanced
    subset's fold_metric on the same target, region)``. Compares SC vs FC
    within each non-balanced subset. The balanced subset is degenerate
    (rel == 1.0 by construction) and is skipped.
    """
    rows = []
    for subset in SUBSETS:
        if subset == "balanced":
            continue  # degenerate
        for metric in ("fold_rmse", "fold_mae"):
            ref_sc = _balanced_ref_map(df, "sc", metric)
            ref_fc = _balanced_ref_map(df, "fc", metric)
            rel_sc = _relative_metric(df, "sc", subset, metric, ref_sc)
            rel_fc = _relative_metric(df, "fc", subset, metric, ref_fc)
            common = sorted(set(rel_sc.index) & set(rel_fc.index))
            if not common:
                continue
            a = rel_sc.loc[common, "rel_metric"].to_numpy(dtype=np.float64)
            b = rel_fc.loc[common, "rel_metric"].to_numpy(dtype=np.float64)
            # Drop NaN/inf introduced by divide-by-zero balanced ref (shouldn't
            # happen in practice; guard anyway).
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            W, p_ts, p_less, p_greater, n_eff = _wilcoxon_paired(a, b)
            rows.append({
                "contrast_type": "cross_target_normalized",
                "subset": subset,
                "metric": metric,
                "n_paired_regions": int(mask.sum()),
                "n_eff_wilcoxon": n_eff,
                "median_rel_SC": float(np.median(a)) if len(a) else float("nan"),
                "median_rel_FC": float(np.median(b)) if len(b) else float("nan"),
                "median_diff_SC_minus_FC": (
                    float(np.median(a - b)) if len(a) else float("nan")
                ),
                "wilcoxon_W": W,
                "p_two_sided": p_ts,
                "p_SC_less_FC": p_less,
                "p_SC_greater_FC": p_greater,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--bias-validation-dir", type=Path, default=None,
        help=(
            "Override the bias_validation root (default "
            "<output.base_dir>/bias_validation)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_yaml_config(args.config)
    out_root = _resolve_out_root(cfg)
    bias_root = args.bias_validation_dir or (out_root / "bias_validation")

    analysis_dir = bias_root / "bias_validation_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("analyze_cross_prediction", output_dir=analysis_dir)
    logger.info("Config: %s", args.config)
    logger.info("bias_validation_dir: %s", bias_root)
    logger.info("analysis output dir: %s", analysis_dir)

    merged = _load_merged_fold_metrics(bias_root)
    merged_path = analysis_dir / "fold_metrics_merged.csv"
    merged.to_csv(merged_path, index=False)
    logger.info("wrote merged fold metrics -> %s (%d rows)", merged_path, len(merged))

    a_rows = contrast_within_target(merged)
    b_rows = contrast_cross_target_normalized(merged)

    # Concat into a single long-format summary with a common column set.
    # Harmonize column names: (A) uses target / subset_A / subset_B;
    # (B) uses subset. We keep them as separate columns and leave empties
    # for the contrast that does not apply.
    summary = pd.concat([a_rows, b_rows], ignore_index=True, sort=False)
    summary_path = analysis_dir / "bias_validation_wilcoxon.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("wrote Wilcoxon summary -> %s (%d rows)", summary_path, len(summary))

    print("\n" + "=" * 80)
    print("Contrast A: within-target, across-subset (paired per region)")
    print("=" * 80)
    a_show = a_rows.query("metric == 'fold_rmse'")[[
        "target", "subset_A", "subset_B", "n_eff_wilcoxon",
        "median_A", "median_B", "median_diff_A_minus_B",
        "wilcoxon_W", "p_two_sided", "p_A_less_B",
    ]]
    print(a_show.to_string(index=False))

    print("\n" + "=" * 80)
    print("Contrast B: cross-target, balanced-normalized (paired per region)")
    print("=" * 80)
    b_show = b_rows.query("metric == 'fold_rmse'")[[
        "subset", "n_eff_wilcoxon",
        "median_rel_SC", "median_rel_FC", "median_diff_SC_minus_FC",
        "wilcoxon_W", "p_two_sided",
    ]]
    print(b_show.to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
