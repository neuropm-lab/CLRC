#!/usr/bin/env python3
"""Driver — CLRC explanatory-value analysis.

Runs two complementary analyses on subject-level outcomes:

* **Variance partition** — Nimon, Lewis, Kane, & Haynes (2008) decomposition of
  cross-validated R^2 between bulk regional gradients (W) and CLRC fingerprints
  (D) for each clinical outcome, with a 1000x LR-shuffle permutation null.
* **Double Machine Learning** — Chernozhukov+ 2018 cross-fitted orthogonal
  score (DML2 form) with subject-level bootstrap 95% CI.

Both share the same subject-level inputs: the per-subject NeuronChat
communication tensor (D, dimensionality-reduced via PCA), the per-subject
pseudobulk regional expression (W, PCA), and 5 clinical outcomes from
``dataset_810_cross-sectional_05-27-2024.xlsx``.

Outputs (under ``--out-dir``, default
``out/rosmap_expanded/explanatory_value/``):

* ``variance_partition.csv`` — per-outcome R^2 + unique/shared decomposition
  + permutation p-value.
* ``dml_coefficients.csv`` — per-outcome × per-PC beta_hat + bootstrap CI.
* ``dml_bootstrap_distribution.npz`` — full bootstrap distribution arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from clrc.causal.explanatory_value import (
    CLINICAL_COVARIATES_DEFAULT,
    CLINICAL_OUTCOMES_DEFAULT,
    build_subject_feature_matrix,
    build_subject_pseudobulk,
    dml_orthogonal_score,
    load_clinical_outcomes,
    permutation_null_unique_clrc,
    variance_partition,
)

def _parse_outcomes(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return CLINICAL_OUTCOMES_DEFAULT
    return tuple(s.strip() for s in spec.split(",") if s.strip())


def _reduce_pca(matrix: pd.DataFrame, *, n_components: int, seed: int) -> np.ndarray:
    """Centre, drop all-NaN rows, fill within-column NaNs with column mean,
    then run randomized truncated PCA. Returns (n_subjects, k)."""
    X = matrix.to_numpy(dtype=np.float32, copy=True)
    # Drop fully-NaN columns first: these features have no information
    # anywhere in the cohort. Without this step, np.nanmean returns NaN for
    # such columns and downstream PCA rejects the input.
    finite_any = np.isfinite(X).any(axis=0)
    if not finite_any.all():
        X = X[:, finite_any]
    col_mean = np.nanmean(X, axis=0).astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        # In-place NaN fill avoids a second full-size temporary (matters at
        # ROSMAP scale where X can be ~1.6 GB).
        rows, cols = np.nonzero(nan_mask)
        X[rows, cols] = col_mean[cols]
    if X.shape[1] == 0:
        raise ValueError("All-NaN feature matrix after column drop")
    k = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(X)


def _align_subjects(
    feature_df: pd.DataFrame,
    bulk_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common = sorted(
        set(feature_df.index) & set(bulk_df.index) & set(clinical_df.index)
    )
    if not common:
        raise ValueError("No subjects shared across feature/bulk/clinical inputs")
    return (
        common,
        feature_df.loc[common],
        bulk_df.loc[common],
        clinical_df.loc[common],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLRC explanatory-value analysis"
    )
    parser.add_argument("--nc-dir", type=Path, required=True)
    parser.add_argument(
        "--h5-pattern",
        type=str,
        default="nc_subj_*_M50.h5",
        help="Glob for NC H5 files (production default = M50).",
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--clinical-xlsx", type=Path, required=True)
    parser.add_argument("--linkage-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--k-pcs", type=int, default=20)
    parser.add_argument(
        "--k-sweep",
        type=str,
        default=None,
        help="Comma-separated K values for the variance-partition sensitivity "
             "sweep (e.g. '5,10,15,20'). If set, variance_partition is run at "
             "each K and the per-K rows are stacked into variance_partition.csv. "
             "Pre-registered as a transparent robustness check.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated list; defaults to all five.",
    )
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    outcomes = _parse_outcomes(args.outcomes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feature_df = build_subject_feature_matrix(args.nc_dir, h5_pattern=args.h5_pattern)
    bulk_df = build_subject_pseudobulk(args.h5ad)
    clinical_df = load_clinical_outcomes(
        args.clinical_xlsx,
        args.linkage_csv,
        subject_ids=feature_df.index.tolist(),
        outcomes=outcomes,
        covariates=CLINICAL_COVARIATES_DEFAULT,
    )
    subjects, feature_df, bulk_df, clinical_df = _align_subjects(
        feature_df, bulk_df, clinical_df
    )

    if args.k_sweep:
        k_values = sorted({int(v) for v in args.k_sweep.split(",") if v.strip()})
    else:
        k_values = [args.k_pcs]
    k_dml = args.k_pcs  # DML always run at the primary k_pcs

    pca_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in sorted(set(k_values + [k_dml])):
        pca_cache[k] = (
            _reduce_pca(feature_df, n_components=k, seed=args.seed),
            _reduce_pca(bulk_df, n_components=k, seed=args.seed),
        )

    vp_records: list[dict] = []
    dml_records: list[dict] = []
    bootstrap_arrays: dict[str, np.ndarray] = {}

    for outcome in outcomes:
        Y = clinical_df[outcome].to_numpy(dtype=np.float64)
        valid = np.isfinite(Y)
        if valid.sum() < args.n_folds * 2:
            raise ValueError(
                f"Outcome {outcome!r} has only {valid.sum()} valid subjects "
                f"(need >= {args.n_folds * 2})"
            )
        Y_v = Y[valid]

        # Variance partition: sweep over K (each row = (outcome, k)).
        for k in k_values:
            D_full, W_full = pca_cache[k]
            D_v = D_full[valid]
            W_v = W_full[valid]
            vp = variance_partition(Y_v, D_v, W_v, n_splits=args.n_folds, seed=args.seed)
            perm = permutation_null_unique_clrc(
                Y_v, D_v, W_v,
                n_perm=args.n_perm, n_splits=args.n_folds, seed=args.seed,
            )
            vp_records.append({
                "outcome": outcome,
                "k_pcs": k,
                "n_subjects": int(valid.sum()),
                **{key: float(v) for key, v in vp.items()},
                "p_perm": float(perm["p_value"]),
            })

        # DML: at the primary k_pcs only.
        D_full, W_full = pca_cache[k_dml]
        D_v = D_full[valid]
        W_v = W_full[valid]
        dml = dml_orthogonal_score(
            Y_v, D_v, W_v,
            n_folds=args.n_folds, n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        for j in range(D_v.shape[1]):
            dml_records.append({
                "outcome": outcome,
                "pc_index": j,
                "beta_hat": float(dml["beta_hat"][j]),
                "ci_lower": float(dml["ci_lower"][j]),
                "ci_upper": float(dml["ci_upper"][j]),
                "bootstrap_p": float(
                    min(
                        (dml["bootstrap_dist"][:, j] <= 0).mean(),
                        (dml["bootstrap_dist"][:, j] >= 0).mean(),
                    ) * 2
                ),
            })
        bootstrap_arrays[outcome] = dml["bootstrap_dist"]

    vp_df = pd.DataFrame(vp_records)
    dml_df = pd.DataFrame(dml_records)
    vp_df.to_csv(args.out_dir / "variance_partition.csv", index=False)
    dml_df.to_csv(args.out_dir / "dml_coefficients.csv", index=False)
    np.savez_compressed(
        args.out_dir / "dml_bootstrap_distribution.npz",
        **bootstrap_arrays,
    )


if __name__ == "__main__":
    main()
