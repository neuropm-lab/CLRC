"""Explanatory-value analyses: variance partition + double machine learning.

ROSMAP per-subject NeuronChat H5 schema:
    /net          (n_lr, n_nodes, n_nodes)  float64  — primary communication tensor
    /net0         (n_lr, n_nodes, n_nodes)  float64  — null/permuted version
    /labels_lr    (n_lr,)                  bytes    — LR-pair names
    /labels_region_ct (n_nodes,)           bytes    — "<region>::<celltype>" labels
    /ligand_genes (n_lr,)                  bytes
    /receptor_genes (n_lr,)                bytes

Filenames follow `nc_subj_<subject_id>_M<n_perm>.h5`. Subject IDs are the
ROSMAP `R<digits>` form.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import KFold

DEFAULT_RIDGE_ALPHAS: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

_SUBJ_RE = re.compile(r"nc_subj_(R\d+)_M\d+\.h5$")


def build_subject_feature_matrix(
    nc_dir: Path,
    *,
    h5_pattern: str = "nc_subj_*_M50.h5",
) -> pd.DataFrame:
    """Stack per-subject NeuronChat communication tensors into a wide matrix.

    Subjects have non-uniform numbers of (region, celltype) nodes because
    not every subject has snRNA-seq data from every region or every cell
    type. This loader uses :func:`clrc.ad.h5_loader.load_subject_h5` plus
    :func:`clrc.ad.aggregation.aggregate_region_collapsed` to (a) read each
    subject's variable-shape tensor, (b) build a global union of cell-type
    and LR labels, and (c) aggregate every subject onto the same canonical
    ``(LR, sender_celltype, receiver_celltype)`` axis (region-collapsed).

    This collapses the regional dimension — the ROSMAP arm has only 6
    regions and the ``aggregate_region_collapsed`` helper is the production
    convention used by the rest of the AD pipeline (see
    ``src/pipeline/pathology_correlation/ad_aggregate.py``). For region-aware variants
    the upstream :func:`clrc.ad.aggregation.aggregate_region_specific` can
    be wired in as a follow-up.

    Parameters
    ----------
    nc_dir
        Directory of ``nc_subj_<id>_M<n>.h5`` files in Python NeuronChat
        format (root attrs ``interaction_names``, ``group_names``).
    h5_pattern
        Glob pattern; defaults to the production ``nc_subj_*_M50.h5``.

    Returns
    -------
    pd.DataFrame
        Index ``subject_id``; columns are ``"<LR> | <sender_ct> -> <receiver_ct>"``
        feature names (NaN where a (sender, receiver) pair has no data in
        any region for that subject).

    Raises
    ------
    FileNotFoundError
        If ``nc_dir`` contains no matching files.
    """
    from clrc.ad.aggregation import (
        aggregate_region_collapsed,
        build_label_index_maps,
        collect_global_labels,
    )
    from clrc.ad.h5_loader import load_subject_h5

    nc_dir = Path(nc_dir)
    paths: list[tuple[str, Path]] = []
    for path in sorted(nc_dir.glob(h5_pattern)):
        match = _SUBJ_RE.match(path.name)
        if match is None:
            continue
        paths.append((match.group(1), path))
    if not paths:
        raise FileNotFoundError(
            f"No NC H5 files matching {h5_pattern!r} under {nc_dir}"
        )

    h5_path_strs = [str(p) for _, p in paths]
    all_region_ct, lr_labels, _, unique_celltypes = collect_global_labels(h5_path_strs)
    unique_regions = sorted({lab.split("::")[0] for lab in all_region_ct})
    global_label_ct_idx, _ = build_label_index_maps(
        all_region_ct, unique_celltypes, unique_regions
    )

    subject_ids = [sid for sid, _ in paths]
    subject_data = {sid: load_subject_h5(p) for sid, p in paths}

    matrix, feature_names = aggregate_region_collapsed(
        subject_data,
        subject_ids,
        lr_labels,
        unique_celltypes,
        global_label_ct_idx,
    )
    df = pd.DataFrame(
        matrix.astype(np.float32, copy=False),
        index=pd.Index(subject_ids, name="subject_id"),
        columns=feature_names,
    )
    return df


def build_subject_pseudobulk(
    h5ad_path: Path,
    *,
    subject_col: str = "ROSMAP_IndividualID",
    region_col: str = "BrainRegion",
) -> pd.DataFrame:
    """Aggregate single-cell expression to (subject, region) means, reshape wide.

    Parameters
    ----------
    h5ad_path
        Path to an AnnData h5ad with ``subject_col`` and ``region_col`` in
        ``.obs``.
    subject_col, region_col
        Column names in ``adata.obs`` identifying the grouping factors.

    Returns
    -------
    pd.DataFrame
        Index ``subject_id``; columns ``"<region>::<gene>"``. Cells are mean
        expression of ``gene`` in cells from that ``subject`` × ``region``.
        Cells where a (subject, region) combination is absent are ``NaN``.
    """
    adata = anndata.read_h5ad(h5ad_path)
    if subject_col not in adata.obs.columns or region_col not in adata.obs.columns:
        raise KeyError(
            f"obs must contain {subject_col!r} and {region_col!r}; "
            f"got {list(adata.obs.columns)}"
        )
    obs = adata.obs[[subject_col, region_col]].copy()
    obs[subject_col] = obs[subject_col].astype(str)
    obs[region_col] = obs[region_col].astype(str)

    genes = list(adata.var_names)
    X = adata.X

    groups = obs.groupby([subject_col, region_col], observed=True).indices
    keys = list(groups.keys())
    mat = np.empty((len(keys), len(genes)), dtype=np.float32)
    for i, key in enumerate(keys):
        sub = X[groups[key], :]
        if sp.issparse(sub):
            mean = np.asarray(sub.mean(axis=0)).ravel()
        else:
            mean = np.asarray(sub).mean(axis=0)
        mat[i] = mean.astype(np.float32, copy=False)

    long = pd.DataFrame(
        mat,
        index=pd.MultiIndex.from_tuples(keys, names=[subject_col, region_col]),
        columns=genes,
    )
    wide = long.unstack(level=region_col)
    wide.columns = [f"{region}::{gene}" for gene, region in wide.columns]
    wide.index.name = "subject_id"
    return wide


CLINICAL_OUTCOMES_DEFAULT = (
    "cogng_demog_slope",
    "tangsqrt",
    "gpath",
    "amylsqrt",
    "cogn_global_lv",
)
CLINICAL_COVARIATES_DEFAULT = ("age_death", "educ", "msex")


def load_clinical_outcomes(
    xlsx_path: Path,
    csv_path: Path,
    subject_ids: list[str],
    *,
    outcomes: tuple[str, ...] = CLINICAL_OUTCOMES_DEFAULT,
    covariates: tuple[str, ...] = CLINICAL_COVARIATES_DEFAULT,
    individual_id_col: str = "individualID",
    projid_col: str = "projid",
) -> pd.DataFrame:
    """Load ROSMAP clinical outcomes + covariates aligned to ``subject_ids``.

    Parameters
    ----------
    xlsx_path
        ``dataset_810`` cross-sectional spreadsheet (must contain ``projid``
        and the requested outcome / covariate columns).
    csv_path
        ``ROSMAP_clinical.csv`` (must contain ``individualID`` ↔ ``projid``).
    subject_ids
        ``R<digits>`` IDs from per-subject NeuronChat filenames.
    outcomes, covariates
        Column names to extract from the xlsx.
    individual_id_col, projid_col
        Column names in the linkage CSV.

    Returns
    -------
    pd.DataFrame
        Indexed by ``subject_id`` (``R<digits>``); columns are
        ``outcomes + covariates``.

    Raises
    ------
    KeyError
        If any ``subject_id`` cannot be mapped through the linkage CSV.
    """
    link = pd.read_csv(csv_path, usecols=[individual_id_col, projid_col])
    link[individual_id_col] = link[individual_id_col].astype(str)
    link = link.dropna(subset=[projid_col]).drop_duplicates(
        subset=individual_id_col, keep="first"
    )
    link = link.astype({projid_col: int}).set_index(individual_id_col)

    missing = sorted(set(subject_ids) - set(link.index))
    if missing:
        raise KeyError(f"subject_ids missing from {csv_path}: {missing[:5]}...")

    projid_lookup = link.loc[subject_ids, projid_col]

    cols = [projid_col, *outcomes, *covariates]
    xlsx = pd.read_excel(xlsx_path, usecols=cols)
    xlsx = xlsx.dropna(subset=[projid_col])
    xlsx = xlsx.astype({projid_col: int})
    if xlsx[projid_col].duplicated().any():
        dup = xlsx[xlsx[projid_col].duplicated(keep=False)][projid_col].tolist()
        raise ValueError(
            f"Duplicate projid values in {xlsx_path}: {sorted(set(dup))[:5]}"
        )
    xlsx = xlsx.set_index(projid_col)

    out = xlsx.reindex(projid_lookup.values)
    out.index = pd.Index(subject_ids, name="subject_id")
    return out[list(outcomes) + list(covariates)]


def _cv_held_out_r2(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kf: KFold,
    ridge_alphas: Sequence[float],
) -> float:
    """Cross-validated held-out R^2 with inner-CV ridge alpha selection.

    Computed as 1 - SSR/SST on the concatenated held-out predictions across
    all KFold splits.
    """
    yhat = np.empty_like(Y, dtype=np.float64)
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=tuple(ridge_alphas))
        model.fit(X[train_idx], Y[train_idx])
        yhat[test_idx] = model.predict(X[test_idx])
    ss_res = float(np.sum((Y - yhat) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def variance_partition(
    Y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    *,
    n_splits: int = 5,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    seed: int = 20260501,
) -> dict[str, float]:
    """Nimon, Lewis, Kane, & Haynes (2008) commonality decomposition via ridge-CV.

    Returns the held-out R^2 of three nested ridge models — Y ~ D, Y ~ W,
    and Y ~ [D, W] — and decomposes the full R^2 into orthogonal pieces:

        unique_clrc = R^2_full - R^2_bulk
        unique_bulk = R^2_full - R^2_clrc
        shared      = R^2_clrc + R^2_bulk - R^2_full

    The three components sum to R^2_full by construction. ``shared`` may be
    negative under suppressor effects.

    Parameters
    ----------
    Y : (n,) array
        Outcome.
    D : (n, p_d) array
        Primary predictor block (e.g. CLRC PCs).
    W : (n, p_w) array
        Nuisance / control predictor block (e.g. bulk regional PCs).
    n_splits : int
        Outer KFold splits.
    ridge_alphas : sequence of float
        RidgeCV inner-CV grid.
    seed : int
        KFold shuffle seed (deterministic).

    Returns
    -------
    dict
        Keys ``r2_full``, ``r2_clrc``, ``r2_bulk``, ``unique_clrc``,
        ``unique_bulk``, ``shared``.

    References
    ----------
    Nimon, K., Lewis, M., Kane, R., & Haynes, R. M. (2008). An R package
    to compute commonality coefficients in the multiple regression case:
    an introduction to the package and a practical example.
    Behavior Research Methods, 40(2), 457-466.
    DOI: 10.3758/BRM.40.2.457.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if D.shape[0] != Y.shape[0] or W.shape[0] != Y.shape[0]:
        raise ValueError(
            f"row-count mismatch: Y={Y.shape[0]}, D={D.shape[0]}, W={W.shape[0]}"
        )
    if D.ndim != 2 or W.ndim != 2:
        raise ValueError("D and W must be 2D arrays")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_full = _cv_held_out_r2(np.hstack([D, W]), Y, kf=kf, ridge_alphas=ridge_alphas)
    r2_clrc = _cv_held_out_r2(D, Y, kf=kf, ridge_alphas=ridge_alphas)
    r2_bulk = _cv_held_out_r2(W, Y, kf=kf, ridge_alphas=ridge_alphas)
    return {
        "r2_full": r2_full,
        "r2_clrc": r2_clrc,
        "r2_bulk": r2_bulk,
        "unique_clrc": r2_full - r2_bulk,
        "unique_bulk": r2_full - r2_clrc,
        "shared": r2_clrc + r2_bulk - r2_full,
    }


def permutation_null_unique_clrc(
    Y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    *,
    n_perm: int = 1000,
    n_splits: int = 5,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    seed: int = 20260501,
) -> dict[str, np.ndarray | float]:
    """LR-shuffle null distribution for ``unique_clrc``.

    Each permutation reorders the **rows** of ``D`` against ``Y`` and ``W``
    (i.e. randomly re-assigns which subject's CLRC fingerprint pairs with
    which subject's outcome and bulk profile). This preserves the joint
    distribution of CLRC features (the within-subject LR covariance
    structure stays intact) while destroying any partial-R^2 signal beyond
    ``W``. It is the standard permutation null for testing whether
    ``unique_clrc`` exceeds the level expected when CLRC carries no
    incremental information about ``Y`` after controlling for ``W``.

    Note
    ----
    This is a row-permutation, not a within-row column-shuffle: shuffling
    columns would additionally destroy the within-subject LR covariance,
    which answers a different (and stricter) question than the
    partial-R^2 null wanted here.

    The empirical p-value uses the conservative finite-sample form
    ``(#{null >= obs} + 1) / (n_perm + 1)`` (Phipson & Smyth 2010).

    The KFold fold structure is held FIXED across observed and all
    permutations (same ``seed``), so split-noise cancels and the test
    isolates signal-vs-shuffled-row contrast.

    Returns
    -------
    dict
        ``observed`` (float), ``null`` (``n_perm``-array), ``p_value`` (float).
    """
    rng = np.random.default_rng(seed)
    observed = variance_partition(
        Y, D, W, n_splits=n_splits, ridge_alphas=ridge_alphas, seed=seed
    )["unique_clrc"]
    null = np.empty(n_perm, dtype=np.float64)
    n = D.shape[0]
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = variance_partition(
            Y,
            D[perm],
            W,
            n_splits=n_splits,
            ridge_alphas=ridge_alphas,
            seed=seed,
        )["unique_clrc"]
    p_value = float((null >= observed).sum() + 1) / float(n_perm + 1)
    return {"observed": float(observed), "null": null, "p_value": p_value}


def dml_orthogonal_score(
    Y: np.ndarray,
    D: np.ndarray,
    W: np.ndarray,
    *,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    ci_quantiles: tuple[float, float] = (0.025, 0.975),
    inner_cv_folds: int = 5,
    seed: int = 20260501,
) -> dict[str, np.ndarray]:
    """Double/Debiased Machine Learning estimator (Chernozhukov+ 2018, DML2).

    Implements the partially-linear-model orthogonal score via cross-fitting:

        Y = D' theta_0 + g_0(W) + U,    E[U | D, W] = 0
        D = m_0(W) + V,                  E[V | W] = 0

    For each cross-fitting fold k, nuisance functions g_k and m_k are fit on
    the held-IN folds via LassoCV (alpha by inner KFold(``inner_cv_folds``));
    residuals U_i, V_i are computed on the held-OUT fold. After all folds,
    residuals are pooled (DML2 form) and the orthogonal score solved as
    ``beta_hat = (V' V)^{-1} V' U``.

    Confidence intervals come from a subject-level non-parametric bootstrap
    of the residuals (does NOT refit nuisance per draw — see Chernozhukov+
    2018 §3.2 for valid bootstrap-on-residuals justification under
    Neyman-orthogonality).

    Parameters
    ----------
    Y : (n,) array
        Outcome.
    D : (n, p_d) array
        Treatment / primary feature matrix (CLRC PCs in our setting).
    W : (n, p_w) array
        Confounder / nuisance matrix (bulk regional PCs).
    n_folds
        Outer cross-fitting folds.
    n_bootstrap
        Bootstrap iterations for CI on theta.
    ci_quantiles
        Lower/upper quantile pair (default 95% CI).
    inner_cv_folds
        Inner CV folds used by ``LassoCV`` for alpha selection.
    seed
        Master seed; KFold and bootstrap rngs are deterministic.

    Returns
    -------
    dict
        ``beta_hat`` (p_d,), ``ci_lower`` (p_d,), ``ci_upper`` (p_d,),
        ``bootstrap_dist`` (n_bootstrap, p_d), ``residuals_Y`` (n,),
        ``residuals_D`` (n, p_d).

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. Econometrics Journal, 21, C1-C68.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    n = Y.shape[0]
    if D.shape[0] != n or W.shape[0] != n:
        raise ValueError(
            f"row-count mismatch: Y={Y.shape[0]}, D={D.shape[0]}, W={W.shape[0]}"
        )
    if D.ndim != 2 or W.ndim != 2:
        raise ValueError("D and W must be 2D arrays")
    p_d = D.shape[1]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res_Y = np.empty(n, dtype=np.float64)
    res_D = np.empty_like(D)

    for train_idx, test_idx in kf.split(Y):
        m_y = LassoCV(cv=inner_cv_folds, random_state=seed, max_iter=20000)
        m_y.fit(W[train_idx], Y[train_idx])
        res_Y[test_idx] = Y[test_idx] - m_y.predict(W[test_idx])

        for j in range(p_d):
            m_d = LassoCV(cv=inner_cv_folds, random_state=seed, max_iter=20000)
            m_d.fit(W[train_idx], D[train_idx, j])
            res_D[test_idx, j] = D[test_idx, j] - m_d.predict(W[test_idx])

    beta_hat, *_ = np.linalg.lstsq(res_D, res_Y, rcond=None)

    rng = np.random.default_rng(seed)
    bs = np.empty((n_bootstrap, p_d), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs[b], *_ = np.linalg.lstsq(res_D[idx], res_Y[idx], rcond=None)

    return {
        "beta_hat": beta_hat,
        "ci_lower": np.quantile(bs, ci_quantiles[0], axis=0),
        "ci_upper": np.quantile(bs, ci_quantiles[1], axis=0),
        "bootstrap_dist": bs,
        "residuals_Y": res_Y,
        "residuals_D": res_D,
    }
