"""brainSMASH-based spatial null map generators (node-level surrogacy).

Each node-level input map (per cell-type abundance or per-gene
expression) is surrogated independently via brainSMASH, preserving
single-map spatial autocorrelation while destroying joint structure
across maps (e.g. ligand-receptor gene pairing).

All functions are thin wrappers over ``brainsmash.mapgen.base.Base`` --
we do not extend or modify the brainSMASH algorithm. ``Base`` (not
``Sampled``) is appropriate for ABC-space atlases at N ~ 100 regions,
far below the N ~ 1000 threshold where ``Sampled`` becomes necessary.

References
----------
Burt, J. B., Helmer, M., Shinn, M., Anticevic, A., & Murray, J. D.
(2020). Generative modeling of brain maps with spatial autocorrelation.
NeuroImage, 220, 117038.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance-matrix validation (brainSMASH's check_distmat contract)
# ---------------------------------------------------------------------------

def _validate_distance_matrix(
    distance_matrix: np.ndarray,
    *,
    n_regions: int,
    atol: float = 1e-8,
) -> None:
    """Raise ValueError if ``distance_matrix`` is not a valid brainSMASH
    input: (n, n) shape, symmetric, non-negative, zero-diagonal, finite.

    Matches the contract enforced by
    ``brainsmash.mapgen.memmap.check_distmat`` but catches errors at the
    Python level with clearer messages before the memmap-backed brainSMASH
    code path has a chance to write files.
    """
    D = np.asarray(distance_matrix)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"distance_matrix must be square 2D; got shape {D.shape}."
        )
    if D.shape[0] != n_regions:
        raise ValueError(
            f"distance_matrix shape {D.shape} does not match map length "
            f"n_regions={n_regions}."
        )
    if not np.all(np.isfinite(D)):
        raise ValueError("distance_matrix contains non-finite values.")
    if not np.allclose(D, D.T, atol=atol):
        raise ValueError("distance_matrix must be symmetric.")
    diag = np.diag(D)
    if not np.allclose(diag, 0.0, atol=atol):
        raise ValueError(
            "distance_matrix must have zero diagonal; found max |diag| = "
            f"{np.max(np.abs(diag)):.3e}."
        )
    if (D < -atol).any():
        raise ValueError("distance_matrix must be non-negative.")


# ---------------------------------------------------------------------------
# Primary single-map wrapper
# ---------------------------------------------------------------------------

def generate_brainsmash_surrogates(
    map_values: np.ndarray,
    distance_matrix: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 0,
    *,
    deltas: Optional[np.ndarray] = None,
    kernel: str = "exp",
    pv: int = 25,
    nh: int = 25,
    batch_size: int = 100,
    n_jobs: int = 1,
    resample: bool = False,
) -> np.ndarray:
    """Generate variogram-matched spatial surrogates for one scalar map.

    Thin wrapper over :class:`brainsmash.mapgen.base.Base`. Parameters
    ``deltas``, ``kernel``, ``pv``, ``nh`` are passed through unchanged
    and match the brainSMASH defaults recommended in Burt et al. 2020.
    This function does not alter the brainSMASH algorithm -- it only
    validates inputs, invokes ``Base``, and logs wall time.

    Parameters
    ----------
    map_values : (n_regions,) array
        Node-level scalar map (e.g. a cell-type abundance vector across
        regions, or a gene-expression vector across regions). Must be 1D.
    distance_matrix : (n_regions, n_regions) array
        Region-region distance matrix. Must be symmetric with zero
        diagonal and non-negative entries. Euclidean centroid distance is
        the recommended choice for volumetric atlases (per brainSMASH
        docs and Burt et al. 2020); fibre-length / connectome-derived
        matrices with structural zeros are NOT appropriate.
    n_surrogates : int, default 1000
        Number of surrogate maps to draw.
    seed : int, default 0
        Base seed passed to brainSMASH for reproducibility.
    deltas : (n_deltas,) array, optional
        Grid of smoothing bandwidths. Defaults to
        ``np.linspace(0.1, 0.9, 9)`` matching brainSMASH's own default.
    kernel, pv, nh : brainSMASH ``Base`` kwargs
        Passed through unchanged. See brainsmash docs.
    batch_size : int, default 100
        Size of surrogate draw batches (controls peak memory inside
        brainSMASH). 100 is a safe default for N ~= 100 regions.
    n_jobs : int, default 1
        Number of parallel workers for brainSMASH's internal per-delta
        fitting. Scales near-linearly up to ~#CPU cores.

    Returns
    -------
    surrogates : (n_surrogates, n_regions) array
        Each row is one surrogate map. NOT guaranteed to be on the same
        scale as ``map_values`` -- brainSMASH's default does not resample
        back to the empirical distribution (``resample=False``).
    """
    from brainsmash.mapgen.base import Base  # local import, optional dep

    x = np.asarray(map_values, dtype=np.float64).ravel()
    if x.ndim != 1:
        raise ValueError(f"map_values must be 1D; got shape {map_values.shape}.")
    _validate_distance_matrix(distance_matrix, n_regions=x.shape[0])

    if deltas is None:
        deltas = np.linspace(0.1, 0.9, 9)

    t0 = time.perf_counter()
    gen = Base(
        x=x,
        D=np.asarray(distance_matrix, dtype=np.float64),
        deltas=deltas,
        kernel=kernel,
        pv=pv,
        nh=nh,
        seed=seed,
        n_jobs=n_jobs,
        resample=bool(resample),
    )
    surrogates = gen(n=int(n_surrogates), batch_size=int(batch_size))
    elapsed = time.perf_counter() - t0
    logger.info(
        "generate_brainsmash_surrogates: n_regions=%d, n_surrogates=%d, "
        "seed=%d, elapsed=%.2fs",
        x.shape[0], n_surrogates, seed, elapsed,
    )

    surrogates = np.asarray(surrogates)
    # brainSMASH squeezes the leading axis when n_surrogates=1 in some
    # versions; restore the 2D expected shape so downstream code can
    # always index by surrogate.
    if surrogates.ndim == 1 and int(n_surrogates) == 1:
        surrogates = surrogates[np.newaxis, :]
    expected_shape = (int(n_surrogates), x.shape[0])
    if surrogates.shape != expected_shape:
        raise RuntimeError(
            f"brainSMASH returned shape {surrogates.shape}, expected "
            f"{expected_shape}. Check brainsmash version / API drift."
        )
    return surrogates


# ---------------------------------------------------------------------------
# Joint multi-map helpers (abundance, expression)
# ---------------------------------------------------------------------------

def _surrogate_matrix_columns(
    matrix: np.ndarray,
    distance_matrix: np.ndarray,
    n_surrogates: int,
    seed: int,
    *,
    what: str,
    **base_kwargs,
) -> np.ndarray:
    """Surrogate each column of ``matrix`` independently, returning
    a (n_surrogates, n_regions, n_columns) array.

    Per-column independence is enforced by deriving a deterministic
    per-column seed from ``seed`` via simple offsetting -- sufficient for
    reproducibility, and brainSMASH's own randomness is driven by that
    per-column seed so the draws differ across columns even for
    identical input maps.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(
            f"{what} must be a 2D (n_regions, n_columns) array; got shape {M.shape}."
        )
    n_regions, n_cols = M.shape
    _validate_distance_matrix(distance_matrix, n_regions=n_regions)

    out = np.empty((int(n_surrogates), n_regions, n_cols), dtype=np.float64)
    t0 = time.perf_counter()
    for j in range(n_cols):
        col_seed = int(seed) + j
        surr = generate_brainsmash_surrogates(
            map_values=M[:, j],
            distance_matrix=distance_matrix,
            n_surrogates=int(n_surrogates),
            seed=col_seed,
            **base_kwargs,
        )
        out[:, :, j] = surr
    elapsed = time.perf_counter() - t0
    logger.info(
        "_surrogate_matrix_columns[%s]: n_columns=%d, n_surrogates=%d, "
        "total_elapsed=%.2fs (%.3fs/map)",
        what, n_cols, n_surrogates, elapsed, elapsed / max(n_cols, 1),
    )
    return out


def surrogate_celltype_abundance_maps(
    abundance: np.ndarray,
    distance_matrix: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 0,
    **base_kwargs,
) -> np.ndarray:
    """Generate per-cell-type surrogate abundance maps.

    Each cell-type column of ``abundance`` is surrogated independently
    via brainSMASH, preserving that cell-type's marginal spatial
    autocorrelation while breaking joint spatial structure across
    cell-types. This is the key step for the cell-type-abundance spatial
    null: after rebuilding CCI features with these surrogate abundances,
    joint ligand x receptor cell-type spatial patterning in the null is
    driven by independent brainSMASH draws rather than real biology.

    Parameters
    ----------
    abundance : (n_regions, n_celltypes) array
        Real cell-type abundance (or analogous scalar) per region per
        cell-type.
    distance_matrix : (n_regions, n_regions) array
        Region-region Euclidean (or otherwise metric) distance matrix.
    n_surrogates : int
        Number of surrogate draws per cell-type.
    seed : int
        Base seed; per-column seeds are ``seed + col_index``.
    base_kwargs
        Forwarded to :func:`generate_brainsmash_surrogates`.

    Returns
    -------
    surrogates : (n_surrogates, n_regions, n_celltypes) array
    """
    return _surrogate_matrix_columns(
        matrix=abundance,
        distance_matrix=distance_matrix,
        n_surrogates=n_surrogates,
        seed=seed,
        what="celltype_abundance",
        **base_kwargs,
    )


def surrogate_expression_maps(
    expression: np.ndarray,
    distance_matrix: np.ndarray,
    n_surrogates: int = 1000,
    seed: int = 0,
    **base_kwargs,
) -> np.ndarray:
    """Generate per-gene surrogate expression maps.

    Same contract as :func:`surrogate_celltype_abundance_maps` but for
    (region, gene) inputs. Independent surrogacy across genes is
    specifically what breaks ligand-receptor pair co-variation in the
    null -- biological specificity is thereby dissociated from the
    underlying spatial smoothness.

    Parameters
    ----------
    expression : (n_regions, n_genes) array
    distance_matrix : (n_regions, n_regions) array
    n_surrogates : int
    seed : int
    base_kwargs
        Forwarded to :func:`generate_brainsmash_surrogates`.

    Returns
    -------
    surrogates : (n_surrogates, n_regions, n_genes) array
    """
    return _surrogate_matrix_columns(
        matrix=expression,
        distance_matrix=distance_matrix,
        n_surrogates=n_surrogates,
        seed=seed,
        what="expression",
        **base_kwargs,
    )
