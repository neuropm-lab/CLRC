"""NeuronChat connectome construction from an annotated expression matrix.

Two scopes are supported and both run the identical NeuronChat procedure:

* ``dataset`` — one run over every cell in the matrix, producing a single
  connectome H5 (the ABC whole-brain connectome).
* ``subject`` — one run per subject, producing one H5 per subject (the
  ROSMAP per-individual connectomes).

The permutation test is never short-circuited or approximated; both scopes
call :func:`neuronchat.run_neuronchat` with the configured ``M``.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path

import anndata
import numpy as np
import scanpy as sc
from scipy import sparse

from neuronchat import create_neuronchat, run_neuronchat, save_h5

logger = logging.getLogger("clrc.preprocessing.connectome")

# Below this many cells or cell groups a permutation test cannot produce a
# meaningful null, so the subject is skipped rather than written out.
MIN_CELLS = 10
MIN_GROUPS = 2


def looks_like_raw_counts(adata: anndata.AnnData) -> bool:
    """Report whether ``adata.X`` appears to hold raw integer counts.

    Samples the leading 100x100 block rather than the full matrix, which for
    whole-brain inputs would mean materialising tens of GB.

    Parameters
    ----------
    adata : anndata.AnnData
        Matrix to inspect.

    Returns
    -------
    bool
        True when the sampled block is integer-valued and exceeds the range
        expected of log-normalised data.
    """
    block = adata.X[:100, :100]
    sample = block.toarray() if sparse.issparse(block) else np.asarray(block)
    max_val = float(np.max(sample))
    integer_like = bool(np.allclose(sample, np.round(sample), atol=1e-6))
    logger.info(
        "expression check: max=%.2f integer-like=%s", max_val, integer_like
    )
    return integer_like and max_val > 20


def log_normalize(adata: anndata.AnnData, *, target_sum: float = 1e4) -> None:
    """Library-size normalise to ``target_sum`` then log1p, in place."""
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)


def normalize_if_needed(adata: anndata.AnnData, mode: str = "auto") -> None:
    """Apply log-normalisation according to ``mode``.

    Parameters
    ----------
    adata : anndata.AnnData
        Matrix to normalise in place.
    mode : {'auto', 'always', 'never'}
        ``auto`` normalises only when :func:`looks_like_raw_counts` is True,
        which leaves already-normalised inputs (such as the ABC atlas
        matrices) untouched.
    """
    if mode == "never":
        return
    if mode == "always" or (mode == "auto" and looks_like_raw_counts(adata)):
        logger.info("log-normalizing expression matrix")
        log_normalize(adata)
    else:
        logger.info("expression matrix already normalized; skipping")


def subset_to_db_genes(
    adata: anndata.AnnData, db: dict
) -> anndata.AnnData:
    """Restrict ``adata`` to genes referenced by the interaction database.

    Parameters
    ----------
    adata : anndata.AnnData
        Expression matrix.
    db : dict
        Interaction database from :func:`neuronchat.load_db`.

    Returns
    -------
    anndata.AnnData
        Copy restricted to the ligand and receptor genes present in both the
        database and the matrix.
    """
    signaling: set[str] = set()
    for entry in db.values():
        signaling.update(entry.lig_contributor)
        signaling.update(entry.receptor_subunit)

    available = sorted(signaling & set(adata.var_names))
    logger.info(
        "subsetting to %d signaling genes (of %d in database)",
        len(available),
        len(signaling),
    )
    return adata[:, available].copy()


def _write_atomic(obj, out_h5: Path) -> None:
    """Save to a sibling temp file then rename, so a crash cannot leave a
    partially written H5 that later runs would mistake for a finished one."""
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=out_h5.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        save_h5(obj, tmp_path)
        tmp_path.replace(out_h5)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def run_connectome(
    adata: anndata.AnnData,
    db: dict,
    out_h5: Path,
    *,
    group_by: str,
    M: int,
    fdr: float = 0.05,
    seed: int | None = 42,
    device: str | list[str] = "cpu",
    n_jobs: int = 1,
    layer: str | None = None,
    progress: bool = True,
) -> Path:
    """Run NeuronChat over every cell in ``adata`` and save one H5.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalised, gene-scoped expression matrix carrying ``group_by`` in
        ``.obs``.
    db : dict
        Interaction database from :func:`neuronchat.load_db`.
    out_h5 : Path
        Destination H5.
    group_by : str
        ``.obs`` column holding the region-by-cell-type node labels.
    M : int
        Permutation count for the NeuronChat null.
    fdr : float
        FDR cutoff for Benjamini-Hochberg correction.
    seed : int | None
        Base seed; NeuronChat expands it into a deterministic per-interaction
        seed so results reproduce under parallel execution.
    device, n_jobs, layer, progress
        Forwarded to NeuronChat.

    Returns
    -------
    Path
        ``out_h5``.
    """
    if group_by not in adata.obs.columns:
        raise KeyError(
            f"group_by column {group_by!r} not in adata.obs "
            f"(columns: {list(adata.obs.columns)})"
        )

    obj = create_neuronchat(
        adata, db=db, group_by=group_by, layer=layer, keep_data=False
    )
    logger.info(
        "created NeuronChat: %d interactions, %d groups",
        len(obj.lr),
        adata.obs[group_by].nunique(),
    )
    obj = run_neuronchat(
        obj,
        M=M,
        fdr=fdr,
        seed=seed,
        device=device,
        n_jobs=n_jobs,
        progress=progress,
    )
    _write_atomic(obj, out_h5)
    logger.info("wrote connectome -> %s", out_h5)
    return out_h5


def run_connectome_by_subject(
    adata: anndata.AnnData,
    db: dict,
    out_dir: Path,
    *,
    subject_col: str,
    group_by: str,
    M: int,
    fdr: float = 0.05,
    seed: int | None = 42,
    device: str | list[str] = "cpu",
    n_jobs: int = 1,
    layer: str | None = None,
    subjects: list[str] | None = None,
    progress: bool = True,
) -> list[Path]:
    """Run NeuronChat separately for each subject, one H5 per subject.

    Subjects whose output already exists are skipped, so an interrupted run
    can be restarted without redoing completed subjects.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalised, gene-scoped matrix carrying ``subject_col`` and
        ``group_by`` in ``.obs``.
    db : dict
        Interaction database.
    out_dir : Path
        Directory receiving ``nc_subj_{subject}_M{M}.h5``.
    subject_col : str
        ``.obs`` column identifying the subject.
    group_by, M, fdr, seed, device, n_jobs, layer, progress
        As for :func:`run_connectome`.
    subjects : list[str] | None
        Restrict to these subject IDs. None processes all.

    Returns
    -------
    list[Path]
        Paths written during this call, excluding skipped subjects.
    """
    for col in (subject_col, group_by):
        if col not in adata.obs.columns:
            raise KeyError(f"column {col!r} not in adata.obs")

    out_dir.mkdir(parents=True, exist_ok=True)

    n_missing = int(adata.obs[subject_col].isna().sum())
    if n_missing:
        logger.warning(
            "%d cells have no %s and are excluded", n_missing, subject_col
        )
    all_subjects = sorted(adata.obs[subject_col].dropna().unique())
    if subjects is not None:
        wanted = set(subjects)
        all_subjects = [s for s in all_subjects if s in wanted]

    done = {p.stem.removeprefix("nc_subj_").removesuffix(f"_M{M}")
            for p in out_dir.glob(f"nc_subj_*_M{M}.h5")}
    pending = [s for s in all_subjects if s not in done]
    logger.info(
        "%d subjects total, %d already complete, %d to run",
        len(all_subjects),
        len(done),
        len(pending),
    )

    written: list[Path] = []
    for i, subject in enumerate(pending, start=1):
        sub = adata[adata.obs[subject_col] == subject].copy()
        n_cells = sub.shape[0]
        n_groups = int(sub.obs[group_by].nunique())
        logger.info(
            "[%d/%d] subject %s: %d cells, %d groups",
            i,
            len(pending),
            subject,
            n_cells,
            n_groups,
        )
        if n_cells < MIN_CELLS or n_groups < MIN_GROUPS:
            logger.warning("  skipping %s: below minimum cells/groups", subject)
            del sub
            continue

        try:
            written.append(
                run_connectome(
                    sub,
                    db,
                    out_dir / f"nc_subj_{subject}_M{M}.h5",
                    group_by=group_by,
                    M=M,
                    fdr=fdr,
                    seed=seed,
                    device=device,
                    n_jobs=n_jobs,
                    layer=layer,
                    progress=progress,
                )
            )
        except Exception:
            logger.exception("  subject %s failed; continuing", subject)
        finally:
            del sub
            gc.collect()

    return written
