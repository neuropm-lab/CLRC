"""NeuronChat object creation and merging."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse

import anndata

from neuronchat.core import InteractionEntry, NeuronChat
from neuronchat.database import load_db


def create_neuronchat(
    adata: anndata.AnnData,
    db: str | dict[str, InteractionEntry] = "mouse",
    group_by: str = "cell_type",
    layer: str | None = None,
    keep_data: bool = True,
) -> NeuronChat:
    """Create a NeuronChat object from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with normalized (not raw count) expression.
    db : str or dict
        'mouse', 'human', or a custom dict of InteractionEntry.
    group_by : str
        Column name in adata.obs for cell group labels. Unlike R's
        createNeuronChat (which defaults group.by=NULL → a single group
        "labels"), this parameter requires an explicit obs column because
        AnnData metadata is always column-based and a single-group default
        is not useful for cell-cell communication inference.
    layer : str | None
        Layer to use. None uses adata.X.
    keep_data : bool
        Whether to store the full expression matrix. Only needed for
        merge_neuronchat with merge_data=True. Set to False to save
        memory on large datasets.

    Returns
    -------
    NeuronChat object ready for run_neuronchat().
    """
    # Load interaction DB
    if isinstance(db, str):
        interaction_db = load_db(db)
    else:
        interaction_db = dict(db)

    # De-duplicate and sort by name
    interaction_db = {
        k: interaction_db[k] for k in sorted(interaction_db.keys())
    }

    # Collect all signaling genes
    signaling_genes: set[str] = set()
    for entry in interaction_db.values():
        signaling_genes.update(entry.lig_contributor)
        signaling_genes.update(entry.receptor_subunit)

    # Filter to genes present in data
    available_genes = set(adata.var_names)
    signaling_genes_sorted = sorted(signaling_genes & available_genes)

    if len(signaling_genes_sorted) == 0:
        raise ValueError(
            "No signaling genes from the interaction database overlap with "
            "the data's var_names. Check that gene names match the database "
            f"(database has {len(signaling_genes)} genes, data has "
            f"{len(available_genes)} genes, 0 in common)."
        )

    # Extract expression matrix
    if layer is not None:
        data_matrix = adata.layers[layer]
    else:
        data_matrix = adata.X

    # Get gene indices (O(1) lookup via dict instead of O(n) list.index)
    var_name_to_idx = {name: i for i, name in enumerate(adata.var_names)}
    gene_idx = [var_name_to_idx[g] for g in signaling_genes_sorted]

    if scipy.sparse.issparse(data_matrix):
        sig_values = data_matrix[:, gene_idx].toarray()
    else:
        sig_values = np.array(data_matrix[:, gene_idx], dtype=np.float64)

    sig_values = sig_values.astype(np.float64)

    # Normalize to [0, 1].
    # R (line 94) always divides: data.signaling/max(data.signaling), which
    # produces NaN for all-zero input (0/0). We intentionally guard against
    # this: zeros remain zeros, giving correct zero probability matrices
    # downstream rather than NaN propagation.
    max_val = sig_values.max()
    if max_val > 0:
        sig_values = sig_values / max_val

    # Build data_signaling DataFrame (cells x genes)
    data_signaling = pd.DataFrame(
        sig_values,
        columns=signaling_genes_sorted,
        index=adata.obs_names,
    )
    data_signaling["cell_subclass"] = adata.obs[group_by].values

    # Group labels
    group_labels = adata.obs[group_by].values
    idents = pd.Categorical(group_labels)

    # Store full data matrix (only if requested)
    if keep_data:
        if scipy.sparse.issparse(data_matrix):
            data = data_matrix.copy()
        else:
            data = np.array(data_matrix, dtype=np.float64)
    else:
        data = None

    return NeuronChat(
        data=data,
        data_signaling=data_signaling,
        meta=adata.obs.copy(),
        idents=idents,
        db=interaction_db,
        lr=list(interaction_db.keys()),
        var_names=list(adata.var_names),
        mode="single",
    )


def _build_cell_names(objects, names, cell_prefix):
    """Build per-object cell name lists, optionally prefixed.

    Does NOT mutate input objects. Returns a list of cell name arrays,
    one per object.
    """
    cell_names_per_obj = []
    for obj, name in zip(objects, names):
        original = obj.data_signaling.index.tolist()
        if cell_prefix:
            cell_names_per_obj.append([f"{c}_{name}" for c in original])
        else:
            cell_names_per_obj.append(original)

    if not cell_prefix:
        all_cells = [c for names_list in cell_names_per_obj for c in names_list]
        if len(all_cells) != len(set(all_cells)):
            raise ValueError(
                "Duplicate cell names detected across datasets. "
                "Set cell_prefix=True to avoid this."
            )

    return cell_names_per_obj


def _merge_signaling_union(objects, cell_names_per_obj):
    """Merge signaling data using UNION of genes, re-derived from raw data.

    Matches R's mergeNeuronChat behavior:
    1. Intersect genes across objects' full data matrices (var_names)
    2. Union signaling gene names across objects' data_signaling
    3. Re-derive signaling data from the merged raw data (no re-normalization)

    Returns (data_sig_joint, data_joint_full, var_names_joint).
    """
    # Intersect full gene sets across objects (R lines 191-194)
    genes_use = set(objects[0].var_names)
    for obj in objects[1:]:
        genes_use &= set(obj.var_names)
    genes_use_sorted = sorted(genes_use)

    # Build merged raw data matrix: cells x genes_use (R lines 195-198)
    var_name_to_idx_per_obj = [
        {name: i for i, name in enumerate(obj.var_names)}
        for obj in objects
    ]
    data_blocks = []
    for obj, idx_map in zip(objects, var_name_to_idx_per_obj):
        col_idx = [idx_map[g] for g in genes_use_sorted]
        if scipy.sparse.issparse(obj.data):
            block = obj.data[:, col_idx].toarray()
        else:
            block = obj.data[:, col_idx]
        data_blocks.append(np.asarray(block, dtype=np.float64))
    data_joint = np.vstack(data_blocks)

    # Union of signaling gene names (R line 199)
    signaling_genes_union: set[str] = set()
    for obj in objects:
        sig_cols = set(obj.data_signaling.columns) - {"cell_subclass"}
        signaling_genes_union |= sig_cols

    # Subset to genes present in the merged data (R line 200)
    genes_use_set = set(genes_use_sorted)
    sig_genes_sorted = sorted(signaling_genes_union & genes_use_set)

    # Re-derive signaling data from raw merged data (R line 200, no
    # re-normalization — matches R's mergeNeuronChat which uses raw values)
    gene_to_col = {g: i for i, g in enumerate(genes_use_sorted)}
    sig_col_idx = [gene_to_col[g] for g in sig_genes_sorted]
    sig_values = data_joint[:, sig_col_idx]

    # Build cell names and cell_subclass
    all_cell_names = [c for names_list in cell_names_per_obj for c in names_list]
    cell_subclass = []
    for obj in objects:
        cell_subclass.extend(obj.data_signaling["cell_subclass"].values.tolist())

    data_sig_joint = pd.DataFrame(
        sig_values, columns=sig_genes_sorted, index=all_cell_names,
    )
    data_sig_joint["cell_subclass"] = cell_subclass

    return data_sig_joint, data_joint, genes_use_sorted


def _merge_signaling_intersection(objects, cell_names_per_obj):
    """Merge signaling data using INTERSECTION of gene columns.

    Fallback when objects lack raw data matrices (keep_data=False).
    Uses pre-normalized data_signaling values directly.
    """
    import warnings

    gene_cols_sets = [
        set(obj.data_signaling.columns) - {"cell_subclass"}
        for obj in objects
    ]
    union_genes = gene_cols_sets[0].union(*gene_cols_sets[1:])
    common_genes = gene_cols_sets[0].intersection(*gene_cols_sets[1:])

    if common_genes != union_genes:
        warnings.warn(
            f"Objects have different signaling gene sets "
            f"({len(union_genes)} union, {len(common_genes)} intersection). "
            "Using intersection because some objects lack raw data matrices "
            "(keep_data=False). To use R-faithful union behavior, recreate "
            "objects with keep_data=True.",
            stacklevel=3,
        )

    common_gene_cols = sorted(common_genes)

    # Rebuild DataFrames with prefixed cell names
    frames = []
    for obj, cell_names in zip(objects, cell_names_per_obj):
        df = obj.data_signaling[common_gene_cols + ["cell_subclass"]].copy()
        df.index = cell_names
        frames.append(df)

    return pd.concat(frames, axis=0)


def merge_neuronchat(
    objects: list[NeuronChat],
    names: list[str] | None = None,
    merge_data: bool = False,
    cell_prefix: bool = False,
) -> NeuronChat:
    """Merge multiple NeuronChat objects.

    Parameters
    ----------
    objects : list[NeuronChat]
        List of NeuronChat objects (must have run_neuronchat completed).
    names : list[str] | None
        Dataset names. Defaults to Dataset_1, Dataset_2, etc.
    merge_data : bool
        Whether to store the merged full data matrix on the result.
    cell_prefix : bool
        Whether to prefix cell names with dataset name to avoid collisions.

    Returns
    -------
    Merged NeuronChat object with mode='merged'.
    """
    if names is None:
        names = [f"Dataset_{i + 1}" for i in range(len(objects))]

    if len(names) != len(objects):
        raise ValueError("Length of names must match length of objects")

    # Build cell names without mutating input objects
    cell_names_per_obj = _build_cell_names(objects, names, cell_prefix)

    # Merge net and idents
    net_merged = {name: obj.net for name, obj in zip(names, objects)}
    idents_merged = {name: obj.idents for name, obj in zip(names, objects)}

    # Merge metadata
    meta_cols = set(objects[0].meta.columns)
    for obj in objects[1:]:
        meta_cols &= set(obj.meta.columns)

    meta_joint = pd.concat(
        [obj.meta[list(meta_cols)] for obj in objects],
        axis=0,
    )

    # Align meta index with (possibly prefixed) cell names.
    # R lines 181-185: reassigns rownames(meta.joint) <- cell.names when they
    # differ (which always happens with cell.prefix=TRUE).
    all_cell_names = [c for names_list in cell_names_per_obj for c in names_list]
    meta_joint.index = all_cell_names

    dataset_labels = []
    for obj, name in zip(objects, names):
        dataset_labels.extend([name] * len(obj.meta))
    meta_joint["datasets"] = pd.Categorical(dataset_labels, categories=names)

    # Merge signaling data: union (R-faithful) or intersection (fallback)
    has_raw_data = all(
        obj.data is not None and obj.var_names is not None
        for obj in objects
    )

    if has_raw_data:
        data_sig_joint, data_joint_raw, var_names_joint = _merge_signaling_union(
            objects, cell_names_per_obj,
        )
    else:
        data_sig_joint = _merge_signaling_intersection(
            objects, cell_names_per_obj,
        )
        data_joint_raw = None
        var_names_joint = None

    # Joint idents
    idents_list = []
    idents_levels = []
    for obj in objects:
        idents_list.extend(obj.idents.tolist())
        for level in obj.idents.categories:
            if level not in idents_levels:
                idents_levels.append(level)
    idents_joint = pd.Categorical(idents_list, categories=idents_levels)
    idents_merged["joint"] = idents_joint

    # Data matrix for the merged object
    if merge_data:
        if data_joint_raw is None:
            missing = [i for i, obj in enumerate(objects) if obj.data is None]
            raise ValueError(
                f"merge_data=True but objects at indices {missing} have no "
                "data matrix (created with keep_data=False). Recreate them "
                "with keep_data=True to use merge_data=True."
            )
        data_out = data_joint_raw
    else:
        data_out = None

    return NeuronChat(
        data=data_out,
        data_signaling=data_sig_joint,
        meta=meta_joint,
        idents=idents_merged,
        db=objects[0].db,
        lr=objects[0].lr,
        var_names=var_names_joint,
        net=net_merged,
        mode="merged",
    )
