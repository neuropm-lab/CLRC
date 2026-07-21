"""Spatial null model utilities for clrc.

Public API re-exports live here so callers can do::

    from clrc.spatial import generate_brainsmash_surrogates
"""

from clrc.spatial.nulls import (
    generate_brainsmash_surrogates,
    surrogate_celltype_abundance_maps,
    surrogate_expression_maps,
)
from clrc.spatial.atlas import (
    build_abc_distance_matrix,
    compute_abc_centroids,
    load_abc_allen_mapping,
    load_allen_centroids,
    load_allen_labels,
    load_allen_voxel_counts,
)
from clrc.spatial.connectivity_targets import (
    load_dsi_studio_matrices,
    load_mat_matrix,
    project_allen_to_abc_naive_and_voxel,
    project_multi_allen_to_abc,
    save_connectivity_target_h5,
    save_connectivity_target_pkl,
)
from clrc.spatial.nc_rebuild import (
    build_null_cci_features,
    build_real_pair_counts,
    reconstruct_pseudo_anndata,
    run_nc_on_pseudo,
)

__all__ = [
    "generate_brainsmash_surrogates",
    "surrogate_celltype_abundance_maps",
    "surrogate_expression_maps",
    "build_abc_distance_matrix",
    "compute_abc_centroids",
    "load_abc_allen_mapping",
    "load_allen_centroids",
    "load_allen_labels",
    "load_allen_voxel_counts",
    "load_dsi_studio_matrices",
    "load_mat_matrix",
    "project_allen_to_abc_naive_and_voxel",
    "project_multi_allen_to_abc",
    "save_connectivity_target_h5",
    "save_connectivity_target_pkl",
    "reconstruct_pseudo_anndata",
    "build_real_pair_counts",
    "run_nc_on_pseudo",
    "build_null_cci_features",
]
