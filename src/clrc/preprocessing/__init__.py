"""clrc.preprocessing — expression-matrix preparation and connectome construction.

``abc`` and ``harmonization`` each expose a ``build_expression_matrix(cfg)`` reaching
the same output contract by different cell-type assignment procedures; import
the modules rather than the functions to keep the distinction visible.
"""

from clrc.preprocessing import abc, harmonization
from clrc.preprocessing.connectome import (
    log_normalize,
    looks_like_raw_counts,
    normalize_if_needed,
    run_connectome,
    run_connectome_by_subject,
    subset_to_db_genes,
)

__all__ = [
    "abc",
    "harmonization",
    "log_normalize",
    "looks_like_raw_counts",
    "normalize_if_needed",
    "run_connectome",
    "run_connectome_by_subject",
    "subset_to_db_genes",
]
