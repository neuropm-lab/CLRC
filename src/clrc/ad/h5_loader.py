"""Load subject-level NeuronChat H5 files (Python NeuronChat format only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def parse_region_celltype(label: str) -> Tuple[str, str]:
    """Parse 'REGION::celltype' into (region, celltype)."""
    parts = label.split("::")
    if len(parts) == 2:
        return parts[0], parts[1]
    return label, label


def load_subject_h5(filepath: str | Path) -> Dict:
    """Load a Python NeuronChat H5 file.

    The Python NeuronChat port stores ``net/`` and ``net0/`` as HDF5 groups
    with one dataset per interaction (2D arrays), unlike the old R format
    which stored a single 3D array.

    Returns dict with keys: net, net0, labels_lr, labels_region_ct.
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        # Check format
        if "format_version" in f.attrs:
            fmt = str(f.attrs["format_version"])
            if fmt not in ("neuronchat_python_v1", "1"):
                logger.warning("Unrecognized format_version=%s in %s", fmt, filepath)

        interaction_names = list(f.attrs["interaction_names"])
        group_names = list(f.attrs["group_names"])

        n_lr = len(interaction_names)
        n_groups = len(group_names)
        net_3d = np.zeros((n_lr, n_groups, n_groups))
        net0_3d = np.zeros((n_lr, n_groups, n_groups))

        for lr_idx, name in enumerate(interaction_names):
            if "net" in f and name in f["net"]:
                net_3d[lr_idx] = f["net"][name][:]
            if "net0" in f and name in f["net0"]:
                net0_3d[lr_idx] = f["net0"][name][:]

    return {
        "net": net_3d,
        "net0": net0_3d,
        "labels_lr": interaction_names,
        "labels_region_ct": group_names,
    }
