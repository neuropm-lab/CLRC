"""Save and load NeuronChat results."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from neuronchat.core import NeuronChat


def save_h5(obj: NeuronChat, path: str | Path) -> None:
    """Save NeuronChat results to an HDF5 file.

    Saves the output of run_neuronchat (net, net0, pvalue, fc, info,
    ligand_abundance, target_abundance) plus metadata (interaction names,
    group labels).

    Parameters
    ----------
    obj : NeuronChat
        Object with run_neuronchat results populated.
    path : str or Path
        Output file path (typically .h5).
    """
    if obj.net is None:
        raise ValueError(
            "No results to save. Run run_neuronchat() first."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    interaction_names = list(obj.net.keys())

    with h5py.File(path, "w") as f:
        # Metadata
        f.attrs["mode"] = obj.mode
        f.attrs["interaction_names"] = interaction_names
        if obj.group_names is not None:
            f.attrs["group_names"] = obj.group_names
        if obj.sender_names is not None:
            f.attrs["sender_names"] = obj.sender_names
        if obj.receiver_names is not None:
            f.attrs["receiver_names"] = obj.receiver_names

        # Dict-of-arrays: net, net0, pvalue
        for slot_name in ("net", "net0", "pvalue"):
            slot = getattr(obj, slot_name)
            if slot is None:
                continue
            grp = f.create_group(slot_name)
            for name, arr in slot.items():
                grp.create_dataset(name, data=arr)

        # 1D/2D arrays
        if obj.fc is not None:
            f.create_dataset("fc", data=obj.fc)
        if obj.info is not None:
            f.create_dataset("info", data=obj.info)
        if obj.ligand_abundance is not None:
            f.create_dataset("ligand_abundance", data=obj.ligand_abundance)
        if obj.target_abundance is not None:
            f.create_dataset("target_abundance", data=obj.target_abundance)


def load_h5(path: str | Path) -> dict:
    """Load NeuronChat results from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Path to .h5 file created by save_h5.

    Returns
    -------
    dict with keys: net, net0, pvalue, fc, info, ligand_abundance,
    target_abundance, interaction_names, mode.
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        mode = f.attrs["mode"]
        interaction_names = list(f.attrs["interaction_names"])

        result = {
            "mode": mode,
            "interaction_names": interaction_names,
            "group_names": (
                list(f.attrs["group_names"])
                if "group_names" in f.attrs else None
            ),
            "sender_names": (
                list(f.attrs["sender_names"])
                if "sender_names" in f.attrs else None
            ),
            "receiver_names": (
                list(f.attrs["receiver_names"])
                if "receiver_names" in f.attrs else None
            ),
        }

        # Dict-of-arrays
        for slot_name in ("net", "net0", "pvalue"):
            if slot_name in f:
                grp = f[slot_name]
                result[slot_name] = {
                    name: np.array(grp[name]) for name in interaction_names
                    if name in grp
                }
            else:
                result[slot_name] = None

        # 1D/2D arrays
        for key in ("fc", "info", "ligand_abundance", "target_abundance"):
            result[key] = np.array(f[key]) if key in f else None

    return result


def to_adata(obj: NeuronChat, adata) -> None:
    """Write NeuronChat results into an AnnData object's .uns slot.

    Stores results under adata.uns["neuronchat"] as nested dicts of
    numpy arrays, which AnnData serializes natively to .h5ad.

    Parameters
    ----------
    obj : NeuronChat
        Object with run_neuronchat results populated.
    adata : anndata.AnnData
        AnnData object to write into. Modified in-place.
    """
    if obj.net is None:
        raise ValueError(
            "No results to save. Run run_neuronchat() first."
        )

    interaction_names = list(obj.net.keys())

    uns = {
        "interaction_names": np.array(interaction_names, dtype=object),
        "mode": obj.mode,
    }

    for slot_name in ("net", "net0", "pvalue"):
        slot = getattr(obj, slot_name)
        if slot is not None:
            uns[slot_name] = {name: arr.copy() for name, arr in slot.items()}

    for key in ("fc", "info", "ligand_abundance", "target_abundance"):
        val = getattr(obj, key)
        if val is not None:
            uns[key] = val.copy()

    adata.uns["neuronchat"] = uns
