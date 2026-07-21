"""Core data model for NeuronChat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse


@dataclass
class InteractionEntry:
    """A single ligand-receptor interaction from the database."""

    lig_contributor: list[str]
    receptor_subunit: list[str]
    lig_contributor_group: list[int]
    lig_contributor_coeff: list[float]
    receptor_subunit_group: list[int]
    receptor_subunit_coeff: list[float]
    interaction_type: str
    ligand_type: str


@dataclass
class NeuronChat:
    """Container for NeuronChat analysis state.

    Mirrors the R S4 class slots. Mutable fields (net, pvalue, etc.)
    are populated by run_neuronchat().
    """

    # Required at creation
    data: np.ndarray | scipy.sparse.spmatrix | None
    data_signaling: pd.DataFrame
    meta: pd.DataFrame
    idents: pd.Categorical
    db: dict[str, InteractionEntry]
    lr: list[str]

    # Gene names for data matrix columns (needed for merge union behavior)
    var_names: list[str] | None = None

    # Populated by run_neuronchat
    net: dict[str, np.ndarray] | None = None
    net0: dict[str, np.ndarray] | None = None
    pvalue: dict[str, np.ndarray] | None = None
    fc: np.ndarray | None = None
    info: np.ndarray | None = None
    ligand_abundance: np.ndarray | None = None
    target_abundance: np.ndarray | None = None

    # Group labels (populated by run_neuronchat)
    group_names: list[str] | None = None      # all groups (rows of abundance)
    sender_names: list[str] | None = None     # rows of net/net0/pvalue
    receiver_names: list[str] | None = None   # cols of net/net0/pvalue

    mode: str = "single"
