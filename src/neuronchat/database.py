"""Interaction database loading and management."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from neuronchat.core import InteractionEntry

_VALID_SPECIES = ("mouse", "human")


def _unwrap_scalar(v: Any) -> Any:
    """Unwrap single-element lists produced by R's toJSON."""
    if isinstance(v, list) and len(v) == 1:
        return v[0]
    return v


def _entry_from_dict(d: dict[str, Any]) -> InteractionEntry:
    """Convert a JSON dict to an InteractionEntry."""
    return InteractionEntry(
        lig_contributor=d["lig_contributor"],
        receptor_subunit=d["receptor_subunit"],
        lig_contributor_group=d["lig_contributor_group"],
        lig_contributor_coeff=[float(x) for x in d["lig_contributor_coeff"]],
        receptor_subunit_group=d["receptor_subunit_group"],
        receptor_subunit_coeff=[float(x) for x in d["receptor_subunit_coeff"]],
        # interaction_type and ligand_type are scalars in R but become
        # single-element lists in JSON due to R's toJSON(auto_unbox=FALSE)
        interaction_type=_unwrap_scalar(d["interaction_type"]),
        ligand_type=_unwrap_scalar(d["ligand_type"]),
    )


def load_db(species_or_path: str | Path) -> dict[str, InteractionEntry]:
    """Load an interaction database.

    Parameters
    ----------
    species_or_path : str or Path
        'mouse', 'human', or a path to a custom JSON file.

    Returns
    -------
    dict mapping interaction name -> InteractionEntry, sorted by name.
    """
    path_obj = Path(species_or_path)

    if isinstance(species_or_path, Path) or (
        str(species_or_path) not in _VALID_SPECIES and path_obj.exists()
    ):
        with open(path_obj) as f:
            raw = json.load(f)
    elif str(species_or_path) in _VALID_SPECIES:
        ref = resources.files("neuronchat.data").joinpath(
            f"interactionDB_{species_or_path}.json"
        )
        with resources.as_file(ref) as path:
            with open(path) as f:
                raw = json.load(f)
    else:
        raise FileNotFoundError(
            f"'{species_or_path}' is not a valid species "
            f"({', '.join(_VALID_SPECIES)}) and no file exists at that path."
        )

    db = {name: _entry_from_dict(raw[name]) for name in sorted(raw.keys())}
    return db


def update_interaction_db(
    db: dict[str, InteractionEntry],
    interaction_name: str,
    lig_contributor: list[str],
    receptor_subunit: list[str],
    interaction_type: str = "user_defined",
    ligand_type: str = "user_defined",
    lig_contributor_group: list[int] | None = None,
    lig_contributor_coeff: list[float] | None = None,
    receptor_subunit_group: list[int] | None = None,
    receptor_subunit_coeff: list[float] | None = None,
) -> dict[str, InteractionEntry]:
    """Add a new interaction to the database.

    Parameters
    ----------
    db : dict
        Current interaction database.
    interaction_name : str
        Name for the new interaction. Must not already exist in db.
    lig_contributor : list[str]
        Gene symbols contributing to ligand abundance.
    receptor_subunit : list[str]
        Gene symbols for receptor subunits.
    lig_contributor_group : list[int] | None
        Group assignment per ligand gene. Defaults to all-1.
    lig_contributor_coeff : list[float] | None
        Stoichiometric coefficient per group. Defaults to [1.0].
    receptor_subunit_group : list[int] | None
        Group assignment per receptor gene. Defaults to all-1.
    receptor_subunit_coeff : list[float] | None
        Stoichiometric coefficient per group. Defaults to [1.0].

    Returns
    -------
    New dict with the entry appended (original dict is not mutated).
    """
    if interaction_name in db:
        raise ValueError(
            f"Interaction name '{interaction_name}' already exists in the database"
        )

    if lig_contributor_group is None:
        lig_contributor_group = [1] * len(lig_contributor)
        lig_contributor_coeff = [1.0]
    elif lig_contributor_coeff is None:
        raise ValueError(
            "lig_contributor_coeff must be provided when lig_contributor_group is specified"
        )
    if receptor_subunit_group is None:
        receptor_subunit_group = [1] * len(receptor_subunit)
        receptor_subunit_coeff = [1.0]
    elif receptor_subunit_coeff is None:
        raise ValueError(
            "receptor_subunit_coeff must be provided when receptor_subunit_group is specified"
        )

    if len(lig_contributor_group) != len(lig_contributor):
        raise ValueError(
            "lig_contributor_group length doesn't match lig_contributor length"
        )
    if len(receptor_subunit_group) != len(receptor_subunit):
        raise ValueError(
            "receptor_subunit_group length doesn't match receptor_subunit length"
        )
    if len(lig_contributor_coeff) != len(set(lig_contributor_group)):
        raise ValueError(
            "lig_contributor_coeff length doesn't match number of unique lig_contributor_group values"
        )
    if len(receptor_subunit_coeff) != len(set(receptor_subunit_group)):
        raise ValueError(
            "receptor_subunit_coeff length doesn't match number of unique receptor_subunit_group values"
        )

    new_entry = InteractionEntry(
        lig_contributor=lig_contributor,
        receptor_subunit=receptor_subunit,
        lig_contributor_group=lig_contributor_group,
        lig_contributor_coeff=lig_contributor_coeff,
        receptor_subunit_group=receptor_subunit_group,
        receptor_subunit_coeff=receptor_subunit_coeff,
        interaction_type=interaction_type,
        ligand_type=ligand_type,
    )

    updated = dict(db)
    updated[interaction_name] = new_entry
    return updated
