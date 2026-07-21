"""LR interaction classification: interaction class and pathway class."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Set

import pandas as pd

logger = logging.getLogger(__name__)

# --- Cell class mapping (ABC Atlas 31 cell types) --------------------------
CELL_CLASS_MAP: Mapping[str, str] = {
    "Splatter": "Other",
    "Deep-layer corticothalamic and 6b": "Excitatory",
    "MGE interneuron": "Inhibitory",
    "LAMP5-LHX6 and Chandelier": "Inhibitory",
    "Deep-layer intratelencephalic": "Excitatory",
    "Upper-layer intratelencephalic": "Excitatory",
    "Miscellaneous": "Other",
    "Deep-layer near-projecting": "Excitatory",
    "Midbrain-derived inhibitory": "Inhibitory",
    "Astrocyte": "Glia",
    "CGE interneuron": "Inhibitory",
    "Amygdala excitatory": "Excitatory",
    "Hippocampal CA1-3": "Other",
    "Medium spiny neuron": "Other",
    "Eccentric medium spiny neuron": "Other",
    "Vascular": "Other",
    "Oligodendrocyte precursor": "Glia",
    "Fibroblast": "Other",
    "Oligodendrocyte": "Glia",
    "Thalamic excitatory": "Excitatory",
    "Upper rhombic lip": "Other",
    "Cerebellar inhibitory": "Inhibitory",
    "Committed oligodendrocyte precursor": "Glia",
    "Microglia": "Glia",
    "Ependymal": "Other",
    "Choroid plexus": "Other",
    # --- Additional ABC Atlas H5 cell types not covered above ---
    "Bergmann glia": "Glia",
    "Hippocampal CA4": "Other",
    "Hippocampal dentate gyrus": "Other",
    "Lower rhombic lip": "Other",
    "Mammillary body": "Other",
}

# --- LR pathway classification patterns -----------------------------------
LR_PATHWAY_PATTERNS: Dict[str, List[str]] = {
    "Neurexin-Neuroligin": ["NRXN", "NLGN", "LRRTM", "CLSTN"],
    "Semaphorin-Plexin": ["SEMA", "NRP", "PLXN"],
    "Slit-Robo": ["SLIT", "ROBO"],
    "Neuregulin-ERBB": ["NRG", "ERBB"],
    "Ephrin": ["EFN", "EPH"],
    "Laminin-Integrin": ["LAM", "ITG", "CD44"],
    "Teneurin-Latrophilin": ["TENM", "ADGRL"],
    "FLRT-Adhesion": ["FLRT", "UNC5"],
    "Cerebellin-Grid": ["CBLN", "GRID"],
    "GABA_Signaling": ["GAD", "GABR", "SLC6A"],
    "Glutamate_Signaling": ["Glutamate", "Glu-"],
    "Cell_Adhesion": ["NEGR", "LRRC4", "NTNG", "C1QL", "ADGRB"],
    "Other": [],
}


def classify_lr(lr_name: str) -> str:
    """Map an LR interaction name to a high-level interaction class."""
    n = str(lr_name).upper()
    if "GABA-A" in n or "GABR_A" in n:
        return "GABA-A"
    elif "GABA-B" in n or "GABBR" in n:
        return "GABA-B"
    elif ("GLUTAMATE" in n or n.startswith("GLU") or "GRIK" in n
          or "GRIA" in n or "GRIN" in n or "GRM" in n):
        return "Glutamate"
    elif "NRXN" in n:
        return "Neurexin"
    elif "NRG" in n and "ERBB" in n:
        return "Neuregulin-ErbB"
    elif "SEMA" in n or "PLXN" in n:
        return "Semaphorin-Plexin"
    elif "SLIT" in n and "ROBO" in n:
        return "Slit-Robo"
    elif n.startswith("COL"):
        return "Collagen"
    elif "NLGN" in n:
        return "Neuroligin"
    elif "LRRTM" in n or "LRRC4" in n:
        return "LRRTM/LRRC"
    elif "FLRT" in n:
        return "FLRT"
    elif "EFNA" in n or "EPHA" in n or "EPHB" in n:
        return "Ephrin"
    elif "5-HT" in n or "HTR" in n:
        return "Serotonin"
    elif "SPP1" in n or ("ITGA" in n and "ITGB" in n):
        return "Integrin"
    elif n.startswith("BMP") or n.startswith("GDF"):
        return "BMP/GDF"
    elif n.startswith("WNT") or "FZD" in n:
        return "Wnt"
    elif n.startswith("FGF"):
        return "FGF"
    else:
        return "Other"


def classify_lr_pathway(lr_name: str) -> str:
    """Classify an L-R interaction into a broad pathway class."""
    lr_upper = lr_name.upper()
    for pathway_class, patterns in LR_PATHWAY_PATTERNS.items():
        if pathway_class == "Other":
            continue
        for pattern in patterns:
            if pattern.upper() in lr_upper:
                return pathway_class
    return "Other"


def parse_gene_string(gene_str) -> List[str]:
    """Parse a gene string that may contain multiple genes separated by '+'."""
    if pd.isna(gene_str) or gene_str == "":
        return []
    return [g.strip() for g in str(gene_str).split("+") if g.strip()]


def extract_genes_for_category(
    full_feature_df: pd.DataFrame,
    lr_names: List[str],
    separate_ligand_receptor: bool = False,
) -> Dict[str, List[str]]:
    """Extract unique gene symbols from LR names matching a specific bias category.

    Looks up ``lr_names`` in ``full_feature_df`` (matched against
    ``lr_name``), parses the ``ligand_genes`` and ``receptor_genes`` columns
    via :func:`parse_gene_string`, and returns the de-duplicated gene sets.

    Parameters
    ----------
    full_feature_df
        Full feature-importance DataFrame with ``lr_name``, ``ligand_genes``
        and ``receptor_genes`` columns.
    lr_names
        List of LR interaction names (matched against ``lr_name``).
    separate_ligand_receptor
        When ``True``, return separate ligand, receptor and union gene
        lists. When ``False`` (default), return only the union under
        ``"all_genes"``.
    """
    subset = full_feature_df[full_feature_df["lr_name"].isin(lr_names)]
    ligand_genes: Set[str] = set()
    receptor_genes: Set[str] = set()

    for _, row in subset.iterrows():
        ligand_genes.update(parse_gene_string(row.get("ligand_genes", "")))
        receptor_genes.update(parse_gene_string(row.get("receptor_genes", "")))

    ligand_genes.discard("")
    receptor_genes.discard("")

    if separate_ligand_receptor:
        return {
            "ligands": sorted(ligand_genes),
            "receptors": sorted(receptor_genes),
            "all_genes": sorted(ligand_genes | receptor_genes),
        }
    return {"all_genes": sorted(ligand_genes | receptor_genes)}


def extract_genes_by_category(
    categorized_df: pd.DataFrame,
    full_feature_df: pd.DataFrame,
) -> Dict[str, Dict[str, List[str]]]:
    """Dict wrapper over :func:`extract_genes_for_category` for all 3 categories.

    Iterates over the standard bias labels (``SC-biased``, ``FC-biased``,
    ``Balanced``), selects the LR names with that category from
    ``categorized_df["group_name"]`` and extracts ligand / receptor /
    combined gene lists via :func:`extract_genes_for_category`. Empty
    categories still appear in the output with empty lists so the schema
    is stable for downstream code.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    for category in ["SC-biased", "FC-biased", "Balanced"]:
        lr_names = categorized_df.loc[
            categorized_df["category"] == category, "group_name"
        ].tolist()
        if lr_names:
            result[category] = extract_genes_for_category(
                full_feature_df, lr_names, separate_ligand_receptor=True
            )
        else:
            result[category] = {"ligands": [], "receptors": [], "all_genes": []}
    return result
