"""Pathway enrichment analysis: Enrichr ORA + GSEA prerank."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from clrc.biology.classification import parse_gene_string

try:
    import gseapy as gp

    HAS_GSEAPY = True
except ImportError:
    gp = None  # ty: ignore[invalid-assignment]
    HAS_GSEAPY = False

logger = logging.getLogger(__name__)

ENRICHMENT_DATABASES = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathway_2023_Human",
]


def run_pathway_enrichment(
    gene_list: List[str],
    databases: List[str] = ENRICHMENT_DATABASES,
    organism: str = "human",
    outdir: Optional[Path] = None,
    prefix: str = "",
) -> Optional[pd.DataFrame]:
    """Run Enrichr ORA via gseapy.enrichr()."""
    if not HAS_GSEAPY:
        warnings.warn("GSEApy not available. Skipping enrichment.")
        return None
    if len(gene_list) < 3:
        warnings.warn(f"Too few genes ({len(gene_list)}) for enrichment. Skipping.")
        return None
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=databases,
            organism=organism,
            outdir=str(outdir) if outdir else None,
            no_plot=True,
        )
        results = enr.results
        if results is not None and len(results) > 0:
            # gseapy ships no type stubs; ty misinfers .results as list[DataFrame]
            results = results[results["Adjusted P-value"] < 0.05].copy()  # ty: ignore[invalid-argument-type]
            results = results.sort_values("Combined Score", ascending=False)
            if outdir and prefix:
                outdir.mkdir(parents=True, exist_ok=True)
                results.to_csv(outdir / f"{prefix}_enrichment.csv", index=False)
            return results
    except Exception as e:
        warnings.warn(f"Enrichment failed: {e}")
    return None


def run_enrichment_for_all_categories(
    genes_by_category: Dict[str, Dict[str, List[str]]],
    outdir: Path,
) -> Dict[str, pd.DataFrame]:
    """Run enrichment for each category's gene list."""
    results: Dict[str, pd.DataFrame] = {}
    for category, gene_dict in genes_by_category.items():
        all_genes = gene_dict.get("all_genes", [])
        if all_genes:
            prefix = category.lower().replace("-", "_").replace(" ", "_")
            result = run_pathway_enrichment(
                all_genes,
                outdir=outdir / "pathway_enrichment",
                prefix=prefix,
            )
            if result is not None and len(result) > 0:
                results[category] = result
    return results


def create_gene_ranking(
    categorized: pd.DataFrame,
    full_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a ranked gene list for GSEA prerank based on log2(SC/FC)."""
    gene_scores: Dict[str, List[float]] = {}

    lr_to_genes: Dict[str, Dict[str, List[str]]] = {}
    for _, row in full_feature_df.iterrows():
        lr_name = row.get("lr_name", "")
        lg = parse_gene_string(row.get("ligand_genes", ""))
        rg = parse_gene_string(row.get("receptor_genes", ""))
        if lr_name and (lg or rg):
            lr_to_genes[lr_name] = {"ligands": lg, "receptors": rg}

    eps = 1e-10
    for _, row in categorized.iterrows():
        lr_name = row["group_name"]
        sc_imp = row["importance_sc"]
        fc_imp = row["importance_fc"]
        log2_ratio = np.log2((sc_imp + eps) / (fc_imp + eps))

        if lr_name in lr_to_genes:
            all_g = lr_to_genes[lr_name]["ligands"] + lr_to_genes[lr_name]["receptors"]
        elif "_" in lr_name:
            parts = lr_name.split("_")
            all_g = [parts[0], "_".join(parts[1:])]
        else:
            continue

        for gene in all_g:
            if gene:
                gene_scores.setdefault(gene, []).append(log2_ratio)

    ranking_data = [
        {
            "gene": gene,
            "ranking_score": np.mean(scores),
            "n_interactions": len(scores),
        }
        for gene, scores in gene_scores.items()
    ]
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values("ranking_score", ascending=False).reset_index(
        drop=True
    )
    return ranking_df


def run_gsea_prerank(
    ranking_df: pd.DataFrame,
    outdir: Path,
    gene_sets: Optional[List[str]] = None,
    min_size: int = 10,
    max_size: int = 500,
    permutation_num: int = 1000,
) -> dict:
    """Run GSEA prerank analysis using gseapy."""
    if not HAS_GSEAPY:
        logger.warning("gseapy not available, skipping GSEA prerank")
        return {}

    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2023",
            "GO_Molecular_Function_2023",
            "KEGG_2021_Human",
            "Reactome_2022",
        ]

    outdir.mkdir(parents=True, exist_ok=True)
    ranking = ranking_df.set_index("gene")["ranking_score"]

    results = {}
    for gs_name in gene_sets:
        try:
            logger.info("Running GSEA prerank with %s...", gs_name)
            gs_results = gp.prerank(
                rnk=ranking,
                gene_sets=gs_name,
                outdir=str(outdir / gs_name),
                min_size=min_size,
                max_size=max_size,
                permutation_num=permutation_num,
                seed=42,
                verbose=False,
            )
            res_df = gs_results.res2d
            if res_df is not None and len(res_df) > 0:
                sig_results = res_df[res_df["FDR q-val"] < 0.25].copy()
                sig_results = sig_results.sort_values("NES", ascending=False)

                res_df.to_csv(outdir / f"{gs_name}_all_results.csv", index=False)
                if len(sig_results) > 0:
                    sig_results.to_csv(
                        outdir / f"{gs_name}_significant.csv", index=False
                    )

                results[gs_name] = {
                    "all": res_df,
                    "significant": sig_results,
                    "n_significant": len(sig_results),
                }
                logger.info(
                    "Found %d significant gene sets (FDR < 0.25)", len(sig_results)
                )
            else:
                results[gs_name] = None
        except Exception as e:
            logger.warning("Error with %s: %s", gs_name, e)
            results[gs_name] = None
    return results


def summarize_gsea_results(
    gsea_results: dict, outdir: Path
) -> pd.DataFrame:
    """Aggregate and save GSEA prerank results across databases."""
    all_sig, all_results = [], []

    for db_name, db_results in gsea_results.items():
        if db_results is None:
            continue
        if db_results.get("significant") is not None:
            sig_df = db_results["significant"].copy()
            if len(sig_df) > 0:
                sig_df["database"] = db_name
                all_sig.append(sig_df)
        if db_results.get("all") is not None:
            all_df = db_results["all"].copy()
            if len(all_df) > 0:
                all_df["database"] = db_name
                all_results.append(all_df)

    if all_results:
        combined_all = pd.concat(all_results, ignore_index=True)
        combined_all["NES"] = pd.to_numeric(combined_all["NES"], errors="coerce")
        combined_all["FDR q-val"] = pd.to_numeric(
            combined_all["FDR q-val"], errors="coerce"
        )
        combined_all = combined_all.dropna(subset=["NES"])
        combined_all = combined_all.sort_values("NES", ascending=False)
        combined_all.to_csv(outdir / "gsea_all_results_combined.csv", index=False)
        logger.info("Total gene sets tested: %d", len(combined_all))
    else:
        combined_all = pd.DataFrame()

    if not all_sig:
        logger.info("No significant GSEA results (FDR < 0.25)")
        return combined_all

    summary = pd.concat(all_sig, ignore_index=True)
    summary["NES"] = pd.to_numeric(summary["NES"], errors="coerce")
    summary["FDR q-val"] = pd.to_numeric(summary["FDR q-val"], errors="coerce")
    summary = summary.dropna(subset=["NES"])

    sc_enriched = summary[summary["NES"] > 0].sort_values("NES", ascending=False)
    fc_enriched = summary[summary["NES"] < 0].sort_values("NES", ascending=True)

    if len(sc_enriched) > 0:
        sc_enriched.to_csv(outdir / "gsea_sc_enriched.csv", index=False)
    if len(fc_enriched) > 0:
        fc_enriched.to_csv(outdir / "gsea_fc_enriched.csv", index=False)

    summary.to_csv(outdir / "gsea_summary.csv", index=False)
    return summary
