"""Abstract compute backend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ComputeBackend(ABC):
    """Abstract base for NeuronChat compute backends."""

    @abstractmethod
    def cal_expr_by_group(
        self,
        df: pd.DataFrame,
        gene_used: list[str],
        mean_method: str | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Compute per-group expression.

        Returns
        -------
        (expr_array, group_names)
            expr_array: (n_groups x n_genes) float64 array
            group_names: sorted list of group labels
        """
        ...

    @abstractmethod
    def cal_prob_mtx(
        self,
        df: pd.DataFrame,
        sender: list[str],
        receiver: list[str],
        lig_genes: list[str],
        lig_groups: list[int],
        lig_coeffs: list[float],
        rec_genes: list[str],
        rec_groups: list[int],
        rec_coeffs: list[float],
        method: str | None = None,
        K: float = 0.5,
        mean_method: str | None = None,
    ) -> dict:
        """Compute communication probability matrix for one interaction.

        Returns
        -------
        dict with keys: prob_mtx, ligand_score, receptor_score
        """
        ...

    @abstractmethod
    def permutation_test(
        self,
        df: pd.DataFrame,
        sender: list[str],
        receiver: list[str],
        lig_genes: list[str],
        lig_groups: list[int],
        lig_coeffs: list[float],
        rec_genes: list[str],
        rec_groups: list[int],
        rec_coeffs: list[float],
        M: int,
        fdr: float,
        method: str | None = None,
        K: float = 0.5,
        mean_method: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """Run permutation test for one interaction.

        Returns
        -------
        dict with keys: net, pvalue, net0, FC, info,
                        ligand_abundance, target_abundance
        """
        ...
