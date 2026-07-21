"""NeuronChat main analysis runner."""

from __future__ import annotations

import copy

import numpy as np
from joblib import Parallel, delayed

from clrc.core.parallel import tqdm_joblib
from neuronchat.backends import get_backend
from neuronchat.core import NeuronChat


def _filter_genes(entry, available_genes, strict):
    """Filter interaction genes to those present and check availability.

    Returns (lig_genes, lig_groups, rec_genes, rec_groups, lig_coeffs,
    rec_coeffs, ok) where ok is False if the interaction should be skipped.
    """
    lig_mask = [g in available_genes for g in entry.lig_contributor]
    rec_mask = [g in available_genes for g in entry.receptor_subunit]
    lig_genes = [g for g, m in zip(entry.lig_contributor, lig_mask) if m]
    rec_genes = [g for g, m in zip(entry.receptor_subunit, rec_mask) if m]
    lig_groups = [g for g, m in zip(entry.lig_contributor_group, lig_mask) if m]
    rec_groups = [g for g, m in zip(entry.receptor_subunit_group, rec_mask) if m]

    lig_present = set(lig_groups)
    rec_present = set(rec_groups)
    all_lig = set(entry.lig_contributor_group)
    all_rec = set(entry.receptor_subunit_group)

    if strict == 1:
        ok = (all_lig == (all_lig & lig_present)) and (all_rec == (all_rec & rec_present))
    else:
        ok = len(lig_present) > 0 and len(rec_present) > 0

    return lig_genes, lig_groups, rec_genes, rec_groups, entry.lig_contributor_coeff, entry.receptor_subunit_coeff, ok


def _process_interaction_arrays(
    backend, data_np, codes_np, n_groups, sender_idx, receiver_idx,
    entry, gene_to_col, available_genes, strict, M, fdr, K, method,
    mean_method, seed,
):
    """Process a single interaction using pre-extracted numpy arrays (CPU path)."""
    lig_genes, lig_groups, rec_genes, rec_groups, lig_coeffs, rec_coeffs, ok = \
        _filter_genes(entry, available_genes, strict)

    if not ok:
        zero_net = np.zeros((len(sender_idx), len(receiver_idx)), dtype=np.float64)
        return {
            "net": zero_net, "pvalue": zero_net.copy(), "net0": zero_net.copy(),
            "FC": 0.0, "info": 0.0,
            "ligand_abundance": np.zeros(0), "target_abundance": np.zeros(0),
        }

    gene_used = list(lig_genes) + list(rec_genes)
    col_indices = [gene_to_col[g] for g in gene_used]
    interaction_data = data_np[:, col_indices]

    return backend._permutation_test_from_arrays(
        interaction_data, codes_np, n_groups, sender_idx, receiver_idx,
        n_lig=len(lig_genes), lig_groups=lig_groups, lig_coeffs=lig_coeffs,
        rec_groups=rec_groups, rec_coeffs=rec_coeffs,
        M=M, fdr=fdr, method=method, K=K, mean_method=mean_method, seed=seed,
    )


def _process_interaction_df(
    backend, data_signaling, sender, receiver,
    entry, available_genes, strict, M, fdr, K, method, mean_method, seed,
):
    """Process a single interaction using DataFrame (GPU path)."""
    lig_genes, lig_groups, rec_genes, rec_groups, lig_coeffs, rec_coeffs, ok = \
        _filter_genes(entry, available_genes, strict)

    if not ok:
        zero_net = np.zeros((len(sender), len(receiver)), dtype=np.float64)
        return {
            "net": zero_net, "pvalue": zero_net.copy(), "net0": zero_net.copy(),
            "FC": 0.0, "info": 0.0,
            "ligand_abundance": np.zeros(0), "target_abundance": np.zeros(0),
        }

    return backend.permutation_test(
        data_signaling, sender, receiver,
        lig_genes, lig_groups, lig_coeffs,
        rec_genes, rec_groups, rec_coeffs,
        M, fdr, method, K, mean_method, seed,
    )


def run_neuronchat(
    obj: NeuronChat,
    sender: list[str] | None = None,
    receiver: list[str] | None = None,
    M: int = 100,
    fdr: float = 0.05,
    K: float = 0.5,
    method: str | None = None,
    mean_method: str | None = None,
    strict: int = 1,
    n_jobs: int = 4,
    device: str | list[str] = "cpu",
    seed: int | None = None,
    progress: bool = True,
) -> NeuronChat:
    """Run NeuronChat communication analysis.

    Parameters
    ----------
    obj : NeuronChat
        Object created by create_neuronchat.
    sender, receiver : list[str] | None
        Cell groups. None means all groups.
    M : int
        Number of permutations. 0 skips permutation test.
    fdr : float
        FDR cutoff for BH correction.
    K : float
        Hill coefficient for CellChat method.
    method : str | None
        None (default), 'CellChat', or 'CellPhoneDB'.
    mean_method : str | None
        None (quantile-weighted) or 'mean'.
    strict : int
        1: require all gene groups present. 0: require any.
    n_jobs : int
        Number of parallel workers (CPU backend only).
    device : str | list[str]
        ``'cpu'`` for NumPy backend, ``'cuda'`` (auto-detect all GPUs),
        ``'cuda:N'`` (single GPU), or ``['cuda:0', 'cuda:1']``
        (explicit multi-GPU) for PyTorch GPU backend.
    seed : int | None
        Random seed for reproducible permutation tests. Each interaction
        receives a deterministic per-interaction seed derived from this
        value (seed + interaction_index) to ensure reproducibility with
        parallel execution. None uses non-deterministic randomness.
    progress : bool
        Show tqdm progress bar for interaction processing.

    Returns
    -------
    NeuronChat object with net, net0, pvalue, fc, info,
    ligand_abundance, target_abundance populated.
    """
    backend = get_backend(device)

    # Default sender/receiver: all groups
    all_groups = sorted(obj.data_signaling["cell_subclass"].unique())
    if sender is None:
        sender = all_groups
    if receiver is None:
        receiver = all_groups

    available_genes = set(
        c for c in obj.data_signaling.columns if c != "cell_subclass"
    )

    interaction_names = list(obj.db.keys())
    entries = list(obj.db.values())

    # Determine parallelism and backend capabilities
    effective_jobs = n_jobs if (isinstance(device, str) and device == "cpu") else 1
    use_batched_gpu = hasattr(backend, 'permutation_test_batched')
    use_array_path = hasattr(backend, '_permutation_test_from_arrays')

    if use_batched_gpu or use_array_path:
        # Pre-extract numpy arrays from DataFrame ONCE.
        gene_cols = sorted(available_genes)
        data_np = obj.data_signaling[gene_cols].values.astype(np.float64)
        gene_to_col = {g: i for i, g in enumerate(gene_cols)}

        labels = obj.data_signaling["cell_subclass"].values
        group_to_code = {g: i for i, g in enumerate(all_groups)}
        codes_np = np.array(
            [group_to_code[g] for g in labels], dtype=np.int64,
        )
        n_groups = len(all_groups)

        sender_idx = [group_to_code[s] for s in sender]
        receiver_idx = [group_to_code[r] for r in receiver]

    if use_batched_gpu:
        # === GPU batched path ===
        # All interactions processed together: grouped expression computed
        # once per permutation for ALL genes, then each interaction indexes
        # into the result (1000x fewer expensive GPU calls).
        interaction_specs = []
        skip_indices = set()

        for j, entry in enumerate(entries):
            lig_genes, lig_groups, rec_genes, rec_groups, lig_coeffs, rec_coeffs, ok = \
                _filter_genes(entry, available_genes, strict)

            if not ok:
                skip_indices.add(j)
                continue

            gene_used = list(lig_genes) + list(rec_genes)
            col_indices = [gene_to_col[g] for g in gene_used]
            interaction_specs.append((
                col_indices, len(lig_genes),
                lig_groups, lig_coeffs, rec_groups, rec_coeffs,
            ))

        # Temporarily free the DataFrame during heavy GPU work to reduce
        # peak host memory usage.
        _saved_df = obj.data_signaling
        obj.data_signaling = None
        try:
            batched_results = backend.permutation_test_batched(
                data_np, codes_np, n_groups, sender_idx, receiver_idx,
                interaction_specs, M=M, fdr=fdr, method=method, K=K,
                mean_method=mean_method, seed=seed, progress=progress,
            )
        finally:
            obj.data_signaling = _saved_df
        del _saved_df

        # Free numpy arrays no longer needed
        del data_np, codes_np

        # Rebuild full results list (insert zeros for skipped interactions)
        results = []
        valid_i = 0
        for j in range(len(entries)):
            if j in skip_indices:
                zero_net = np.zeros((len(sender), len(receiver)), dtype=np.float64)
                results.append({
                    "net": zero_net, "pvalue": zero_net.copy(), "net0": zero_net.copy(),
                    "FC": 0.0, "info": 0.0,
                    "ligand_abundance": np.zeros(0), "target_abundance": np.zeros(0),
                })
            else:
                results.append(batched_results[valid_i])
                valid_i += 1

    else:
        # === Per-interaction paths (CPU array or legacy DataFrame) ===
        interaction_seeds = (
            [seed + i for i in range(len(entries))] if seed is not None else
            [None] * len(entries)
        )

        if use_array_path:
            tasks = [
                delayed(_process_interaction_arrays)(
                    backend, data_np, codes_np, n_groups, sender_idx, receiver_idx,
                    entry, gene_to_col, available_genes, strict,
                    M, fdr, K, method, mean_method, i_seed,
                )
                for entry, i_seed in zip(entries, interaction_seeds)
            ]
        else:
            n_groups = len(all_groups)
            tasks = [
                delayed(_process_interaction_df)(
                    backend, obj.data_signaling, sender, receiver,
                    entry, available_genes, strict,
                    M, fdr, K, method, mean_method, i_seed,
                )
                for entry, i_seed in zip(entries, interaction_seeds)
            ]

        if progress:
            from tqdm.auto import tqdm

            with tqdm_joblib(
                tqdm(total=len(tasks), desc="Interactions", unit="pair")
            ):
                results = Parallel(n_jobs=effective_jobs)(tasks)
        else:
            results = Parallel(n_jobs=effective_jobs)(tasks)

    # Collect results
    n_interactions = len(interaction_names)

    net = {}
    net0 = {}
    pvalue = {}
    fc_all = np.zeros(n_interactions, dtype=np.float64)
    info_all = np.zeros(n_interactions, dtype=np.float64)
    lig_abundance = np.zeros((n_groups, n_interactions), dtype=np.float64)
    tar_abundance = np.zeros((n_groups, n_interactions), dtype=np.float64)

    for j, (name, res) in enumerate(zip(interaction_names, results)):
        net[name] = res["net"]
        net0[name] = res["net0"]
        pvalue[name] = res["pvalue"]
        fc_all[j] = res["FC"]
        info_all[j] = res["info"]
        if len(res["ligand_abundance"]) > 0:
            lig_abundance[:, j] = res["ligand_abundance"]
            tar_abundance[:, j] = res["target_abundance"]

    # Return new object with results
    result = copy.copy(obj)
    result.net = net
    result.net0 = net0
    result.pvalue = pvalue
    result.fc = fc_all
    result.info = info_all
    result.ligand_abundance = lig_abundance
    result.target_abundance = tar_abundance
    result.group_names = all_groups
    result.sender_names = list(sender)
    result.receiver_names = list(receiver)

    return result
