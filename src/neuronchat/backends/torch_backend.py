"""PyTorch/GPU compute backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from neuronchat._utils import bh_fdr_filter
from neuronchat.backends.base import ComputeBackend


class TorchBackend(ComputeBackend):
    """GPU compute backend using PyTorch.

    Moves grouped expression aggregation and probability matrix computation
    to GPU for each permutation.  Supports multi-GPU by splitting M
    permutations across devices via ThreadPoolExecutor.

    Device specification:
    - ``"cuda"``  — auto-detect all available GPUs
    - ``"cuda:0"`` — single specific GPU
    - ``["cuda:0", "cuda:1"]`` — explicit multi-GPU

    The observed probability matrix is always computed on CPU (via
    NumpyBackend) so that M=0 results are bit-identical to the CPU backend.
    Permuted matrices are computed on GPU; floating-point results may differ
    slightly from CPU (different order of operations in quantile, etc.).
    """

    def __init__(self, device: str | list[str] = "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install PyTorch with CUDA support "
                "or use device='cpu'."
            )
        if isinstance(device, list):
            self.devices = [torch.device(d) for d in device]
        elif device == "cuda":
            n_gpus = torch.cuda.device_count()
            self.devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
        else:
            self.devices = [torch.device(device)]

        self.device = self.devices[0]

        from neuronchat.backends.numpy_backend import NumpyBackend
        self._cpu = NumpyBackend()

    # ------------------------------------------------------------------
    # Public API — delegate to CPU for single-shot computations
    # ------------------------------------------------------------------

    def cal_expr_by_group(self, df, gene_used, mean_method=None):
        return self._cpu.cal_expr_by_group(df, gene_used, mean_method)

    def cal_prob_mtx(self, df, sender, receiver, lig_genes, lig_groups,
                     lig_coeffs, rec_genes, rec_groups, rec_coeffs,
                     method=None, K=0.5, mean_method=None):
        return self._cpu.cal_prob_mtx(
            df, sender, receiver, lig_genes, lig_groups, lig_coeffs,
            rec_genes, rec_groups, rec_coeffs, method, K, mean_method,
        )

    # ------------------------------------------------------------------
    # GPU helper kernels (static — used by _permutation_chunk workers)
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_grouped_expr(data_t, labels_t, n_groups, mean_method):
        """Compute per-group expression on GPU.

        Parameters
        ----------
        data_t : Tensor (n_cells, n_genes) float64 on GPU
        labels_t : Tensor (n_cells,) int64 on GPU, 0-indexed group codes
        n_groups : int
        mean_method : str | None

        Returns
        -------
        Tensor (n_groups, n_genes) float64 on GPU
        """
        n_genes = data_t.shape[1]
        device = data_t.device
        result = torch.empty(
            (n_groups, n_genes), dtype=torch.float64, device=device
        )

        # Sort by group for contiguous memory access — avoids per-group
        # boolean masking which launches O(n_groups) small GPU kernels.
        order = torch.argsort(labels_t, stable=True)
        sorted_data = data_t[order]

        counts = torch.bincount(labels_t, minlength=n_groups)
        boundaries = torch.cumsum(counts, dim=0)
        starts = torch.zeros(n_groups, dtype=torch.long, device=device)
        starts[1:] = boundaries[:-1]

        # Transfer to CPU for Python loop control
        starts_np = starts.cpu().numpy()
        boundaries_np = boundaries.cpu().numpy()
        counts_np = counts.cpu().numpy()

        for i in range(n_groups):
            if counts_np[i] == 0:
                result[i] = 0.0
                continue
            group_data = sorted_data[starts_np[i]:boundaries_np[i]]
            if mean_method == "mean":
                result[i] = group_data.mean(dim=0)
            else:
                q1 = torch.quantile(group_data, 0.25, dim=0, interpolation="linear")
                q2 = torch.quantile(group_data, 0.50, dim=0, interpolation="linear")
                q3 = torch.quantile(group_data, 0.75, dim=0, interpolation="linear")
                result[i] = 0.25 * q1 + 0.5 * q2 + 0.25 * q3
        return result

    @staticmethod
    def _gpu_stoichiometric_expr(gene_expr, groups, coeffs, n_groups):
        """GPU port of NumpyBackend._stoichiometric_expr.

        Parameters
        ----------
        gene_expr : Tensor (n_groups, n_component_genes) on GPU
        groups : list[int] — 1-indexed group IDs parallel to genes
        coeffs : list[float] — stoichiometric coefficients per group
        n_groups : int

        Returns
        -------
        Tensor (n_groups,) on GPU
        """
        device = gene_expr.device
        groups_t = torch.tensor(groups, dtype=torch.int64, device=device)
        n_coeff = len(coeffs)

        group_expr = torch.zeros(
            (n_groups, n_coeff), dtype=torch.float64, device=device
        )
        for i in range(n_coeff):
            group_id = i + 1  # R is 1-indexed
            gene_idx = (groups_t == group_id).nonzero(as_tuple=True)[0]
            if len(gene_idx) == 0:
                group_expr[:, i] = 1.0
            elif len(gene_idx) == 1:
                group_expr[:, i] = gene_expr[:, gene_idx[0]]
            else:
                group_expr[:, i] = gene_expr[:, gene_idx].mean(dim=1)

        if n_coeff == 1 and coeffs[0] == 1:
            return group_expr[:, 0]

        rep_coeff = []
        for i in range(n_coeff):
            rep_coeff.extend([i] * int(coeffs[i]))
        expanded = group_expr[:, rep_coeff]

        return torch.exp(torch.mean(torch.log(expanded), dim=1))

    @staticmethod
    def _gpu_prob_mtx(expr_gene, n_lig, sender_idx, receiver_idx,
                      lig_groups, lig_coeffs, rec_groups, rec_coeffs,
                      method, K):
        """Compute probability matrix on GPU.

        Parameters
        ----------
        expr_gene : Tensor (n_groups, n_genes) on GPU
        n_lig : int
        sender_idx, receiver_idx : list[int]
        lig_groups, lig_coeffs, rec_groups, rec_coeffs : interaction params
        method : str | None
        K : float

        Returns
        -------
        (prob_mtx_sub, lig_expr, rec_expr) — tensors on GPU
        """
        n_groups = expr_gene.shape[0]

        if method is None:
            lig_expr = TorchBackend._gpu_stoichiometric_expr(
                expr_gene[:, :n_lig], lig_groups, lig_coeffs, n_groups
            )
            rec_expr = TorchBackend._gpu_stoichiometric_expr(
                expr_gene[:, n_lig:], rec_groups, rec_coeffs, n_groups
            )
            prob_mtx = lig_expr.unsqueeze(1) * rec_expr.unsqueeze(0)

        elif method == "CellChat":
            if n_lig > 1:
                lig_expr = torch.exp(
                    torch.mean(torch.log(expr_gene[:, :n_lig]), dim=1)
                )
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = expr_gene.shape[1] - n_lig
            if n_rec > 1:
                rec_expr = torch.exp(
                    torch.mean(torch.log(expr_gene[:, n_lig:]), dim=1)
                )
            else:
                rec_expr = expr_gene[:, n_lig]

            lig_hill = lig_expr ** 2 / (lig_expr ** 2 + K ** 2)
            rec_hill = rec_expr ** 2 / (rec_expr ** 2 + K ** 2)
            prob_mtx = lig_hill.unsqueeze(1) * rec_hill.unsqueeze(0)

        elif method == "CellPhoneDB":
            if n_lig > 1:
                lig_expr = torch.min(expr_gene[:, :n_lig], dim=1).values
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = expr_gene.shape[1] - n_lig
            if n_rec > 1:
                rec_expr = torch.min(expr_gene[:, n_lig:], dim=1).values
            else:
                rec_expr = expr_gene[:, n_lig]

            lig_bcast = lig_expr.unsqueeze(1).expand(-1, n_groups)
            rec_bcast = rec_expr.unsqueeze(0).expand(n_groups, -1)
            prob_mtx = (lig_bcast + rec_bcast) / 2
            prob_mtx = prob_mtx * (lig_bcast > 0) * (rec_bcast > 0)

        else:
            raise ValueError(f"Unknown method: '{method}'")

        prob_mtx_sub = prob_mtx[sender_idx][:, receiver_idx]
        return prob_mtx_sub, lig_expr, rec_expr

    # ------------------------------------------------------------------
    # Permutation worker (one per GPU)
    # ------------------------------------------------------------------

    @staticmethod
    def _permutation_chunk(device, data_np, label_codes_np,
                           sender_mask, receiver_mask,
                           sender_idx, receiver_idx,
                           n_groups, n_lig,
                           lig_groups, lig_coeffs,
                           rec_groups, rec_coeffs,
                           method, K, mean_method,
                           M_chunk, rng, obs_prob_np):
        """Run *M_chunk* permutations on a specific GPU.

        Label permutation uses numpy RNG on CPU for cross-device seed
        reproducibility; expression aggregation and probability computation
        run on GPU.

        Returns
        -------
        count : ndarray (n_sender, n_receiver)
            Number of times permuted prob > observed prob.
        """
        dev = torch.device(device)
        data_t = torch.tensor(data_np, dtype=torch.float64, device=dev)
        obs_t = torch.tensor(obs_prob_np, dtype=torch.float64, device=dev)

        n_sender = len(sender_idx)
        n_receiver = len(receiver_idx)
        count_t = torch.zeros(
            (n_sender, n_receiver), dtype=torch.float64, device=dev
        )

        for _j in range(M_chunk):
            labels_j = label_codes_np.copy()
            labels_j[sender_mask] = rng.permutation(
                label_codes_np[sender_mask]
            )
            labels_j[receiver_mask] = rng.permutation(
                label_codes_np[receiver_mask]
            )

            labels_t = torch.tensor(labels_j, dtype=torch.int64, device=dev)

            expr_gene = TorchBackend._gpu_grouped_expr(
                data_t, labels_t, n_groups, mean_method
            )
            prob_sub, _, _ = TorchBackend._gpu_prob_mtx(
                expr_gene, n_lig, sender_idx, receiver_idx,
                lig_groups, lig_coeffs, rec_groups, rec_coeffs,
                method, K,
            )

            count_t += (prob_sub > obs_t).to(torch.float64)

        return count_t.cpu().numpy()

    # ------------------------------------------------------------------
    # Permutation test — main entry point
    # ------------------------------------------------------------------

    def permutation_test(self, df, sender, receiver, lig_genes, lig_groups,
                         lig_coeffs, rec_genes, rec_groups, rec_coeffs,
                         M=100, fdr=0.05, method=None, K=0.5,
                         mean_method=None, seed=None):
        # Compute observed on CPU for exact numpy compatibility
        obs = self._cpu.cal_prob_mtx(
            df, sender, receiver, lig_genes, lig_groups, lig_coeffs,
            rec_genes, rec_groups, rec_coeffs, method, K, mean_method,
        )
        prob_mtx = obs["prob_mtx"]

        fc_lig = obs["ligand_score"].max() - obs["ligand_score"].min()
        fc_rec = obs["receptor_score"].max() - obs["receptor_score"].min()
        fc = max(fc_lig, fc_rec)

        if M == 0 or prob_mtx.max() == 0:
            return {
                "net": prob_mtx.copy(),
                "pvalue": np.full_like(prob_mtx, np.nan),
                "net0": prob_mtx.copy(),
                "FC": fc,
                "info": prob_mtx.sum(),
                "ligand_abundance": obs["ligand_score"],
                "target_abundance": obs["receptor_score"],
            }

        # -- Preprocess DataFrame to numpy arrays for GPU transfer ----------

        gene_used = list(lig_genes) + list(rec_genes)
        groups_sorted = sorted(df["cell_subclass"].unique())
        n_groups = len(groups_sorted)
        group_to_code = {g: i for i, g in enumerate(groups_sorted)}

        data_np = df[gene_used].values.astype(np.float64)
        original_labels = df["cell_subclass"].values
        label_codes_np = np.array(
            [group_to_code[lbl] for lbl in original_labels], dtype=np.int64
        )

        sender_codes = np.array(
            [group_to_code[s] for s in sender], dtype=np.int64
        )
        receiver_codes = np.array(
            [group_to_code[r] for r in receiver], dtype=np.int64
        )
        sender_mask = np.isin(label_codes_np, sender_codes)
        receiver_mask = np.isin(label_codes_np, receiver_codes)

        sender_idx = [group_to_code[s] for s in sender]
        receiver_idx = [group_to_code[r] for r in receiver]
        n_lig = len(lig_genes)

        # -- Dispatch permutations ------------------------------------------

        n_devices = len(self.devices)
        base_rng = np.random.default_rng(seed)

        if n_devices == 1:
            count = self._permutation_chunk(
                str(self.devices[0]), data_np, label_codes_np,
                sender_mask, receiver_mask,
                sender_idx, receiver_idx,
                n_groups, n_lig,
                lig_groups, lig_coeffs,
                rec_groups, rec_coeffs,
                method, K, mean_method,
                M, base_rng, prob_mtx,
            )
        else:
            # Split M permutations across GPUs
            child_rngs = base_rng.spawn(n_devices)
            base_chunk = M // n_devices
            remainder = M % n_devices
            chunks = [
                base_chunk + (1 if i < remainder else 0)
                for i in range(n_devices)
            ]

            with ThreadPoolExecutor(max_workers=n_devices) as pool:
                futures = []
                for i, dev in enumerate(self.devices):
                    if chunks[i] == 0:
                        continue
                    futures.append(pool.submit(
                        TorchBackend._permutation_chunk,
                        str(dev), data_np, label_codes_np,
                        sender_mask, receiver_mask,
                        sender_idx, receiver_idx,
                        n_groups, n_lig,
                        lig_groups, lig_coeffs,
                        rec_groups, rec_coeffs,
                        method, K, mean_method,
                        chunks[i], child_rngs[i], prob_mtx,
                    ))

                count = np.zeros_like(prob_mtx)
                for f in futures:
                    count += f.result()

        pvalue = count / M
        net = bh_fdr_filter(prob_mtx, pvalue, fdr, M=M)

        return {
            "net": net,
            "pvalue": pvalue,
            "net0": prob_mtx,
            "FC": fc,
            "info": net.sum(),
            "ligand_abundance": obs["ligand_score"],
            "target_abundance": obs["receptor_score"],
        }

    # ------------------------------------------------------------------
    # Batched permutation — all interactions share grouped expression
    # ------------------------------------------------------------------

    @staticmethod
    def _permutation_chunk_batched(
        data_t, codes_np,
        sender_mask, receiver_mask,
        sender_idx, receiver_idx,
        n_groups, interaction_specs,
        obs_prob_3d,
        method, K, mean_method,
        M_chunk, rng, counts_dtype=np.int16, pbar=None,
    ):
        """Run M_chunk permutations for ALL interactions on one GPU.

        The key optimization: grouped expression (expensive quantile
        aggregation over all cells) is computed ONCE per permutation for
        ALL genes on GPU, then each interaction indexes into the result.
        The per-interaction probability computation and comparison is
        vectorized using 3-D numpy operations across all interactions.

        Parameters
        ----------
        data_t : torch.Tensor
            Pre-created GPU tensor (n_cells, n_all_genes), float64.
        codes_np : ndarray (n_cells,) int64
        sender_mask, receiver_mask : ndarray (n_cells,) bool
        sender_idx, receiver_idx : list[int]
        n_groups : int
        interaction_specs : list of tuples
            Each: ``(col_indices, n_lig, lig_groups, lig_coeffs,
            rec_groups, rec_coeffs)``.
        obs_prob_3d : ndarray (n_sender, n_receiver, n_interactions)
            Pre-stacked observed probability matrices.
        method, K, mean_method : permutation test parameters
        M_chunk : int
        rng : numpy Generator
        pbar : tqdm bar or None

        Returns
        -------
        list of ndarray (n_sender, n_receiver)
            Count arrays per interaction.
        """
        from neuronchat.backends.numpy_backend import NumpyBackend

        n_interactions = len(interaction_specs)
        n_sender = len(sender_idx)
        n_receiver = len(receiver_idx)
        sender_ix = np.array(sender_idx, dtype=np.intp)
        receiver_ix = np.array(receiver_idx, dtype=np.intp)

        counts_3d = np.zeros(
            (n_sender, n_receiver, n_interactions), dtype=counts_dtype
        )

        for _j in range(M_chunk):
            # Permute labels on CPU for seed reproducibility
            codes_j = codes_np.copy()
            codes_j[sender_mask] = rng.permutation(codes_np[sender_mask])
            codes_j[receiver_mask] = rng.permutation(codes_np[receiver_mask])

            # Compute grouped expression for ALL genes on GPU (once)
            labels_t = torch.tensor(codes_j, dtype=torch.int64, device=data_t.device)
            expr_all_gpu = TorchBackend._gpu_grouped_expr(
                data_t, labels_t, n_groups, mean_method
            )

            # Transfer to CPU (small: n_groups x n_all_genes)
            expr_all = expr_all_gpu.cpu().numpy()

            # Batch: compute lig/rec for all interactions, then
            # vectorized outer product + comparison.
            lig_exprs, rec_exprs = NumpyBackend._compute_lig_rec_batched(
                expr_all, interaction_specs, n_groups, method, K,
            )
            lig_sub = lig_exprs[sender_ix]    # (n_sender, n_interactions)
            rec_sub = rec_exprs[receiver_ix]  # (n_receiver, n_interactions)

            NumpyBackend._prob_compare_accumulate(
                lig_sub, rec_sub, obs_prob_3d, counts_3d, method, K,
            )

            if pbar is not None:
                pbar.update(1)

        # Convert to list for compatibility with aggregation code
        return [counts_3d[:, :, i] for i in range(n_interactions)]

    def permutation_test_batched(
        self,
        data_np,
        codes_np,
        n_groups,
        sender_idx,
        receiver_idx,
        interaction_specs,
        M=100, fdr=0.05, method=None, K=0.5,
        mean_method=None, seed=None,
        progress=False,
    ):
        """Run permutation test for multiple interactions in a batch.

        Instead of processing each interaction independently (each computing
        grouped expression from scratch), this method computes grouped
        expression for ALL genes once per permutation, then each interaction
        indexes into the result.  This reduces the expensive grouped
        expression computation from ``M * n_interactions`` to ``M`` calls.

        Parameters
        ----------
        data_np : ndarray (n_cells, n_all_genes) float64
        codes_np : ndarray (n_cells,) int64, 0-indexed group codes
        n_groups : int
        sender_idx, receiver_idx : list[int]
        interaction_specs : list of tuples
            Each: ``(col_indices, n_lig, lig_groups, lig_coeffs,
            rec_groups, rec_coeffs)``.
        M : int
        fdr : float
        method, K, mean_method : computation parameters
        seed : int | None
        progress : bool

        Returns
        -------
        list of dict
            Per-interaction results with keys: net, pvalue, net0, FC,
            info, ligand_abundance, target_abundance.
        """
        from neuronchat._utils import bh_fdr_filter
        from neuronchat.backends.numpy_backend import NumpyBackend

        n_interactions = len(interaction_specs)
        if n_interactions == 0:
            return []

        n_sender = len(sender_idx)
        n_receiver = len(receiver_idx)
        sender_ix = np.array(sender_idx, dtype=np.intp)
        receiver_ix = np.array(receiver_idx, dtype=np.intp)

        # Step 1: Observed grouped expression — on GPU (faster quantiles)
        dev = self.devices[0]
        data_t = torch.tensor(data_np, dtype=torch.float64, device=dev)
        codes_t = torch.tensor(codes_np, dtype=torch.int64, device=dev)
        expr_all = self._gpu_grouped_expr(
            data_t, codes_t, n_groups, mean_method
        ).cpu().numpy()
        del codes_t
        # data_t kept alive — reused by permutation chunks on device 0

        # Step 2: Observed lig/rec for all interactions (batched, CPU)
        lig_exprs, rec_exprs = NumpyBackend._compute_lig_rec_batched(
            expr_all, interaction_specs, n_groups, method, K,
        )

        # FC: vectorized across all interactions
        fc_lig = lig_exprs.max(axis=0) - lig_exprs.min(axis=0)
        fc_rec = rec_exprs.max(axis=0) - rec_exprs.min(axis=0)
        fc_vals = np.maximum(fc_lig, fc_rec)

        # Sender/receiver subsetting
        lig_sub = lig_exprs[sender_ix]    # (n_sender, n_interactions)
        rec_sub = rec_exprs[receiver_ix]  # (n_receiver, n_interactions)

        # Build results with scalar/1D fields
        results = [None] * n_interactions
        for i in range(n_interactions):
            results[i] = {
                "FC": fc_vals[i],
                "ligand_abundance": lig_exprs[:, i].copy(),
                "target_abundance": rec_exprs[:, i].copy(),
            }

        # Handle M=0: build all obs_prob via chunked outer products
        if M == 0:
            obs_3d = np.empty(
                (n_sender, n_receiver, n_interactions), dtype=np.float64
            )
            NumpyBackend._build_obs_prob_3d(
                lig_sub, rec_sub, obs_3d, method, K,
            )
            for i in range(n_interactions):
                entry = results[i]
                assert entry is not None
                entry["net0"] = obs_3d[:, :, i]
                entry["net"] = obs_3d[:, :, i].copy()
                entry["pvalue"] = np.full(
                    (n_sender, n_receiver), np.nan,
                )
                entry["info"] = obs_3d[:, :, i].sum()
            return results

        # Determine which interactions need permutation (vectorized).
        # An interaction has non-zero obs_prob iff there exist sender s
        # and receiver r with both lig[s] > 0 and rec[r] > 0.
        needs_perm = (lig_sub > 0).any(axis=0) & (rec_sub > 0).any(axis=0)
        perm_indices = np.where(needs_perm)[0]
        n_perm = len(perm_indices)

        # Fill results for interactions that don't need permutation
        zero_net = np.zeros((n_sender, n_receiver), dtype=np.float64)
        for i in range(n_interactions):
            if not needs_perm[i]:
                entry = results[i]
                assert entry is not None
                entry["net0"] = zero_net.copy()
                entry["net"] = zero_net.copy()
                entry["pvalue"] = np.full(
                    (n_sender, n_receiver), np.nan,
                )
                entry["info"] = 0.0

        if n_perm == 0:
            return results

        # Use compact int dtype for counts — saves ~75% memory vs float64
        _counts_dtype = np.int16 if M <= np.iinfo(np.int16).max else np.int32

        # Build perm_obs_3d directly via chunked outer products —
        # avoids the ~39 GB copy from np.stack.
        lig_perm = lig_sub[:, perm_indices]   # (n_sender, n_perm)
        rec_perm = rec_sub[:, perm_indices]   # (n_receiver, n_perm)
        perm_obs_3d = np.empty(
            (n_sender, n_receiver, n_perm), dtype=np.float64,
        )
        NumpyBackend._build_obs_prob_3d(
            lig_perm, rec_perm, perm_obs_3d, method, K,
        )

        # Store net0 as views into perm_obs_3d (no copy)
        for k, orig_i in enumerate(perm_indices):
            results[orig_i]["net0"] = perm_obs_3d[:, :, k]

        # Step 3: Permutation test
        perm_specs = [interaction_specs[i] for i in perm_indices]
        sender_codes = np.array(sender_idx, dtype=np.int64)
        receiver_codes = np.array(receiver_idx, dtype=np.int64)
        sender_mask = np.isin(codes_np, sender_codes)
        receiver_mask = np.isin(codes_np, receiver_codes)

        base_rng = np.random.default_rng(seed)
        n_devices = len(self.devices)

        pbar = None
        if progress:
            from tqdm.auto import tqdm
            pbar = tqdm(total=M, desc="GPU permutations", unit="perm")

        try:
            if n_devices == 1:
                counts = self._permutation_chunk_batched(
                    data_t, codes_np,
                    sender_mask, receiver_mask,
                    sender_idx, receiver_idx,
                    n_groups, perm_specs, perm_obs_3d,
                    method, K, mean_method,
                    M, base_rng, _counts_dtype, pbar,
                )
            else:
                child_rngs = base_rng.spawn(n_devices)
                base_chunk = M // n_devices
                remainder = M % n_devices
                chunks = [
                    base_chunk + (1 if i < remainder else 0)
                    for i in range(n_devices)
                ]

                # Pre-create GPU tensors: reuse device 0, upload to others
                data_ts = [data_t]
                for i in range(1, n_devices):
                    if chunks[i] > 0:
                        data_ts.append(torch.tensor(
                            data_np, dtype=torch.float64,
                            device=self.devices[i],
                        ))
                    else:
                        data_ts.append(None)

                with ThreadPoolExecutor(max_workers=n_devices) as pool:
                    futures = []
                    for i, dev in enumerate(self.devices):
                        if chunks[i] == 0:
                            continue
                        futures.append(pool.submit(
                            TorchBackend._permutation_chunk_batched,
                            data_ts[i], codes_np,
                            sender_mask, receiver_mask,
                            sender_idx, receiver_idx,
                            n_groups, perm_specs, perm_obs_3d,
                            method, K, mean_method,
                            chunks[i], child_rngs[i], _counts_dtype, pbar,
                        ))

                    # Memory-efficient aggregation: reuse first thread's
                    # result, accumulate in-place, free futures immediately.
                    counts = None
                    for i, f in enumerate(futures):
                        chunk_counts = f.result()
                        futures[i] = None
                        if counts is None:
                            counts = chunk_counts
                        else:
                            for k in range(len(perm_specs)):
                                counts[k] += chunk_counts[k]
                            del chunk_counts
                    del futures
        finally:
            if pbar is not None:
                pbar.close()

        del data_t  # Free GPU tensor

        assert counts is not None  # guaranteed set above since n_perm > 0

        # Step 4: p-values and FDR — free counts progressively
        for k, orig_i in enumerate(perm_indices):
            pvalue = counts[k] / M
            counts[k] = None
            results[orig_i]["pvalue"] = pvalue
            results[orig_i]["net"] = bh_fdr_filter(
                results[orig_i]["net0"], pvalue, fdr, M=M,
            )
            results[orig_i]["info"] = results[orig_i]["net"].sum()
        del counts

        return results
