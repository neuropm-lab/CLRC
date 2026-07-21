"""NumPy/CPU compute backend."""

from __future__ import annotations

import numpy as np

from neuronchat.backends.base import ComputeBackend


class NumpyBackend(ComputeBackend):
    """CPU compute backend using NumPy."""

    def cal_expr_by_group(self, df, gene_used, mean_method=None):
        if mean_method is not None and mean_method != "mean":
            raise ValueError(
                f"Unknown mean_method: '{mean_method}'. "
                "Supported values: None (quantile-weighted) or 'mean'."
            )

        groups_sorted = sorted(df["cell_subclass"].unique())
        n_groups = len(groups_sorted)
        n_genes = len(gene_used)
        result = np.empty((n_groups, n_genes), dtype=np.float64)

        for i, group in enumerate(groups_sorted):
            mask = df["cell_subclass"] == group
            group_data = df.loc[mask, gene_used].values  # (n_cells_in_group, n_genes)

            if mean_method == "mean":
                result[i, :] = group_data.mean(axis=0)
            else:
                q1 = np.quantile(group_data, 0.25, axis=0, method="linear")
                q2 = np.quantile(group_data, 0.50, axis=0, method="linear")
                q3 = np.quantile(group_data, 0.75, axis=0, method="linear")
                result[i, :] = 0.25 * q1 + 0.5 * q2 + 0.25 * q3

        return result, groups_sorted

    @staticmethod
    def _cal_expr_by_group_numpy(data, codes, n_groups, mean_method=None):
        """Compute per-group expression from pure numpy arrays.

        This is the hot-path optimized version of cal_expr_by_group. It
        avoids pandas overhead and string comparisons by working with
        integer group codes and contiguous array slicing.

        Parameters
        ----------
        data : ndarray (n_cells, n_genes) float64
        codes : ndarray (n_cells,) int64, 0-indexed group codes
        n_groups : int
        mean_method : str | None

        Returns
        -------
        ndarray (n_groups, n_genes) float64
        """
        if mean_method is not None and mean_method != "mean":
            raise ValueError(
                f"Unknown mean_method: '{mean_method}'. "
                "Supported values: None (quantile-weighted) or 'mean'."
            )

        n_genes = data.shape[1]
        result = np.empty((n_groups, n_genes), dtype=np.float64)

        # Sort by group code for contiguous memory access
        order = np.argsort(codes, kind="stable")
        sorted_data = data[order]

        # Group boundaries via bincount
        counts = np.bincount(codes, minlength=n_groups)
        boundaries = np.cumsum(counts)
        starts = np.empty(n_groups, dtype=np.intp)
        starts[0] = 0
        starts[1:] = boundaries[:-1]

        for i in range(n_groups):
            if counts[i] == 0:
                result[i, :] = 0.0
                continue
            group_data = sorted_data[starts[i]:boundaries[i]]
            if mean_method == "mean":
                result[i, :] = group_data.mean(axis=0)
            else:
                q1 = np.quantile(group_data, 0.25, axis=0, method="linear")
                q2 = np.quantile(group_data, 0.50, axis=0, method="linear")
                q3 = np.quantile(group_data, 0.75, axis=0, method="linear")
                result[i, :] = 0.25 * q1 + 0.5 * q2 + 0.25 * q3

        return result

    def cal_prob_mtx(self, df, sender, receiver, lig_genes, lig_groups,
                     lig_coeffs, rec_genes, rec_groups, rec_coeffs,
                     method=None, K=0.5, mean_method=None):
        gene_used = list(lig_genes) + list(rec_genes)
        expr_gene, cell_rownames = self.cal_expr_by_group(df, gene_used, mean_method)
        n_groups = len(cell_rownames)
        n_lig = len(lig_genes)

        if method is None:
            lig_expr = self._stoichiometric_expr(
                expr_gene[:, :n_lig], lig_groups, lig_coeffs, n_groups
            )
            rec_expr = self._stoichiometric_expr(
                expr_gene[:, n_lig:], rec_groups, rec_coeffs, n_groups
            )
            prob_mtx = lig_expr.reshape(-1, 1) @ rec_expr.reshape(1, -1)

        elif method == "CellChat":
            if n_lig > 1:
                lig_expr = np.exp(np.mean(np.log(expr_gene[:, :n_lig]), axis=1))
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = len(rec_genes)
            if n_rec > 1:
                rec_expr = np.exp(np.mean(np.log(expr_gene[:, n_lig:]), axis=1))
            else:
                rec_expr = expr_gene[:, n_lig]

            def hill(x, k):
                return x ** 2 / (x ** 2 + k ** 2)

            prob_mtx = hill(lig_expr.reshape(-1, 1), K) @ hill(rec_expr.reshape(1, -1), K)

        elif method == "CellPhoneDB":
            if n_lig > 1:
                lig_expr = np.min(expr_gene[:, :n_lig], axis=1)
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = len(rec_genes)
            if n_rec > 1:
                rec_expr = np.min(expr_gene[:, n_lig:], axis=1)
            else:
                rec_expr = expr_gene[:, n_lig]

            lig_bcast = lig_expr.reshape(-1, 1) @ np.ones((1, n_groups))
            rec_bcast = np.ones((n_groups, 1)) @ rec_expr.reshape(1, -1)
            prob_mtx = (lig_bcast + rec_bcast) / 2
            prob_mtx *= (lig_bcast > 0) * (rec_bcast > 0)
        else:
            raise ValueError(f"Unknown method: '{method}'")

        # Validate sender/receiver groups exist in data
        missing_senders = [s for s in sender if s not in cell_rownames]
        if missing_senders:
            raise ValueError(
                f"Sender groups not found in data: {missing_senders}. "
                f"Available groups: {cell_rownames}"
            )
        missing_receivers = [r for r in receiver if r not in cell_rownames]
        if missing_receivers:
            raise ValueError(
                f"Receiver groups not found in data: {missing_receivers}. "
                f"Available groups: {cell_rownames}"
            )

        # Sender/receiver subsetting
        sender_idx = [cell_rownames.index(s) for s in sender]
        receiver_idx = [cell_rownames.index(r) for r in receiver]
        prob_mtx_sub = prob_mtx[np.ix_(sender_idx, receiver_idx)]

        return {
            "prob_mtx": prob_mtx_sub,
            "ligand_score": lig_expr,
            "receptor_score": rec_expr,
        }

    @staticmethod
    def _stoichiometric_expr(
        gene_expr: np.ndarray,
        groups: list[int],
        coeffs: list[float],
        n_groups: int,
    ) -> np.ndarray:
        """Compute stoichiometric expression per cell group.

        Matches R logic in cal_prob_mtx_downstream lines 312-327:
        - R iterates i in 1:length(coeff), searching for group == i
        - For each group i, average expression of genes in that group
        - If no genes in group i, set to 1.0 (neutral in geometric mean)
        - Expand by coefficients: repeat each group column by its coeff
        - Geometric mean across expanded columns
        """
        groups_arr = np.array(groups)
        n_coeff = len(coeffs)

        # Step 1: per-group average — iterate over group IDs 1..n_coeff
        # (matching R's for(i in 1:length(coeff)) with which(group == i))
        group_expr = np.zeros((n_groups, n_coeff), dtype=np.float64)
        for i in range(n_coeff):
            group_id = i + 1  # R is 1-indexed
            gene_idx = np.where(groups_arr == group_id)[0]
            if len(gene_idx) == 0:
                group_expr[:, i] = 1.0
            elif len(gene_idx) == 1:
                group_expr[:, i] = gene_expr[:, gene_idx[0]]
            else:
                group_expr[:, i] = gene_expr[:, gene_idx].mean(axis=1)

        # Step 2: stoichiometric combination
        if n_coeff == 1 and coeffs[0] == 1:
            return group_expr[:, 0]

        # Expand: repeat columns by coefficients
        # R: rep(1:length(coeff), coeff) — repeat each column index by its coeff
        rep_coeff = []
        for i in range(n_coeff):
            rep_coeff.extend([i] * int(coeffs[i]))
        expanded = group_expr[:, rep_coeff]

        # Geometric mean across expanded columns
        result = np.exp(np.mean(np.log(expanded), axis=1))
        return result

    def permutation_test(self, df, sender, receiver, lig_genes, lig_groups,
                         lig_coeffs, rec_genes, rec_groups, rec_coeffs,
                         M=100, fdr=0.05, method=None, K=0.5,
                         mean_method=None, seed=None):
        from neuronchat._utils import bh_fdr_filter

        # Compute observed probability matrix (uses DataFrame path, unchanged)
        obs = self.cal_prob_mtx(
            df, sender, receiver, lig_genes, lig_groups, lig_coeffs,
            rec_genes, rec_groups, rec_coeffs, method, K, mean_method,
        )
        prob_mtx = obs["prob_mtx"]

        # FC: max(range(lig), range(rec))
        fc_lig = obs["ligand_score"].max() - obs["ligand_score"].min()
        fc_rec = obs["receptor_score"].max() - obs["receptor_score"].min()
        fc = max(fc_lig, fc_rec)

        # Skip permutation if M=0 or all zeros
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

        rng = np.random.default_rng(seed)

        n_sender = len(sender)
        n_receiver = len(receiver)
        count = np.zeros((n_sender, n_receiver), dtype=np.float64)
        early_stop_threshold = fdr * M

        # ------------------------------------------------------------------
        # ONE-TIME SETUP: convert DataFrame to numpy arrays
        # ------------------------------------------------------------------
        gene_used = list(lig_genes) + list(rec_genes)
        groups_sorted = sorted(df["cell_subclass"].unique())
        n_groups = len(groups_sorted)
        group_to_code = {g: i for i, g in enumerate(groups_sorted)}

        data_np = df[gene_used].values.astype(np.float64)
        codes_np = np.array(
            [group_to_code[g] for g in df["cell_subclass"]],
            dtype=np.int64,
        )

        sender_codes = np.array(
            [group_to_code[s] for s in sender], dtype=np.int64
        )
        receiver_codes = np.array(
            [group_to_code[r] for r in receiver], dtype=np.int64
        )
        sender_mask = np.isin(codes_np, sender_codes)
        receiver_mask = np.isin(codes_np, receiver_codes)

        sender_idx = [group_to_code[s] for s in sender]
        receiver_idx = [group_to_code[r] for r in receiver]
        n_lig = len(lig_genes)

        # Track which cell pairs are still "alive" (not early-stopped)
        alive = np.ones((n_sender, n_receiver), dtype=bool)

        for _j in range(M):
            codes_j = codes_np.copy()

            # Shuffle sender labels among sender cells
            codes_j[sender_mask] = rng.permutation(codes_np[sender_mask])

            # Shuffle receiver labels among receiver cells
            codes_j[receiver_mask] = rng.permutation(codes_np[receiver_mask])

            # Compute group expression via optimized numpy path
            expr_gene = self._cal_expr_by_group_numpy(
                data_np, codes_j, n_groups, mean_method
            )

            # Compute probability matrix (inlined from cal_prob_mtx)
            if method is None:
                lig_expr = self._stoichiometric_expr(
                    expr_gene[:, :n_lig], lig_groups, lig_coeffs, n_groups
                )
                rec_expr = self._stoichiometric_expr(
                    expr_gene[:, n_lig:], rec_groups, rec_coeffs, n_groups
                )
                perm_prob = lig_expr.reshape(-1, 1) @ rec_expr.reshape(1, -1)

            elif method == "CellChat":
                if n_lig > 1:
                    lig_expr = np.exp(
                        np.mean(np.log(expr_gene[:, :n_lig]), axis=1)
                    )
                else:
                    lig_expr = expr_gene[:, 0]
                n_rec = expr_gene.shape[1] - n_lig
                if n_rec > 1:
                    rec_expr = np.exp(
                        np.mean(np.log(expr_gene[:, n_lig:]), axis=1)
                    )
                else:
                    rec_expr = expr_gene[:, n_lig]

                lig_hill = lig_expr ** 2 / (lig_expr ** 2 + K ** 2)
                rec_hill = rec_expr ** 2 / (rec_expr ** 2 + K ** 2)
                perm_prob = lig_hill.reshape(-1, 1) @ rec_hill.reshape(1, -1)

            elif method == "CellPhoneDB":
                if n_lig > 1:
                    lig_expr = np.min(expr_gene[:, :n_lig], axis=1)
                else:
                    lig_expr = expr_gene[:, 0]
                n_rec = expr_gene.shape[1] - n_lig
                if n_rec > 1:
                    rec_expr = np.min(expr_gene[:, n_lig:], axis=1)
                else:
                    rec_expr = expr_gene[:, n_lig]

                lig_bcast = lig_expr.reshape(-1, 1) @ np.ones((1, n_groups))
                rec_bcast = np.ones((n_groups, 1)) @ rec_expr.reshape(1, -1)
                perm_prob = (lig_bcast + rec_bcast) / 2
                perm_prob *= (lig_bcast > 0) * (rec_bcast > 0)
            else:
                raise ValueError(f"Unknown method: '{method}'")

            perm_prob_sub = perm_prob[np.ix_(sender_idx, receiver_idx)]

            exceeds = (perm_prob_sub > prob_mtx).astype(np.float64)
            count += exceeds * alive

            # Sequential rejection: mark pairs where count > fdr * M.
            # Once count exceeds fdr * M, the p-value (count/M) is > fdr,
            # which is guaranteed to fail BH correction (max BH threshold
            # is fdr). The net result is identical to R; however, reported
            # p-values for non-significant entries will be underestimates
            # since we stop counting early.
            alive &= count <= early_stop_threshold

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

    def _compute_prob_components(
        self, expr_gene, n_lig, lig_groups, lig_coeffs,
        rec_groups, rec_coeffs, n_groups, method, K,
    ):
        """Compute probability matrix components from group expression.

        Returns (prob_mtx, lig_expr, rec_expr).
        """
        if method is None:
            lig_expr = self._stoichiometric_expr(
                expr_gene[:, :n_lig], lig_groups, lig_coeffs, n_groups
            )
            rec_expr = self._stoichiometric_expr(
                expr_gene[:, n_lig:], rec_groups, rec_coeffs, n_groups
            )
            prob_mtx = lig_expr.reshape(-1, 1) @ rec_expr.reshape(1, -1)

        elif method == "CellChat":
            if n_lig > 1:
                lig_expr = np.exp(
                    np.mean(np.log(expr_gene[:, :n_lig]), axis=1)
                )
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = expr_gene.shape[1] - n_lig
            if n_rec > 1:
                rec_expr = np.exp(
                    np.mean(np.log(expr_gene[:, n_lig:]), axis=1)
                )
            else:
                rec_expr = expr_gene[:, n_lig]

            lig_hill = lig_expr ** 2 / (lig_expr ** 2 + K ** 2)
            rec_hill = rec_expr ** 2 / (rec_expr ** 2 + K ** 2)
            prob_mtx = lig_hill.reshape(-1, 1) @ rec_hill.reshape(1, -1)

        elif method == "CellPhoneDB":
            if n_lig > 1:
                lig_expr = np.min(expr_gene[:, :n_lig], axis=1)
            else:
                lig_expr = expr_gene[:, 0]
            n_rec = expr_gene.shape[1] - n_lig
            if n_rec > 1:
                rec_expr = np.min(expr_gene[:, n_lig:], axis=1)
            else:
                rec_expr = expr_gene[:, n_lig]

            lig_bcast = lig_expr.reshape(-1, 1) @ np.ones((1, n_groups))
            rec_bcast = np.ones((n_groups, 1)) @ rec_expr.reshape(1, -1)
            prob_mtx = (lig_bcast + rec_bcast) / 2
            prob_mtx *= (lig_bcast > 0) * (rec_bcast > 0)
        else:
            raise ValueError(f"Unknown method: '{method}'")

        return prob_mtx, lig_expr, rec_expr

    @staticmethod
    def _compute_lig_rec_batched(expr_all, interaction_specs, n_groups, method, K):
        """Compute ligand and receptor expressions for ALL interactions.

        Separates the lig/rec computation (per-interaction, must loop due
        to heterogeneous gene structures) from the outer product + comparison
        (can be vectorized in chunks).

        Parameters
        ----------
        expr_all : ndarray (n_groups, n_all_genes) float64
        interaction_specs : list of tuples
            Each: ``(col_indices, n_lig, lig_groups, lig_coeffs,
            rec_groups, rec_coeffs)``.
        n_groups, method, K : computation parameters

        Returns
        -------
        lig_exprs : ndarray (n_groups, n_interactions) float64
        rec_exprs : ndarray (n_groups, n_interactions) float64
        """
        n_interactions = len(interaction_specs)
        lig_exprs = np.empty((n_groups, n_interactions), dtype=np.float64)
        rec_exprs = np.empty((n_groups, n_interactions), dtype=np.float64)

        for i, (col_idx, n_lig, lig_groups, lig_coeffs,
                rec_groups, rec_coeffs) in enumerate(interaction_specs):
            expr_gene = expr_all[:, col_idx]

            if method is None:
                lig_exprs[:, i] = NumpyBackend._stoichiometric_expr(
                    expr_gene[:, :n_lig], lig_groups, lig_coeffs, n_groups
                )
                rec_exprs[:, i] = NumpyBackend._stoichiometric_expr(
                    expr_gene[:, n_lig:], rec_groups, rec_coeffs, n_groups
                )
            elif method == "CellChat":
                if n_lig > 1:
                    lig_exprs[:, i] = np.exp(
                        np.mean(np.log(expr_gene[:, :n_lig]), axis=1)
                    )
                else:
                    lig_exprs[:, i] = expr_gene[:, 0]
                n_rec = expr_gene.shape[1] - n_lig
                if n_rec > 1:
                    rec_exprs[:, i] = np.exp(
                        np.mean(np.log(expr_gene[:, n_lig:]), axis=1)
                    )
                else:
                    rec_exprs[:, i] = expr_gene[:, n_lig]
            elif method == "CellPhoneDB":
                if n_lig > 1:
                    lig_exprs[:, i] = np.min(expr_gene[:, :n_lig], axis=1)
                else:
                    lig_exprs[:, i] = expr_gene[:, 0]
                n_rec = expr_gene.shape[1] - n_lig
                if n_rec > 1:
                    rec_exprs[:, i] = np.min(expr_gene[:, n_lig:], axis=1)
                else:
                    rec_exprs[:, i] = expr_gene[:, n_lig]
            else:
                raise ValueError(f"Unknown method: '{method}'")

        return lig_exprs, rec_exprs

    @staticmethod
    def _build_obs_prob_3d(lig_sub, rec_sub, obs_3d, method, K):
        """Fill observed probability 3D array via chunked outer products.

        Same chunking strategy as ``_prob_compare_accumulate`` to bound
        temporary memory, but writes the probabilities themselves (no
        comparison).  Fills *obs_3d* in-place.

        Parameters
        ----------
        lig_sub : ndarray (n_sender, n_interactions)
        rec_sub : ndarray (n_receiver, n_interactions)
        obs_3d : ndarray (n_sender, n_receiver, n_interactions)
            Filled in-place.
        method : str | None
        K : float
        """
        if method == "CellChat":
            lig_sub = lig_sub ** 2 / (lig_sub ** 2 + K ** 2)
            rec_sub = rec_sub ** 2 / (rec_sub ** 2 + K ** 2)

        n_interactions = lig_sub.shape[1]
        n_sender = lig_sub.shape[0]
        n_receiver = rec_sub.shape[0]

        bytes_per_interaction = n_sender * n_receiver * 8  # float64
        chunk_size = max(1, int(2e9 / max(bytes_per_interaction, 1)))

        for start in range(0, n_interactions, chunk_size):
            end = min(start + chunk_size, n_interactions)
            lig_c = lig_sub[:, start:end]
            rec_c = rec_sub[:, start:end]

            if method is None or method == "CellChat":
                obs_3d[:, :, start:end] = (
                    lig_c[:, np.newaxis, :] * rec_c[np.newaxis, :, :]
                )
            elif method == "CellPhoneDB":
                lig_3d = lig_c[:, np.newaxis, :]
                rec_3d = rec_c[np.newaxis, :, :]
                prob = (lig_3d + rec_3d) / 2
                prob *= (lig_3d > 0) * (rec_3d > 0)
                obs_3d[:, :, start:end] = prob
            else:
                raise ValueError(f"Unknown method: '{method}'")

    @staticmethod
    def _prob_compare_accumulate(
        lig_sub, rec_sub, obs_prob_3d, counts_3d, method, K,
    ):
        """Vectorized prob computation + comparison, accumulated in-place.

        Processes interactions in chunks along the interaction axis to
        bound temporary memory.  Each chunk materializes only
        (n_sender, n_receiver, chunk_size) floats instead of the full
        (n_sender, n_receiver, n_interactions) tensor.

        Parameters
        ----------
        lig_sub : ndarray (n_sender, n_interactions)
        rec_sub : ndarray (n_receiver, n_interactions)
        obs_prob_3d : ndarray (n_sender, n_receiver, n_interactions)
        counts_3d : ndarray (n_sender, n_receiver, n_interactions)
            Accumulated in-place (``+= (prob > obs)``).
        method : str | None
        K : float
        """
        if method == "CellChat":
            lig_sub = lig_sub ** 2 / (lig_sub ** 2 + K ** 2)
            rec_sub = rec_sub ** 2 / (rec_sub ** 2 + K ** 2)

        n_interactions = lig_sub.shape[1]
        n_sender = lig_sub.shape[0]
        n_receiver = rec_sub.shape[0]

        # Target ~2 GB for the float64 temporary per chunk
        bytes_per_interaction = n_sender * n_receiver * 9  # float64 + bool
        chunk_size = max(1, int(2e9 / max(bytes_per_interaction, 1)))

        for start in range(0, n_interactions, chunk_size):
            end = min(start + chunk_size, n_interactions)
            lig_c = lig_sub[:, start:end]
            rec_c = rec_sub[:, start:end]

            if method is None or method == "CellChat":
                prob = lig_c[:, np.newaxis, :] * rec_c[np.newaxis, :, :]
            elif method == "CellPhoneDB":
                lig_3d = lig_c[:, np.newaxis, :]
                rec_3d = rec_c[np.newaxis, :, :]
                prob = (lig_3d + rec_3d) / 2
                prob *= (lig_3d > 0) * (rec_3d > 0)
            else:
                raise ValueError(f"Unknown method: '{method}'")

            counts_3d[:, :, start:end] += prob > obs_prob_3d[:, :, start:end]

    def _permutation_test_from_arrays(
        self, data, codes, n_groups,
        sender_idx, receiver_idx,
        n_lig, lig_groups, lig_coeffs,
        rec_groups, rec_coeffs,
        M=100, fdr=0.05, method=None, K=0.5,
        mean_method=None, seed=None,
    ):
        """Permutation test from pre-extracted numpy arrays.

        Like permutation_test but skips all DataFrame/string conversion.
        Used by run_neuronchat to avoid joblib serialization of DataFrames.

        Parameters
        ----------
        data : ndarray (n_cells, n_interaction_genes) float64
            Gene columns for this interaction only (lig genes then rec genes).
        codes : ndarray (n_cells,) int64
            0-indexed group codes for all cells.
        n_groups : int
            Total number of groups.
        sender_idx : list[int]
            0-indexed group indices for sender groups.
        receiver_idx : list[int]
            0-indexed group indices for receiver groups.
        n_lig : int
            Number of ligand gene columns (first n_lig cols of data).
        """
        from neuronchat._utils import bh_fdr_filter

        # Observed expression
        expr_gene = self._cal_expr_by_group_numpy(
            data, codes, n_groups, mean_method
        )

        # Compute observed probability matrix
        prob_mtx_full, lig_expr, rec_expr = self._compute_prob_components(
            expr_gene, n_lig, lig_groups, lig_coeffs,
            rec_groups, rec_coeffs, n_groups, method, K,
        )
        prob_mtx = prob_mtx_full[np.ix_(sender_idx, receiver_idx)]

        # FC: max(range(lig), range(rec))
        fc_lig = lig_expr.max() - lig_expr.min()
        fc_rec = rec_expr.max() - rec_expr.min()
        fc = max(fc_lig, fc_rec)

        # Skip permutation if M=0 or all zeros
        if M == 0 or prob_mtx.max() == 0:
            return {
                "net": prob_mtx.copy(),
                "pvalue": np.full_like(prob_mtx, np.nan),
                "net0": prob_mtx.copy(),
                "FC": fc,
                "info": prob_mtx.sum(),
                "ligand_abundance": lig_expr,
                "target_abundance": rec_expr,
            }

        rng = np.random.default_rng(seed)

        n_sender = len(sender_idx)
        n_receiver = len(receiver_idx)
        count = np.zeros((n_sender, n_receiver), dtype=np.float64)
        early_stop_threshold = fdr * M

        sender_mask = np.isin(codes, sender_idx)
        receiver_mask = np.isin(codes, receiver_idx)

        alive = np.ones((n_sender, n_receiver), dtype=bool)

        for _j in range(M):
            codes_j = codes.copy()
            codes_j[sender_mask] = rng.permutation(codes[sender_mask])
            codes_j[receiver_mask] = rng.permutation(codes[receiver_mask])

            expr_perm = self._cal_expr_by_group_numpy(
                data, codes_j, n_groups, mean_method
            )

            perm_prob_full, _, _ = self._compute_prob_components(
                expr_perm, n_lig, lig_groups, lig_coeffs,
                rec_groups, rec_coeffs, n_groups, method, K,
            )
            perm_prob_sub = perm_prob_full[np.ix_(sender_idx, receiver_idx)]

            exceeds = (perm_prob_sub > prob_mtx).astype(np.float64)
            count += exceeds * alive

            # Sequential rejection: mark pairs where count > fdr * M.
            # Once count exceeds fdr * M, the p-value (count/M) is > fdr,
            # which is guaranteed to fail BH correction (max BH threshold
            # is fdr). The net result is identical to R; however, reported
            # p-values for non-significant entries will be underestimates
            # since we stop counting early.
            alive &= count <= early_stop_threshold

        pvalue = count / M
        net = bh_fdr_filter(prob_mtx, pvalue, fdr, M=M)

        return {
            "net": net,
            "pvalue": pvalue,
            "net0": prob_mtx,
            "FC": fc,
            "info": net.sum(),
            "ligand_abundance": lig_expr,
            "target_abundance": rec_expr,
        }
