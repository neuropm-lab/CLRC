"""Shared utility functions."""

from __future__ import annotations

import numpy as np


def bh_fdr_filter(
    prob_mtx: np.ndarray, pvalue: np.ndarray, fdr: float, M: int | None = None,
) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction and filter probability matrix.

    Matches the R implementation exactly:
        pvalue_v_sort <- sort.int(pvalue_v, decreasing=FALSE, index.return=TRUE)
        alpha_i <- fdr * (1:length(pvalue_v)) / length(pvalue_v)
        k <- which(pvalue_v_sort$x < alpha_i)
        prob_mtx_sig[pvalue_v_sort$ix[k]] <- prob_mtx[pvalue_v_sort$ix[k]]

    When *M* is provided, p-values are known to be discrete multiples of
    1/M, so a counting-sort approach replaces the O(n log n) argsort with
    O(n + M) bucket-based rank computation.
    """
    # Fast path: if the smallest non-zero p-value exceeds fdr, only
    # zero-pvalue entries can survive BH correction.  The BH threshold
    # alpha_i = fdr * i / n is at most fdr (at i=n).  Any p > fdr is
    # guaranteed to fail, so only p == 0 passes.  This avoids the O(n
    # log n) argsort for common cases (e.g. M=5 where min non-zero
    # p-value is 0.2 >> 0.05).
    nonzero_mask = pvalue > 0
    if nonzero_mask.any():
        min_nonzero = pvalue[nonzero_mask].min()
        if min_nonzero > fdr:
            net = np.zeros_like(prob_mtx)
            zero_mask = ~nonzero_mask
            if zero_mask.any():
                net[zero_mask] = prob_mtx[zero_mask]
            return net

    # Flatten column-major to match R's c() which is column-major
    pvalue_v = pvalue.ravel(order="F")
    n = len(pvalue_v)

    if M is not None and M > 0:
        # Counting-sort BH: O(n + M) instead of O(n log n).
        # P-values are discrete at multiples of 1/M, so we use
        # bucket-based rank computation instead of argsort.
        levels = np.rint(pvalue_v * M).astype(np.int32)
        counts = np.bincount(levels, minlength=M + 1)
        cum = np.cumsum(counts)
        prev_cum = np.empty(M + 1, dtype=np.int64)
        prev_cum[0] = 0
        prev_cum[1:] = cum[:-1]

        # For each level l, element at rank j passes if l/M < fdr*j/n,
        # i.e. j > l*n/(M*fdr).  Vectorized across all levels:
        level_idx = np.arange(M + 1, dtype=np.float64)
        floor_thresholds = np.floor(level_idx * n / (M * fdr)).astype(np.int64)
        n_pass = np.maximum(0, cum - np.maximum(prev_cum, floor_thresholds))
        n_pass[0] = counts[0]  # p=0 always passes

        # Classify levels: fully passing, fully failing, or partial
        fully_passing = (n_pass == counts) & (counts > 0)

        # Build pass mask via O(n) lookup for fully-passing levels
        level_passes = np.zeros(M + 1, dtype=bool)
        level_passes[fully_passing] = True
        pass_mask = level_passes[levels]

        # Handle partial levels (typically 0–1 boundary levels)
        partial = ~fully_passing & (n_pass > 0)
        if partial.any():
            for lvl in np.where(partial)[0]:
                level_indices = np.where(levels == lvl)[0]
                # Last n_pass[lvl] elements = highest ranks in stable sort
                pass_mask[level_indices[-int(n_pass[lvl]):]] = True

        net = np.zeros_like(prob_mtx)
        if pass_mask.any():
            flat_prob = prob_mtx.ravel(order="F")
            flat_net = net.ravel(order="F")
            flat_net[pass_mask] = flat_prob[pass_mask]
            net = flat_net.reshape(prob_mtx.shape, order="F")
        return net

    # Fallback: standard argsort-based BH (when M is unknown)
    sort_idx = np.argsort(pvalue_v, kind="stable")
    pvalue_sorted = pvalue_v[sort_idx]

    alpha_i = fdr * np.arange(1, n + 1) / n

    passes = pvalue_sorted < alpha_i
    kept_sorted_positions = np.where(passes)[0]

    net = np.zeros_like(prob_mtx)
    if len(kept_sorted_positions) > 0:
        original_indices = sort_idx[kept_sorted_positions]
        flat_prob = prob_mtx.ravel(order="F")
        flat_net = net.ravel(order="F")
        flat_net[original_indices] = flat_prob[original_indices]
        # Reshape back (ravel with order='F' returns a copy, so write back)
        net = flat_net.reshape(prob_mtx.shape, order="F")

    return net
