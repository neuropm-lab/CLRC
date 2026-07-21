"""Network aggregation across ligand-target pairs."""

from __future__ import annotations

import numpy as np

_VALID_METHODS = ("weight", "count", "weighted_count", "weighted_count2", "weight_threshold")


def net_aggregation(
    net: dict[str, np.ndarray],
    method: str = "weight",
    cut_off: float = 0.05,
) -> np.ndarray:
    """Aggregate communication networks over all ligand-target pairs.

    Parameters
    ----------
    net : dict[str, np.ndarray]
        Mapping of interaction name -> communication strength matrix.
    method : str
        One of 'weight', 'count', 'weighted_count', 'weighted_count2', 'weight_threshold'.
    cut_off : float
        Quantile threshold for 'weighted_count2' and 'weight_threshold'.

    Returns
    -------
    Aggregated network matrix.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {_VALID_METHODS}")

    if not net:
        raise ValueError("net dict is empty — no interactions to aggregate")

    net_list = list(net.values())
    result = np.zeros_like(net_list[0])

    if method == "weight":
        for n in net_list:
            result += n

    elif method == "count":
        for n in net_list:
            result += (n > 0).astype(result.dtype)

    elif method == "weighted_count":
        for n in net_list:
            result += n.sum() * (n > 0).astype(result.dtype)

    elif method == "weighted_count2":
        for n in net_list:
            thresh = np.quantile(n, cut_off)
            above = (n > thresh).astype(result.dtype)
            count_above = above.sum()
            result += n.sum() / (1e-6 + count_above) * above

    elif method == "weight_threshold":
        for n in net_list:
            thresh = np.quantile(n, cut_off)
            result += n * (n > thresh).astype(result.dtype)

    return result
