"""Loss metrics and fold-level aggregation."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def pseudohuber(
    y_true: np.ndarray, y_pred: np.ndarray, *, delta: float = 1.0
) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return float("nan")
    delta = float(delta)
    residual = y_pred - y_true
    return float(
        np.mean((np.sqrt(1.0 + (residual / delta) ** 2) - 1.0) * (delta**2))
    )


def metric_value_for_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    loss: str,
    huber_slope: Optional[float],
) -> float:
    if loss == "rmse":
        return rmse(y_true, y_pred)
    if loss == "mae":
        return mae(y_true, y_pred)
    if loss == "pseudohuber":
        slope = 1.0 if huber_slope is None else float(huber_slope)
        return pseudohuber(y_true, y_pred, delta=slope)
    raise ValueError(f"Unsupported loss for metric aggregation: {loss}")


def aggregate_fold_metrics_for_loss(
    fold_artifacts: Sequence,
    *,
    loss: str,
    huber_slope: Optional[float],
) -> Tuple[float, float, List[float]]:
    """Compute micro/macro aggregate of a loss metric across LOBO folds.

    Parameters
    ----------
    fold_artifacts : sequence of FoldArtifact
        Each must have .y_test_t, .y_pred, .n_test attributes.
    """
    fold_metrics: List[float] = []
    fold_ns: List[int] = []
    for fa in fold_artifacts:
        fold_metrics.append(
            metric_value_for_loss(
                fa.y_test_t, fa.y_pred, loss=loss, huber_slope=huber_slope
            )
        )
        fold_ns.append(int(fa.n_test))

    if len(fold_metrics) == 0:
        return float("nan"), float("nan"), []

    n_tot = float(np.sum(fold_ns))
    if n_tot == 0:
        return float("nan"), float("nan"), fold_metrics

    macro_metric = float(np.mean(fold_metrics))
    if loss == "rmse":
        micro_metric = float(
            np.sqrt(
                np.sum(
                    [n * (m**2) for n, m in zip(fold_ns, fold_metrics)]
                )
                / n_tot
            )
        )
    else:
        micro_metric = float(
            np.sum([n * m for n, m in zip(fold_ns, fold_metrics)]) / n_tot
        )
    return micro_metric, macro_metric, fold_metrics
