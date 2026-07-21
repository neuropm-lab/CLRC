"""Target variable transforms (ECDF, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ECDFTransform:
    """Empirical CDF transform fitted on training data."""

    y_sorted: np.ndarray

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        ranks = np.searchsorted(self.y_sorted, y, side="right")
        denom = float(self.y_sorted.size)
        out = ranks / denom
        return out.astype(np.float32, copy=False)


def fit_ecdf(y_train: np.ndarray) -> ECDFTransform:
    """Fit an ECDF on finite training values."""
    y1 = np.asarray(y_train, dtype=float).ravel()
    y1 = y1[np.isfinite(y1)]
    return ECDFTransform(y_sorted=np.sort(y1))
