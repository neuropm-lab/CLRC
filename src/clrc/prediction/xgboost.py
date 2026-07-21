"""XGBoost training wrapper (single-stage early stopping)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import xgboost as xgb

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None  # ty: ignore[invalid-assignment]
    _HAS_CUPY = False

logger = logging.getLogger(__name__)


def _to_device_array(
    x: Union[np.ndarray, Any],  # Any covers cupy.ndarray when CuPy is installed
    *,
    device: Literal["cpu", "cuda"],
    dtype: Optional["np.typing.DTypeLike"] = None,
):
    """Move array to CPU or CUDA (via CuPy)."""
    if device == "cpu":
        arr = np.asarray(x)
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr
    if not _HAS_CUPY:
        raise RuntimeError("device='cuda' requested but CuPy is not available.")
    assert cp is not None
    arr = cp.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def train_predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    params: Dict,
    num_boost_round: int,
    split_seed: int,
    booster_seed: Optional[int] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    valid_fraction: float = 0.15,
    early_stopping_rounds: int = 300,
) -> Tuple[np.ndarray, int, bytes]:
    """Single-stage XGBoost training with early stopping.

    Splits training data into subtrain/valid for early stopping, trains
    a single model, and uses that early-stopped model directly for prediction.

    Returns (y_pred, best_iteration, model_raw_bytes).
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_test = np.asarray(X_test)

    n = X_train.shape[0]
    n_valid = max(1, int(round(n * valid_fraction)))

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    valid_idx = perm[:n_valid]
    sub_idx = perm[n_valid:]

    X_sub = X_train[sub_idx]
    y_sub = y_train[sub_idx]
    X_val = X_train[valid_idx]
    y_val = y_train[valid_idx]

    # Build params
    p = dict(params)
    p.setdefault("objective", "reg:squarederror")
    eval_metric = p.get("eval_metric", "rmse")
    if isinstance(eval_metric, (list, tuple)):
        p["eval_metric"] = list(eval_metric)
    else:
        p["eval_metric"] = [eval_metric]
    seed_use = split_seed if booster_seed is None else booster_seed
    p.setdefault("seed", int(seed_use))
    p.setdefault("tree_method", "hist")
    p.setdefault("device", "cuda" if (device == "cuda") else "cpu")
    max_bin = int(p.get("max_bin", 256))
    p["max_bin"] = max_bin

    use_cuda = (device == "cuda") and _HAS_CUPY
    if use_cuda:
        X_sub_d = _to_device_array(X_sub, device="cuda", dtype=np.float32)
        y_sub_d = _to_device_array(y_sub, device="cuda", dtype=np.float32)
        X_val_d = _to_device_array(X_val, device="cuda", dtype=np.float32)
        y_val_d = _to_device_array(y_val, device="cuda", dtype=np.float32)
        X_test_d = _to_device_array(X_test, device="cuda", dtype=np.float32)
        dtrain = xgb.QuantileDMatrix(X_sub_d, y_sub_d, max_bin=max_bin)
        dvalid = xgb.QuantileDMatrix(X_val_d, y_val_d, ref=dtrain, max_bin=max_bin)
        dtest = xgb.QuantileDMatrix(X_test_d, ref=dtrain, max_bin=max_bin)
    else:
        dtrain = xgb.DMatrix(X_sub, label=y_sub)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

    booster = xgb.train(
        params=p,
        dtrain=dtrain,
        num_boost_round=int(num_boost_round),
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=int(early_stopping_rounds),
        verbose_eval=False,
    )
    best_iter = int(getattr(booster, "best_iteration", num_boost_round - 1))
    best_iter = max(0, best_iter)

    y_pred = booster.predict(dtest)

    if use_cuda and _HAS_CUPY:
        assert cp is not None
        y_pred = cp.asnumpy(y_pred)
    model_raw = bytes(booster.save_raw())
    return np.asarray(y_pred, dtype=float).ravel(), best_iter, model_raw
