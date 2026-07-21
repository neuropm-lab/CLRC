"""Null-draw utilities for bias-label cross-prediction sanity checks.

Three building blocks:

1. :func:`draw_random_lr_subsets_uniform` — uniform-random LR-pair subsets.
2. :func:`draw_random_lr_subsets_importance_matched` — null draws whose
   summed combined importance matches a target sum within a tolerance.
   Matching everything except the label under test follows the
   defense-in-depth null-design convention in Markello 2021 / Hansen 2022.
3. :func:`feature_mask_for_lr_subset` — lift an LR-pair subset to a
   feature-level boolean mask, consistent with the aligned
   ``(feature_names, meta)`` contract returned by
   :func:`clrc.prediction.lobo.select_features`.

Training is not implemented here: callers pass the resulting feature
mask to :func:`clrc.prediction.xgboost.train_predict_xgb`.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Sequence

import numpy as np

from clrc.core.types import FeatureMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Uniform null
# ---------------------------------------------------------------------------


def draw_random_lr_subsets_uniform(
    all_lr_pairs: Sequence[str],
    k: int,
    n_draws: int,
    seed: int,
) -> List[List[str]]:
    """Draw ``n_draws`` uniform-random LR-pair subsets of size ``k``.

    Each draw is sampled without replacement from ``all_lr_pairs``. Successive
    draws are independent; duplicate draws are allowed (and essentially
    impossible for realistic k/N — they would indicate a pathologically small
    pool and we do not filter them out).

    Parameters
    ----------
    all_lr_pairs
        Pool of LR-pair names to sample from (e.g. all non-zero LR pairs from
        the HPO-best categorization).
    k
        Size of each drawn subset.
    n_draws
        Number of independent subsets to produce.
    seed
        Deterministic seed — same seed → identical list of subsets.

    Returns
    -------
    list of lists of str, length ``n_draws``, each inner list length ``k``.
    """
    all_lr_pairs = list(all_lr_pairs)
    if k > len(all_lr_pairs):
        raise ValueError(
            f"Cannot draw k={k} without replacement from pool of size "
            f"{len(all_lr_pairs)}."
        )
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}.")
    if n_draws <= 0:
        raise ValueError(f"n_draws must be > 0, got {n_draws}.")

    rng = np.random.default_rng(seed)
    draws: List[List[str]] = []
    arr = np.asarray(all_lr_pairs, dtype=object)
    for _ in range(n_draws):
        idx = rng.choice(len(arr), size=k, replace=False)
        draws.append([str(x) for x in arr[idx]])
    return draws


# ---------------------------------------------------------------------------
#  Importance-matched null
# ---------------------------------------------------------------------------


def draw_random_lr_subsets_importance_matched(
    all_lr_pairs: Sequence[str],
    lr_importances: Mapping[str, float],
    target_sum: float,
    k: int,
    n_draws: int,
    seed: int,
    tol: float = 0.10,
    max_attempts_per_draw: int = 20000,
) -> List[List[str]]:
    """Draw ``n_draws`` random subsets whose summed importance matches ``target_sum``.

    **Approach: rejection sampling from uniform proposals.**

    Rationale: for k=150 out of 811 pairs with the observed skewed-positive
    importance distribution on the HPO-best categorization, rejection
    sampling converges quickly (acceptance rate typically >1% at ±5%). This
    is simpler than stratified-quantile construction and avoids any
    distributional assumptions about the importance values. If the
    acceptance rate turns out lower than anticipated we raise rather than
    silently converge on an ill-matched null, because a failure here is
    scientifically informative.

    Parameters
    ----------
    all_lr_pairs
        Pool of LR-pair names (typically the 811 non-zero pairs).
    lr_importances
        Mapping ``lr_name -> combined_importance``. Missing keys → 0.0.
    target_sum
        Target sum of combined importance (e.g. the summed combined
        importance of the 150 SC-biased LR pairs).
    k
        Subset size (should match the target subset it is nulling against).
    n_draws
        How many accepted matched subsets to produce.
    seed
        Deterministic seed.
    tol
        Relative tolerance ``|draw_sum - target_sum| / target_sum <= tol``.
    max_attempts_per_draw
        Upper bound on rejection proposals per accepted draw. If hit, the
        function raises RuntimeError. This guards against misconfigured
        targets (e.g. a target_sum physically unreachable from the pool).

    Returns
    -------
    list of lists of str, length ``n_draws``, each inner list length ``k``.

    Raises
    ------
    RuntimeError
        If ``max_attempts_per_draw`` is exceeded before acceptance —
        indicates an infeasible or near-infeasible target.
    """
    all_lr_pairs = list(all_lr_pairs)
    if k > len(all_lr_pairs):
        raise ValueError(
            f"Cannot draw k={k} without replacement from pool of size "
            f"{len(all_lr_pairs)}."
        )
    if target_sum <= 0:
        raise ValueError(f"target_sum must be > 0, got {target_sum}.")

    # Vectorize importance lookup — avoid a per-proposal dict touch.
    imp_arr = np.asarray(
        [float(lr_importances.get(p, 0.0)) for p in all_lr_pairs], dtype=float
    )
    name_arr = np.asarray(all_lr_pairs, dtype=object)

    rng = np.random.default_rng(seed)
    draws: List[List[str]] = []
    n_pool = len(all_lr_pairs)
    total_attempts = 0

    for draw_i in range(n_draws):
        accepted = False
        attempts = 0
        while attempts < max_attempts_per_draw:
            idx = rng.choice(n_pool, size=k, replace=False)
            s = float(imp_arr[idx].sum())
            attempts += 1
            total_attempts += 1
            if abs(s - target_sum) / target_sum <= tol:
                draws.append([str(x) for x in name_arr[idx]])
                accepted = True
                break
        if not accepted:
            raise RuntimeError(
                f"Importance-matched rejection sampling failed after "
                f"{max_attempts_per_draw} attempts for draw #{draw_i + 1}. "
                f"target_sum={target_sum}, tol={tol}. The target may be "
                f"physically unreachable from this pool — try loosening "
                f"tol or checking that lr_importances covers the pool."
            )

    logger.info(
        "Importance-matched draws: produced %d / %d in %d total attempts "
        "(acceptance rate ≈ %.3f%%).",
        len(draws),
        n_draws,
        total_attempts,
        100.0 * len(draws) / max(total_attempts, 1),
    )
    return draws


# ---------------------------------------------------------------------------
#  Feature-level mask
# ---------------------------------------------------------------------------


def feature_mask_for_lr_subset(
    feature_names: Sequence[str],
    meta: Sequence[FeatureMeta],
    lr_subset: Sequence[str],
) -> np.ndarray:
    """Lift an LR-pair subset to a feature-level boolean mask.

    Each LR pair contributes 31×31 = 961 per-cell-type-pair CCI features in
    the full ABC_expanded feature space (smaller in synthetic/test data).
    This function returns a mask of shape ``(n_features,)`` that is ``True``
    for every feature whose ``meta[i]["lr_name"]`` is in ``lr_subset``, and
    ``False`` otherwise. Non-CCI features (lr_name=None, e.g. the synthetic
    fiber_distance entry) are always ``False``.

    Parameters
    ----------
    feature_names, meta
        The aligned triple-partners from
        :func:`clrc.prediction.lobo.select_features`. ``len(meta) == n_features``
        and ``meta[i]["feature_name"] == feature_names[i]``.
    lr_subset
        LR-pair names to keep. Every name MUST appear in ``meta`` as some
        ``lr_name`` — otherwise we raise, because a silent miss here would
        let a misconfigured subset sneak through as "zero features selected".

    Returns
    -------
    np.ndarray of dtype bool, shape ``(n_features,)``.

    Raises
    ------
    ValueError
        If ``lr_subset`` contains a name that does not appear in ``meta``.
    """
    if len(feature_names) != len(meta):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != meta length "
            f"({len(meta)}). Use clrc.prediction.lobo.select_features which "
            f"returns an aligned (X, feature_names, meta) triple."
        )

    lr_set = set(lr_subset)
    # Inventory which LR names exist in meta (skip None = non-CCI features).
    available_lrs = {m["lr_name"] for m in meta if m["lr_name"] is not None}
    missing = lr_set - available_lrs
    if missing:
        raise ValueError(
            f"lr_subset contains {len(missing)} name(s) not present in meta: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}. "
            f"Check that lr_subset was built from the same categorization "
            f"artifact as the alignment pickle's meta."
        )

    mask = np.array(
        [m["lr_name"] in lr_set for m in meta],
        dtype=bool,
    )
    return mask
