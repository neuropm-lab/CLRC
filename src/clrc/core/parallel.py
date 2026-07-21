"""Parallelism helpers shared across pipeline modules.

Currently hosts :func:`tqdm_joblib`, a context manager that drives a tqdm
progress bar from joblib's batch-completion callback. Use it to wrap a
``joblib.Parallel(...)`` call when you want a real progress bar with
ETA instead of joblib's built-in ``verbose=N`` prints.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from tqdm.auto import tqdm


@contextmanager
def tqdm_joblib(tqdm_bar: "tqdm") -> Iterator["tqdm"]:
    """Patch joblib.Parallel so it advances ``tqdm_bar`` on batch completion.

    Usage::

        from joblib import Parallel, delayed
        from tqdm.auto import tqdm
        from clrc.core.parallel import tqdm_joblib

        with tqdm_joblib(tqdm(total=len(tasks), desc="my loop")) as bar:
            results = Parallel(n_jobs=16)(
                delayed(fn)(x) for x in tasks
            )

    The bar is advanced by ``batch_size`` per completed batch (joblib's
    default batching groups multiple tasks per dispatched unit of work,
    so this gives the correct per-task accounting).

    Parameters
    ----------
    tqdm_bar
        A tqdm instance with ``total`` set to the number of tasks.

    Yields
    ------
    The same tqdm instance (convenience for ``as`` binding).

    Notes
    -----
    joblib's private API (``joblib.parallel.BatchCompletionCallBack``) is
    monkey-patched for the duration of the context. The original class
    is restored on exit even if the parallel block raises, so this helper
    is safe to use in concurrent code paths.
    """
    import joblib.parallel

    old_cb = joblib.parallel.BatchCompletionCallBack

    class _TqdmCallback(old_cb):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = _TqdmCallback  # ty: ignore[invalid-assignment]
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb


__all__ = ["tqdm_joblib"]
