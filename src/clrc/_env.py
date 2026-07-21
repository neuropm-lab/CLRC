"""Process-level environment setup that MUST run before third-party imports.

This module's only job is to set environment variables that have to be in
place before certain third-party libraries are imported. It is imported by
``clrc/__init__.py`` before any other clrc submodule so that any downstream
``import numba`` (direct or transitive) sees the right configuration.

Kept as a standalone module (rather than folded into ``clrc.core.logging``)
because:

* ``clrc.core.logging`` has a clear single responsibility — configuring
  Python logging handlers. Mixing environment mutation into it would
  entangle two unrelated concerns.
* Importing ``clrc.core.logging`` is relatively heavy (pulls optuna), and
  env-var setup needs to happen as early and as cheaply as possible.
* A tiny, well-named module makes it obvious to future readers *why* this
  ``os.environ`` mutation exists and that it needs to be imported first.
"""

from __future__ import annotations

import os

# numba 0.59+ deprecates TBB as the default threading layer and emits a
# DeprecationWarning at import time. Set NUMBA_THREADING_LAYER=omp (OpenMP)
# before numba is imported to silence the warning and pin to a stable layer.
# MUST be set before any `import numba` in the process; os.environ is the
# only way numba can see this configuration.
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
