"""clrc — core analysis package for the Cell-type-resolved directed Ligand-Receptor Connectome (CLRC).

The very first import is ``clrc._env`` which sets process-level environment
variables (e.g. ``NUMBA_THREADING_LAYER``) that MUST be in place before any
third-party library reads them. Keeping that import at the top of this file
means ``import clrc`` (or ``from clrc.anything import ...``) triggers the
env setup once, regardless of which submodule is imported first.
"""

from __future__ import annotations

from clrc import _env  # noqa: F401  — import for side effects (env vars)
