"""Logging setup: file + console handlers attached at the package level.

Handlers are attached to a *parent* logger (by default ``clrc``) so that every
child logger created by modules via ``logging.getLogger(__name__)`` — e.g.
``clrc.prediction.hpo``, ``clrc.prediction.lobo``, ``clrc.core.io`` — inherits
them through Python's logging hierarchy. Optuna's own logger is also routed
through the same handlers via ``enable_propagation`` so trial summaries appear
in both the console and the run's ``.log`` file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Formatter reused by console + file handlers and tagged so we can detect
# handlers we previously installed (for idempotent setup_logging calls).
_LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s — %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_HANDLER_TAG = "_clrc_setup_logging_handler"


def _make_formatter() -> logging.Formatter:
    return logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)


def _tag_handler(handler: logging.Handler) -> logging.Handler:
    """Mark a handler as installed by ``setup_logging`` for idempotency."""
    setattr(handler, _HANDLER_TAG, True)
    return handler


def _has_tagged_handler(
    logger: logging.Logger, handler_cls: type
) -> bool:
    for h in logger.handlers:
        if isinstance(h, handler_cls) and getattr(h, _HANDLER_TAG, False):
            return True
    return False


def _configure_optuna_propagation(
    parent: logging.Logger, level: int
) -> None:
    """Route Optuna's internal logger through our parent logger's handlers.

    Optuna installs its own stderr handler by default so
    ``[I ...] Trial N finished`` lines bypass every clrc handler. We:

    1. Disable Optuna's built-in stderr handler.
    2. Enable propagation on Optuna's logger (so it does not swallow records).
    3. Attach the *same* handler instances we installed on ``parent`` onto
       Optuna's top-level logger. Python's logger hierarchy is name-based —
       ``optuna`` is NOT a child of ``clrc``, so propagation alone would
       only reach the root logger. Mirroring the handlers onto ``optuna``
       is the simplest way to ensure Optuna records hit the exact same
       console + file sinks without muddying the root logger.

    Guarded with ``try/except ImportError`` so the logging module does not
    hard-require optuna as a dependency.
    """
    try:
        from optuna import logging as optuna_logging
    except ImportError:
        return

    optuna_logging.disable_default_handler()
    optuna_logging.enable_propagation()
    optuna_logger = optuna_logging.get_logger("optuna")
    optuna_logger.setLevel(level)

    # Mirror clrc handlers onto the optuna logger so its records flow through
    # the same formatter + file sink. Only copy handlers we installed (tagged),
    # and skip ones already present on the optuna logger to stay idempotent.
    for h in parent.handlers:
        if not getattr(h, _HANDLER_TAG, False):
            continue
        if h in optuna_logger.handlers:
            continue
        optuna_logger.addHandler(h)


def setup_logging(
    name: str,
    output_dir: Optional[Path] = None,
    *,
    level: int = logging.INFO,
    parent_logger_name: str = "clrc",
) -> logging.Logger:
    """Configure console + optional file handlers on a package-level logger.

    Handlers are attached to ``parent_logger_name`` (default ``"clrc"``) so
    that any child logger created via ``logging.getLogger("clrc.<subpkg>")``
    inherits them and emits records through the same sinks.

    Parameters
    ----------
    name : str
        Run identifier. Used as the stem of the log file
        (``{output_dir}/{name}.log``) so each pipeline script produces its
        own named log file. Not used as the logger name.
    output_dir : Path, optional
        If provided, a ``{name}.log`` file handler is added in this directory.
        The directory is created if missing.
    level : int
        Logging level for the parent logger and both handlers.
    parent_logger_name : str
        Logger on which handlers are installed. Default ``"clrc"``. Children
        of this logger (``clrc.prediction.hpo``, ``clrc.core.io``, ...) will
        inherit these handlers automatically.

    Returns
    -------
    logging.Logger
        The configured parent logger. Callers may log through it directly;
        messages will propagate to the installed console and file handlers.
    """
    parent = logging.getLogger(parent_logger_name)
    parent.setLevel(level)

    formatter = _make_formatter()

    # Console handler — install once per parent logger.
    if not _has_tagged_handler(parent, logging.StreamHandler):
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        parent.addHandler(_tag_handler(ch))

    # File handler — install once per (parent, file path). We look for any
    # tagged FileHandler already pointing at the target path so that repeat
    # calls with the same (name, output_dir) are a no-op.
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{name}.log"

        already_installed = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, _HANDLER_TAG, False)
            and Path(getattr(h, "baseFilename", "")) == log_path.resolve()
            for h in parent.handlers
        )
        if not already_installed:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            parent.addHandler(_tag_handler(fh))

    # Route Optuna's internal logger through these handlers as well.
    _configure_optuna_propagation(parent, level)

    return parent
