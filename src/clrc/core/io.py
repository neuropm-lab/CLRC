"""IO utilities: pickle, JSON, YAML config, alignment data loading."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import yaml

from clrc.core.types import AlignmentData

logger = logging.getLogger(__name__)


# Required keys for every entry in AlignmentData.meta. Values may be None for
# non-CCI features (e.g. fiber_distance), but every key must be present so
# downstream code can use direct subscript access instead of .get() fallbacks.
_REQUIRED_META_KEYS = frozenset(
    ("feature_name", "lr_name", "ct_L", "ct_R", "lr_index")
)


def _validate_feature_meta(meta: List[Any], expected_n: int) -> None:
    """Validate that ``meta`` is a list of dicts conforming to FeatureMeta.

    Raises
    ------
    ValueError
        If the list length doesn't match ``expected_n``, or any entry is not a
        dict, or any entry is missing a required key from ``_REQUIRED_META_KEYS``.
    """
    if len(meta) != expected_n:
        raise ValueError(
            f"AlignmentData.meta has {len(meta)} entries but X has "
            f"{expected_n} columns — they must be aligned row-for-row."
        )
    for i, m in enumerate(meta):
        if not isinstance(m, dict):
            raise ValueError(
                f"AlignmentData.meta[{i}] is {type(m).__name__}, expected dict. "
                f"All meta entries must be FeatureMeta-conforming dicts."
            )
        missing = _REQUIRED_META_KEYS - set(m.keys())
        if missing:
            raise ValueError(
                f"AlignmentData.meta[{i}] is missing required FeatureMeta keys: "
                f"{sorted(missing)}. Entry keys present: {sorted(m.keys())}."
            )


# ---------------------------------------------------------------------------
#  Filesystem helpers
# ---------------------------------------------------------------------------

def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Walk up from *start* to find the directory containing ``.git/``."""
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Could not find repo root (.git/)")


def safe_filename(s: str) -> str:
    """Make a filesystem-safe token from region/metric names."""
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"__+", "_", s).strip("_")
    return s or "x"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_hash_int(s: str, *, mod: int = 2**63 - 1) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h, 16) % mod


# ---------------------------------------------------------------------------
#  Serialization
# ---------------------------------------------------------------------------

class _CompatUnpickler(pickle.Unpickler):
    """Redirect old FoldArtifact pickle references to clrc.core.types."""

    def find_class(self, module: str, name: str):
        if name == "FoldArtifact":
            from clrc.core.types import FoldArtifact
            return FoldArtifact
        return super().find_class(module, name)


def load_pickle(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("rb") as f:
        return _CompatUnpickler(f).load()


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
#  YAML config
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _load_paths_env(repo_root: Path) -> Dict[str, str]:
    """Load ``paths.env`` from repo root into a dict. Missing file → empty dict.

    Format: ``KEY=value`` per line; ``#`` comments and blank lines ignored.
    Lines with unquoted values preserve the raw string (no shell escaping).
    """
    env_path = repo_root / "paths.env"
    if not env_path.is_file():
        return {}
    entries: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        entries[key.strip()] = val.strip().strip('"').strip("'")
    return entries


def _expand_env_vars(obj: Any, env: Dict[str, str]) -> Any:
    """Recursively expand ``${VAR}`` tokens in strings of a nested structure.

    Uses ``env`` first, then ``os.environ``. Missing variables raise KeyError
    with a clear message so misconfiguration is loud, not silent.
    """
    if isinstance(obj, str):
        def _sub(match: "re.Match[str]") -> str:
            var = match.group(1)
            if var in env:
                return env[var]
            import os
            if var in os.environ:
                return os.environ[var]
            raise KeyError(
                f"Config references ${{{var}}} but it is not set in "
                f"paths.env or os.environ. Copy paths.env.example and edit."
            )
        return _ENV_VAR_RE.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v, env) for v in obj]
    return obj


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config, expand env vars, resolve paths relative to repo root.

    Env vars are sourced from ``<repo_root>/paths.env`` (gitignored) with
    ``os.environ`` as fallback. References use ``${VAR_NAME}`` syntax.
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = find_repo_root(config_path.parent)

    env = _load_paths_env(repo_root)
    cfg = _expand_env_vars(cfg, env)

    # Resolve paths in 'data' section
    if "data" in cfg:
        for key, val in cfg["data"].items():
            if isinstance(val, str) and not Path(val).is_absolute():
                cfg["data"][key] = str(repo_root / val)

    # Resolve output base_dir
    if "output" in cfg and "base_dir" in cfg["output"]:
        val = cfg["output"]["base_dir"]
        if isinstance(val, str) and not Path(val).is_absolute():
            cfg["output"]["base_dir"] = str(repo_root / val)

    # Resolve paths in spatial_null section (except atlas_nii, already absolute
    # after env expansion from paths.env)
    if "spatial_null" in cfg:
        for key, val in cfg["spatial_null"].items():
            if isinstance(val, str) and not Path(val).is_absolute():
                cfg["spatial_null"][key] = str(repo_root / val)

    # Resolve paths in abc_region_data section (shared ABC <-> Allen mapping files).
    if "abc_region_data" in cfg:
        for key, val in cfg["abc_region_data"].items():
            if isinstance(val, str) and not Path(val).is_absolute():
                cfg["abc_region_data"][key] = str(repo_root / val)

    return cfg


# ---------------------------------------------------------------------------
#  Alignment data loading
# ---------------------------------------------------------------------------

def load_alignment_data(
    pkl_path: Union[str, Path],
    *,
    version: str = "voxel",
    target_scale: float = 1.0,
) -> AlignmentData:
    """Load pre-aligned CCI features + connectivity targets from pickle."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_path}")

    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in pickle, got: {type(obj)}")

    # --- edge_table ---
    edge_table = obj.get("edge_table", None)
    if not isinstance(edge_table, pd.DataFrame):
        edge_table = pd.DataFrame(edge_table)

    X = obj.get("X_kept_np", None)
    if X is None:
        X = obj.get("X", None)
    if X is None:
        raise KeyError("pickle missing 'X_kept_np' (or fallback 'X')")
    X = np.asarray(X)

    metric_names = obj.get("metric_names", None)
    if metric_names is None:
        raise KeyError("pickle missing 'metric_names'")
    metric_names = list(metric_names)

    SC_naive = obj.get("SC_naive_mat", None)
    if SC_naive is None:
        SC_naive = obj.get("SC_naive", None)
    if SC_naive is None:
        raise KeyError("pickle missing 'SC_naive_mat' (or fallback 'SC_naive')")
    SC_naive = np.asarray(SC_naive)

    SC_voxel = obj.get("SC_voxel_mat", None)
    if SC_voxel is None:
        SC_voxel = obj.get("SC_voxel", None)
    if SC_voxel is None:
        raise KeyError("pickle missing 'SC_voxel_mat' (or fallback 'SC_voxel')")
    SC_voxel = np.asarray(SC_voxel)

    n_edges = len(edge_table)
    if X.shape[0] != n_edges:
        raise ValueError(
            f"X has {X.shape[0]} rows but edge_table has {n_edges} rows."
        )
    if SC_naive.shape != (n_edges, len(metric_names)):
        raise ValueError(
            f"SC_naive shape {SC_naive.shape} != {(n_edges, len(metric_names))}"
        )
    if SC_voxel.shape != (n_edges, len(metric_names)):
        raise ValueError(
            f"SC_voxel shape {SC_voxel.shape} != {(n_edges, len(metric_names))}"
        )

    # Scale target matrices (e.g., FC values are small and benefit from 1e2 scaling)
    if target_scale != 1.0:
        SC_naive = SC_naive * target_scale
        SC_voxel = SC_voxel * target_scale
        logger.info("Applied target_scale=%.1f to SC matrices.", target_scale)

    # Load distance vector (use version-matched)
    distance_key = (
        "distance_vec_voxel" if version == "voxel" else "distance_vec_naive"
    )
    distance_vec = obj.get(distance_key, None)
    if distance_vec is not None:
        distance_vec = np.asarray(distance_vec).ravel()
        logger.info(
            "Loaded distance feature (%s), shape=%s", distance_key, distance_vec.shape
        )
    else:
        logger.info("No distance feature found in pickle.")

    # Feature names (real LR+CT names, not synthetic cci_0, cci_1, ...)
    feature_names = obj.get("feature_names_kept", None)
    if feature_names is not None:
        feature_names = list(feature_names)
        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"feature_names_kept has {len(feature_names)} entries "
                f"but X has {X.shape[1]} columns."
            )
        logger.info("Loaded %d feature names.", len(feature_names))

    # Per-feature metadata: list of dicts with ct_L, ct_R, lr_name, lr_index, feature_name
    meta = obj.get("meta_ABC_kept", None)
    if meta is None:
        meta = obj.get("meta", None)
    if meta is not None:
        meta = list(meta)
        _validate_feature_meta(meta, expected_n=X.shape[1])
        logger.info("Loaded %d feature meta entries.", len(meta))

    return AlignmentData(
        edge_table=edge_table,
        X=X,
        metric_names=metric_names,
        SC_naive=SC_naive,
        SC_voxel=SC_voxel,
        feature_names=feature_names,
        meta=meta,
        distance_vec=distance_vec,
    )
