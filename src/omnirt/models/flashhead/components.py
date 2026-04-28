"""Deployment metadata for SoulX-FlashHead."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


ENV_PREFIX = "OMNIRT_FLASHHEAD_"

_ENV_KEYS = {
    "repo_path": "REPO_PATH",
    "ckpt_dir": "CKPT_DIR",
    "wav2vec_dir": "WAV2VEC_DIR",
    "ascend_env_script": "ASCEND_ENV_SCRIPT",
    "python_executable": "PYTHON",
}

_PROJECT_CONFIG_RELATIVE = Path("configs") / "flashhead.yaml"
_USER_CONFIG_RELATIVE = Path(".omnirt") / "flashhead.yaml"


class FlashHeadConfigurationError(RuntimeError):
    """Raised when a required FlashHead deployment setting is missing."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when pyyaml missing.
        raise FlashHeadConfigurationError(
            f"Reading {path} requires PyYAML. Install it or set {ENV_PREFIX}* env vars instead."
        ) from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise FlashHeadConfigurationError(f"{path} must contain a YAML mapping at the top level.")
    return data


@lru_cache(maxsize=1)
def _yaml_settings() -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(_read_yaml(_project_root() / _PROJECT_CONFIG_RELATIVE))
    merged.update(_read_yaml(Path.home() / _USER_CONFIG_RELATIVE))
    return merged


def reset_config_cache() -> None:
    """Invalidate the YAML cache. Tests use this after patching env/yaml."""
    _yaml_settings.cache_clear()


def flashhead_setting(key: str, *, required: bool = False) -> Optional[str]:
    """Resolve a FlashHead deployment setting.

    Lookup order: ``OMNIRT_FLASHHEAD_<KEY>`` env var -> ``configs/flashhead.yaml``
    -> ``~/.omnirt/flashhead.yaml``.
    """
    env_key = _ENV_KEYS.get(key)
    if env_key is None:
        raise KeyError(f"Unknown FlashHead setting: {key!r}")
    env_value = os.environ.get(ENV_PREFIX + env_key)
    if env_value and env_value.strip():
        return env_value.strip()
    yaml_value = _yaml_settings().get(key)
    if isinstance(yaml_value, str) and yaml_value.strip():
        return yaml_value.strip()
    if required:
        raise FlashHeadConfigurationError(
            f"FlashHead setting {key!r} is not configured. "
            f"Set the {ENV_PREFIX + env_key} environment variable or add "
            f"'{key}' to configs/flashhead.yaml or ~/.omnirt/flashhead.yaml."
        )
    return None
