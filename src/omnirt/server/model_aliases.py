"""Model alias resolution for HTTP routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yaml


def load_model_aliases(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Model aliases file not found: {file_path}")
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if isinstance(payload, dict) and "aliases" in payload and isinstance(payload["aliases"], dict):
        payload = payload["aliases"]
    if not isinstance(payload, dict):
        raise ValueError("Model aliases file must define a mapping.")
    return {str(key): str(value) for key, value in payload.items()}


def resolve_model_alias(model: str, aliases: Dict[str, str]) -> str:
    return aliases.get(model, model)
