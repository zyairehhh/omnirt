"""Weight loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omnirt.core.types import DependencyUnavailableError, WeightFormatError


class WeightLoader:
    """Load model weights from safetensors files."""

    @staticmethod
    def validate_path(path: str) -> Path:
        weight_path = Path(path)
        if weight_path.suffix != ".safetensors":
            raise WeightFormatError(f"Only .safetensors weights are supported, got: {weight_path.name}")
        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        return weight_path

    def load(self, path: str, *, device: str = "cpu") -> Dict[str, Any]:
        weight_path = self.validate_path(path)
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise DependencyUnavailableError("safetensors is required to load model weights.") from exc
        return load_file(str(weight_path), device=device)

