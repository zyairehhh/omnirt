from pathlib import Path

import pytest

from omnirt.core.types import WeightFormatError
from omnirt.core.weight_loader import WeightLoader


def test_weight_loader_rejects_non_safetensors(tmp_path: Path) -> None:
    weight_path = tmp_path / "demo.bin"
    weight_path.write_bytes(b"bad")

    with pytest.raises(WeightFormatError):
        WeightLoader.validate_path(str(weight_path))


def test_weight_loader_accepts_safetensors_suffix(tmp_path: Path) -> None:
    weight_path = tmp_path / "demo.safetensors"
    weight_path.write_bytes(b"fake")

    assert WeightLoader.validate_path(str(weight_path)) == weight_path

