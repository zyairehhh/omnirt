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


def test_weight_loader_downloads_hf_scheme_ref(monkeypatch, tmp_path: Path) -> None:
    cached = tmp_path / "hf-cache.safetensors"
    cached.write_bytes(b"fake")
    recorded = {}

    def fake_download(ref):
        recorded["repo_id"] = ref.repo_id
        recorded["filename"] = ref.filename
        recorded["revision"] = ref.revision
        return cached

    monkeypatch.setattr(WeightLoader, "_download_hf_file", fake_download)

    resolved = WeightLoader.validate_path("hf://acme/demo-adapter/weights/style.safetensors?revision=main")

    assert resolved == cached
    assert recorded == {
        "repo_id": "acme/demo-adapter",
        "filename": "weights/style.safetensors",
        "revision": "main",
    }


def test_weight_loader_downloads_hf_resolve_url(monkeypatch, tmp_path: Path) -> None:
    cached = tmp_path / "hf-cache.safetensors"
    cached.write_bytes(b"fake")
    recorded = {}

    def fake_download(ref):
        recorded["repo_id"] = ref.repo_id
        recorded["filename"] = ref.filename
        recorded["revision"] = ref.revision
        return cached

    monkeypatch.setattr(WeightLoader, "_download_hf_file", fake_download)

    resolved = WeightLoader.validate_path(
        "https://huggingface.co/acme/demo-adapter/resolve/refs%2Fpr%2F7/weights/style.safetensors"
    )

    assert resolved == cached
    assert recorded == {
        "repo_id": "acme/demo-adapter",
        "filename": "weights/style.safetensors",
        "revision": "refs/pr/7",
    }


def test_weight_loader_rejects_invalid_hf_ref() -> None:
    with pytest.raises(WeightFormatError):
        WeightLoader.validate_path("hf://acme/demo-adapter/weights/style.bin")
