"""Weight loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from omnirt.core.types import DependencyUnavailableError, WeightFormatError


@dataclass(frozen=True)
class HuggingFaceFileRef:
    repo_id: str
    filename: str
    revision: str | None = None


class WeightLoader:
    """Load model weights from safetensors files."""

    @staticmethod
    def _validate_safetensors_name(name: str) -> None:
        if not name.endswith(".safetensors"):
            raise WeightFormatError(f"Only .safetensors weights are supported, got: {name}")

    @classmethod
    def _parse_hf_scheme_ref(cls, path: str) -> HuggingFaceFileRef:
        parsed = urlparse(path)
        path_parts = [segment for segment in parsed.path.split("/") if segment]
        if not parsed.netloc or len(path_parts) < 2:
            raise WeightFormatError(
                "HF weight references must use the form hf://<repo-id>/<path/to/file.safetensors>"
            )
        repo_id = f"{parsed.netloc}/{path_parts[0]}"
        filename = "/".join(path_parts[1:])
        cls._validate_safetensors_name(filename)
        revision = parse_qs(parsed.query).get("revision", [None])[0]
        return HuggingFaceFileRef(repo_id=unquote(repo_id), filename=unquote(filename), revision=unquote(revision) if revision else None)

    @classmethod
    def _parse_hf_resolve_url(cls, path: str) -> HuggingFaceFileRef | None:
        parsed = urlparse(path)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != "huggingface.co":
            return None

        parts = [segment for segment in parsed.path.split("/") if segment]
        if "resolve" not in parts:
            return None

        resolve_index = parts.index("resolve")
        if resolve_index < 1 or len(parts) <= resolve_index + 2:
            raise WeightFormatError(
                "Hugging Face resolve URLs must look like https://huggingface.co/<repo-id>/resolve/<revision>/<file>"
            )

        repo_id = "/".join(parts[:resolve_index])
        revision = unquote(parts[resolve_index + 1])
        filename = unquote("/".join(parts[resolve_index + 2 :]))
        cls._validate_safetensors_name(filename)
        return HuggingFaceFileRef(repo_id=repo_id, filename=filename, revision=revision)

    @classmethod
    def _download_hf_file(cls, ref: HuggingFaceFileRef) -> Path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise DependencyUnavailableError(
                "huggingface_hub is required to download weights from Hugging Face."
            ) from exc

        return Path(
            hf_hub_download(
                repo_id=ref.repo_id,
                filename=ref.filename,
                revision=ref.revision,
            )
        )

    @classmethod
    def validate_path(cls, path: str) -> Path:
        if path.startswith("hf://"):
            return cls._download_hf_file(cls._parse_hf_scheme_ref(path))

        resolve_ref = cls._parse_hf_resolve_url(path)
        if resolve_ref is not None:
            return cls._download_hf_file(resolve_ref)

        weight_path = Path(path)
        cls._validate_safetensors_name(weight_path.name)
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
