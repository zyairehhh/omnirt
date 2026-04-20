"""Adapter loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from omnirt.core.types import AdapterRef, DependencyUnavailableError
from omnirt.core.weight_loader import WeightLoader


@dataclass
class LoadedAdapter:
    ref: AdapterRef
    path: str


class AdapterManager:
    """Validate adapters at pipeline initialization time and apply them once at runtime."""

    def __init__(self) -> None:
        self.loaded: List[LoadedAdapter] = []

    def load_lora(self, ref: AdapterRef, *, device: str = "cpu") -> LoadedAdapter:
        del device
        validated_path = WeightLoader.validate_path(ref.path)
        adapter = LoadedAdapter(ref=ref, path=str(validated_path))
        self.loaded.append(adapter)
        return adapter

    def load_all(self, adapters: Iterable[AdapterRef], *, device: str = "cpu") -> List[LoadedAdapter]:
        return [self.load_lora(adapter, device=device) for adapter in adapters]

    def apply_to_pipeline(self, pipeline: object) -> None:
        if not self.loaded:
            return
        if not hasattr(pipeline, "load_lora_weights"):
            raise DependencyUnavailableError("Current pipeline does not support LoRA loading.")

        for adapter in self.loaded:
            pipeline.load_lora_weights(adapter.path)
            if hasattr(pipeline, "fuse_lora"):
                pipeline.fuse_lora(lora_scale=adapter.ref.scale)
