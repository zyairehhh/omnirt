"""Adapter loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from omnirt.core.types import AdapterRef
from omnirt.core.weight_loader import WeightLoader


@dataclass
class LoadedAdapter:
    ref: AdapterRef
    weights: Dict[str, Any]


class AdapterManager:
    """Load adapters at pipeline initialization time."""

    def __init__(self) -> None:
        self.weight_loader = WeightLoader()
        self.loaded: List[LoadedAdapter] = []

    def load_lora(self, ref: AdapterRef, *, device: str = "cpu") -> LoadedAdapter:
        weights = self.weight_loader.load(ref.path, device=device)
        adapter = LoadedAdapter(ref=ref, weights=weights)
        self.loaded.append(adapter)
        return adapter

    def load_all(self, adapters: Iterable[AdapterRef], *, device: str = "cpu") -> List[LoadedAdapter]:
        return [self.load_lora(adapter, device=device) for adapter in adapters]

