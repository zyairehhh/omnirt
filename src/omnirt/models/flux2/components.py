"""Flux2 component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


DEFAULT_FLUX2_DEV_MODEL_SOURCE = "black-forest-labs/FLUX.2-dev"


@dataclass
class Flux2Components:
    text_encoder: Optional[object] = None
    transformer: Optional[object] = None
    vae: Optional[object] = None
