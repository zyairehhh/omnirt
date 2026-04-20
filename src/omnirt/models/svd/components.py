"""SVD component metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_SVD_MODEL_SOURCE = "stabilityai/stable-video-diffusion-img2vid-xt"


@dataclass
class SVDComponents:
    image_encoder: Optional[object] = None
    unet: Optional[object] = None
    vae: Optional[object] = None
