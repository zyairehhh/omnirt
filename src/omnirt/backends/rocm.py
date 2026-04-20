"""Experimental ROCm backend placeholder."""

from __future__ import annotations

from typing import Any

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import Capabilities


class RocmBackend(BackendRuntime):
    name = "rocm"
    device_name = "cuda"

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
        except ImportError:
            self.torch = None
        else:
            self.torch = torch

    def is_available(self) -> bool:
        if self.torch is None:
            return False
        hip_version = getattr(getattr(self.torch, "version", None), "hip", None)
        cuda = getattr(self.torch, "cuda", None)
        return bool(hip_version and cuda is not None and cuda.is_available())

    def capabilities(self) -> Capabilities:
        device_count = 0
        if self.torch is not None and getattr(self.torch, "cuda", None) is not None:
            try:
                device_count = int(self.torch.cuda.device_count())
            except Exception:
                device_count = 0
        return Capabilities(
            device="rocm",
            dtype_options=["fp16", "bf16", "fp32"],
            compile_available=False,
            device_count=device_count,
        )

    def _compile(self, module: Any, tag: str) -> Any:
        raise NotImplementedError(
            f"ROCm compile path for {tag!r} is not wired yet; experimental placeholder backend only."
        )
