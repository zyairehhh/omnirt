"""Experimental Intel XPU backend placeholder."""

from __future__ import annotations

from typing import Any

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import Capabilities


class XpuBackend(BackendRuntime):
    name = "xpu"
    device_name = "xpu"

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
        except ImportError:
            self.torch = None
        else:
            self.torch = torch

    def is_available(self) -> bool:
        xpu = getattr(self.torch, "xpu", None) if self.torch is not None else None
        return bool(xpu is not None and xpu.is_available())

    def capabilities(self) -> Capabilities:
        device_count = 0
        xpu = getattr(self.torch, "xpu", None) if self.torch is not None else None
        if xpu is not None:
            try:
                device_count = int(xpu.device_count())
            except Exception:
                device_count = 0
        return Capabilities(
            device="xpu",
            dtype_options=["fp16", "bf16", "fp32"],
            compile_available=False,
            device_count=device_count,
        )

    def _compile(self, module: Any, tag: str) -> Any:
        raise NotImplementedError(
            f"XPU compile path for {tag!r} is not wired yet; experimental placeholder backend only."
        )
