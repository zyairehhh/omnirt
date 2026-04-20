"""CUDA backend runtime."""

from __future__ import annotations

from typing import Any, List

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import BackendUnavailableError, Capabilities


class CudaBackend(BackendRuntime):
    name = "cuda"
    device_name = "cuda"

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError("PyTorch is required for the CUDA backend.") from exc
        self.torch = torch

    def is_available(self) -> bool:
        return bool(self.torch.cuda.is_available())

    def capabilities(self) -> Capabilities:
        dtype_options: List[str] = ["fp16"]
        if hasattr(self.torch, "bfloat16"):
            dtype_options.append("bf16")
        return Capabilities(
            device="cuda",
            dtype_options=dtype_options,
            compile_available=hasattr(self.torch, "compile"),
            device_count=int(self.torch.cuda.device_count()),
        )

    def reset_memory_stats(self) -> None:
        if self.is_available() and hasattr(self.torch.cuda, "reset_peak_memory_stats"):
            self.torch.cuda.reset_peak_memory_stats()

    def memory_stats(self) -> dict:
        if self.is_available() and hasattr(self.torch.cuda, "max_memory_allocated"):
            peak_mb = self.torch.cuda.max_memory_allocated() / (1024 * 1024)
            return {"peak_mb": round(float(peak_mb), 3)}
        return {}

    def available_memory_gb(self):
        if not self.is_available():
            return None
        current_device = self.torch.cuda.current_device()
        props = self.torch.cuda.get_device_properties(current_device)
        return round(float(props.total_memory) / (1024 ** 3), 3)

    def _compile(self, module: Any, tag: str) -> Any:
        if not hasattr(self.torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        return self.torch.compile(module, mode="reduce-overhead")
