"""Ascend backend runtime."""

from __future__ import annotations

from typing import Any, Optional

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import BackendUnavailableError, Capabilities


class AscendBackend(BackendRuntime):
    name = "ascend"
    device_name = "npu"

    def __init__(self) -> None:
        super().__init__()
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError("PyTorch is required for the Ascend backend.") from exc
        self.torch = torch
        try:
            import torch_npu
        except ImportError:
            torch_npu = None
        self.torch_npu = torch_npu

    def is_available(self) -> bool:
        return self._device_count() > 0

    def capabilities(self) -> Capabilities:
        device_count = self._device_count()
        return Capabilities(
            device="ascend",
            dtype_options=["fp16", "bf16"],
            compile_available=False,
            device_count=device_count,
        )

    def memory_stats(self) -> dict:
        npu = getattr(self.torch_npu, "npu", None) if self.torch_npu is not None else None
        max_allocated = getattr(npu, "max_memory_allocated", None) if npu is not None else None
        if not callable(max_allocated):
            return {}
        try:
            peak_bytes = float(max_allocated())
        except Exception:
            return {}
        return {"peak_mb": round(peak_bytes / (1024 * 1024), 3)}

    def available_memory_gb(self) -> Optional[float]:
        if self.torch_npu is None:
            return None
        npu = getattr(self.torch_npu, "npu", None)
        get_mem_info = getattr(npu, "mem_get_info", None) if npu is not None else None
        if callable(get_mem_info):
            try:
                free_bytes, _total = get_mem_info()
                return round(float(free_bytes) / (1024 ** 3), 3)
            except Exception:
                pass
        reserved = getattr(npu, "memory_reserved", None) if npu is not None else None
        allocated = getattr(npu, "memory_allocated", None) if npu is not None else None
        if callable(reserved) and callable(allocated):
            try:
                return round(float(reserved() - allocated()) / (1024 ** 3), 3)
            except Exception:
                pass
        return None

    def synchronize(self) -> None:
        npu = getattr(self.torch_npu, "npu", None) if self.torch_npu is not None else None
        sync = getattr(npu, "synchronize", None) if npu is not None else None
        if callable(sync):
            try:
                sync()
            except Exception:
                pass

    def _compile(self, module: Any, tag: str) -> Any:
        raise NotImplementedError(
            "torch_npu compile not wired up in v0.1; wrap_module will fall back to eager or kernel_override"
        )

    def _device_count(self) -> int:
        if self.torch_npu is None:
            return 0

        npu = getattr(self.torch_npu, "npu", None)
        device_count = getattr(npu, "device_count", None)
        if not callable(device_count):
            return 0

        try:
            return int(device_count())
        except Exception:
            return 0
