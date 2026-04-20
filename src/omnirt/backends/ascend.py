"""Ascend backend runtime."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

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
        if self.torch_npu is None:
            return False
        return bool(hasattr(self.torch_npu, "npu"))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            device="ascend",
            dtype_options=["fp16", "bf16"],
            compile_available=self.torch_npu is not None,
            device_count=1 if self.torch_npu is not None else 0,
        )

    def memory_stats(self) -> dict:
        return {}

    def _compile(self, module: Any, tag: str) -> Any:
        if self.torch_npu is None:
            raise RuntimeError("torch_npu is unavailable")
        graph_mode = getattr(getattr(self.torch_npu, "npu", None), "graph_mode", None)
        context = graph_mode() if callable(graph_mode) else nullcontext()
        with context:
            return module
