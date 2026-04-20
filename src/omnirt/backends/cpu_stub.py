"""CPU stub backend for local dry-runs without GPU/NPU hardware.

Intended for developer machines (e.g. macOS) where neither CUDA nor Ascend is
available. Walks prepare_conditions / resolve_run_config so the contract and
registry integration can be exercised, but refuses to execute denoise_loop.
"""

from __future__ import annotations

from typing import Any, Optional

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import BackendUnavailableError, Capabilities


class CpuStubBackend(BackendRuntime):
    name = "cpu-stub"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> Capabilities:
        return Capabilities(
            device="cpu",
            dtype_options=["fp32"],
            compile_available=False,
            device_count=1,
        )

    def available_memory_gb(self) -> Optional[float]:
        return None

    def memory_stats(self) -> dict:
        return {}

    def synchronize(self) -> None:
        return None

    def _compile(self, module: Any, tag: str) -> Any:
        raise NotImplementedError("cpu-stub has no compile path; wrap_module falls back to eager")

    def to_device(self, tensor_or_module: Any, dtype: Optional[str] = None) -> Any:
        return tensor_or_module


def denoise_guard(runtime: BackendRuntime) -> None:
    """Raise if the provided runtime cannot execute a denoise loop."""

    if getattr(runtime, "name", "") == "cpu-stub":
        raise BackendUnavailableError(
            "cpu-stub cannot execute denoise; for dry-run only. "
            "Install CUDA or Ascend hardware, or pass a smaller/mock pipeline."
        )
