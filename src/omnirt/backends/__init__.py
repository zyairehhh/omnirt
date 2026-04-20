"""Backend runtime selection."""

from __future__ import annotations

from omnirt.backends.ascend import AscendBackend
from omnirt.backends.cuda import CudaBackend
from omnirt.core.types import BackendUnavailableError


def resolve_backend(name: str):
    requested = name or "auto"
    if requested == "cuda":
        backend = CudaBackend()
        if not backend.is_available():
            raise BackendUnavailableError("CUDA backend requested but no CUDA device is available.")
        return backend
    if requested == "ascend":
        backend = AscendBackend()
        if not backend.is_available():
            raise BackendUnavailableError("Ascend backend requested but torch_npu/NPU device is unavailable.")
        return backend
    if requested != "auto":
        raise BackendUnavailableError(f"Unsupported backend: {requested}")

    cuda = CudaBackend()
    if cuda.is_available():
        return cuda

    ascend = AscendBackend()
    if ascend.is_available():
        return ascend

    raise BackendUnavailableError("No supported backend is available. Tried CUDA and Ascend.")
