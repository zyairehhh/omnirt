from types import SimpleNamespace

import pytest

from omnirt.backends import resolve_backend
from omnirt.backends.rocm import RocmBackend
from omnirt.backends.xpu import XpuBackend
from omnirt.core.types import BackendUnavailableError


def test_rocm_backend_reports_capabilities_from_torch_cuda() -> None:
    backend = object.__new__(RocmBackend)
    backend.torch = SimpleNamespace(
        version=SimpleNamespace(hip="6.0"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2),
    )

    assert backend.is_available() is True
    assert backend.capabilities().device == "rocm"
    assert backend.capabilities().device_count == 2
    assert backend.capabilities().compile_available is False


def test_xpu_backend_reports_capabilities_from_torch_xpu() -> None:
    backend = object.__new__(XpuBackend)
    backend.torch = SimpleNamespace(
        xpu=SimpleNamespace(is_available=lambda: True, device_count=lambda: 4),
    )

    assert backend.is_available() is True
    assert backend.capabilities().device == "xpu"
    assert backend.capabilities().device_count == 4
    assert backend.capabilities().compile_available is False


def test_resolve_backend_rejects_unavailable_placeholder_backends(monkeypatch) -> None:
    monkeypatch.setattr("omnirt.backends.RocmBackend", lambda: SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("omnirt.backends.XpuBackend", lambda: SimpleNamespace(is_available=lambda: False))

    with pytest.raises(BackendUnavailableError):
        resolve_backend("rocm")
    with pytest.raises(BackendUnavailableError):
        resolve_backend("xpu")
