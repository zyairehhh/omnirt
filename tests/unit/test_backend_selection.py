"""Tests for the three-level fallback selection semantics of BackendRuntime.wrap_module."""

from __future__ import annotations

from typing import Optional

from omnirt.backends.base import BackendRuntime
from omnirt.core.types import BackendAttempt


class _Backend(BackendRuntime):
    name = "test"

    def __init__(self, *, compile_error: Optional[str] = None) -> None:
        super().__init__()
        self.compile_error = compile_error

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        if self.compile_error is not None:
            raise RuntimeError(self.compile_error)
        return {"compiled": module}


def _selected_levels(attempts):
    return [a.level for a in attempts if a.selected]


def test_compile_selected_records_skipped_override_and_eager() -> None:
    backend = _Backend()
    backend.wrap_module("mod", tag="unet")

    attempts = backend.backend_timeline[0].attempts
    assert [a.level for a in attempts] == ["compile", "kernel_override", "eager"]
    assert _selected_levels(attempts) == ["compile"]
    assert attempts[1].ok is False and "compile selected" in (attempts[1].reason or "")
    assert attempts[2].ok is False


def test_compile_fail_without_override_selects_eager() -> None:
    backend = _Backend(compile_error="unsupported op: sdpa")
    backend.wrap_module("mod", tag="unet")

    attempts = backend.backend_timeline[0].attempts
    assert _selected_levels(attempts) == ["eager"]
    assert attempts[0].ok is False and "unsupported op" in (attempts[0].reason or "")
    assert attempts[1].ok is False and attempts[1].reason == "no override registered"
    assert attempts[2].ok is True


def test_compile_fail_with_override_selects_override() -> None:
    backend = _Backend(compile_error="graph capture failed")
    sentinel = object()
    backend.register_override("unet", sentinel)
    result = backend.wrap_module("mod", tag="unet")

    attempts = backend.backend_timeline[0].attempts
    assert result is sentinel
    assert _selected_levels(attempts) == ["kernel_override"]
    assert attempts[2].ok is False and "kernel_override selected" in (attempts[2].reason or "")


def test_compile_ok_with_registered_override_notes_unused() -> None:
    backend = _Backend()
    backend.register_override("unet", object())
    backend.wrap_module("mod", tag="unet")

    attempts = backend.backend_timeline[0].attempts
    assert _selected_levels(attempts) == ["compile"]
    assert "override registered but unused" in (attempts[1].reason or "")


def test_backend_attempt_selected_defaults_false_for_back_compat() -> None:
    attempt = BackendAttempt(level="eager", ok=True)
    assert attempt.selected is False


def test_backend_attempt_from_dict_ignores_missing_selected() -> None:
    attempt = BackendAttempt.from_dict({"level": "compile", "ok": True})
    assert attempt.selected is False
    roundtrip = BackendAttempt.from_dict({"level": "compile", "ok": True, "selected": True})
    assert roundtrip.selected is True
