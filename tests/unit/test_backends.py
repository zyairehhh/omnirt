from typing import Optional

from omnirt.backends.base import BackendRuntime


class DummyBackend(BackendRuntime):
    name = "dummy"

    def __init__(self, *, compile_error: Optional[str] = None):
        super().__init__()
        self.compile_error = compile_error

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        if self.compile_error is not None:
            raise RuntimeError(self.compile_error)
        return {"compiled": module, "tag": tag}


def test_wrap_module_uses_compiled_version() -> None:
    backend = DummyBackend()
    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped["tag"] == "unet"
    assert backend.backend_timeline[0].attempts[0].ok is True


def test_wrap_module_falls_back_to_eager_without_override() -> None:
    backend = DummyBackend(compile_error="unsupported op")
    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped == "module"
    assert [attempt.level for attempt in backend.backend_timeline[0].attempts] == [
        "compile",
        "kernel_override",
        "eager",
    ]


def test_wrap_module_uses_registered_override() -> None:
    backend = DummyBackend(compile_error="unsupported op")
    backend.register_override("unet", {"override": True})

    wrapped = backend.wrap_module(module="module", tag="unet")

    assert wrapped == {"override": True}
    assert backend.backend_timeline[0].attempts[1].ok is True
