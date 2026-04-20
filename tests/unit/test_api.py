from omnirt.api import generate
from omnirt.core.registry import ModelSpec, clear_registry, register_model
from omnirt.core.types import GenerateRequest


def test_generate_rejects_task_model_mismatch(monkeypatch) -> None:
    clear_registry()

    @register_model(id="dummy-video", task="image2video")
    class DummyPipeline:
        def __init__(self, **kwargs):
            raise AssertionError("pipeline should not be constructed for invalid requests")

    monkeypatch.setattr("omnirt.api.ensure_registered", lambda: None)
    monkeypatch.setattr("omnirt.api.resolve_backend", lambda name: object())

    request = GenerateRequest(
        task="text2image",
        model="dummy-video",
        backend="auto",
        inputs={"prompt": "hello"},
        config={},
    )

    try:
        generate(request)
    except ValueError as exc:
        assert "dummy-video" in str(exc)
        assert "image2video" in str(exc)
        assert "text2image" in str(exc)
    else:
        raise AssertionError("Expected task/model mismatch to raise ValueError")

    clear_registry()
