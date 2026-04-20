from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import clear_registry, get_model, register_model
from omnirt.core.types import GenerateRequest, ModelNotRegisteredError


class DummyPipeline(BasePipeline):
    def prepare_conditions(self, req: GenerateRequest):
        return {}

    def prepare_latents(self, req: GenerateRequest, conditions):
        return {}

    def denoise_loop(self, latents, conditions, config):
        return {}

    def decode(self, latents):
        return latents

    def export(self, raw, req):
        return []


def test_register_model_decorator() -> None:
    clear_registry()

    @register_model(id="dummy", task="text2image")
    class RegisteredPipeline(DummyPipeline):
        pass

    spec = get_model("dummy")

    assert spec.id == "dummy"
    assert spec.pipeline_cls is RegisteredPipeline


def test_get_model_raises_for_unknown() -> None:
    clear_registry()

    try:
        get_model("missing")
    except ModelNotRegisteredError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected ModelNotRegisteredError")

