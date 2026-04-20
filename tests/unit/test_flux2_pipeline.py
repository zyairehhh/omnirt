from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.flux2.components import DEFAULT_FLUX2_DEV_MODEL_SOURCE
from omnirt.models.flux2.pipeline import Flux2Pipeline


class FakeFluxRuntime(BackendRuntime):
    name = "cuda"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 12.0}

    def available_memory_gb(self):
        return 32.0


class FakeFluxDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.transformer = object()
        self.vae = object()
        self.device = None
        self.dtype = None
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None):
        pipeline = cls()
        pipeline.source = source
        pipeline.dtype = torch_dtype
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        image = Image.new("RGB", (kwargs["width"], kwargs["height"]), color="white")
        return SimpleNamespace(images=[image])


def build_model_spec(model_id: str = "flux2.dev") -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task="text2image",
        pipeline_cls=Flux2Pipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 24, "dtype": "bf16"},
    )


def test_flux2_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("flux2.dev")

    assert spec.id == "flux2.dev"
    assert spec.task == "text2image"
    assert spec.pipeline_cls.__name__ == "Flux2Pipeline"


def test_flux2_pipeline_exports_png(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Flux2Pipeline, "_diffusers_pipeline_cls", lambda self: FakeFluxDiffusersPipeline)

    request = GenerateRequest(
        task="text2image",
        model="flux2.dev",
        backend="cuda",
        inputs={"prompt": "a paper dragon in a lantern shop"},
        config={"output_dir": str(tmp_path), "width": 64, "height": 32, "seed": 7},
    )
    pipeline = Flux2Pipeline(runtime=FakeFluxRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert len(result.outputs) == 1
    output_path = Path(result.outputs[0].path)
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert result.metadata.memory["peak_mb"] == 12.0
    assert [entry.module for entry in result.metadata.backend_timeline] == ["text_encoder", "transformer", "vae"]

    created = FakeFluxDiffusersPipeline.created[-1]
    assert created.source == DEFAULT_FLUX2_DEV_MODEL_SOURCE
    assert created.calls[-1]["prompt"] == "a paper dragon in a lantern shop"
    assert created.calls[-1]["max_sequence_length"] == 512


def test_flux2_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    request = GenerateRequest(
        task="text2image",
        model="flux2.dev",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = Flux2Pipeline(runtime=FakeFluxRuntime(), model_spec=build_model_spec())
    pipeline._diffusers_pipeline_cls = lambda: (_ for _ in ()).throw(
        DependencyUnavailableError("diffusers with Flux2 support is required for Flux2 execution.")
    )

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)
