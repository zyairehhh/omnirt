from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.svd.components import DEFAULT_SVD_MODEL_SOURCE
from omnirt.models.svd.pipeline import SVDPipeline


class FakeVideoRuntime(BackendRuntime):
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
        return {"peak_mb": 16.0}

    def available_memory_gb(self):
        return 32.0


class FakeSVDDiffusersPipeline:
    created = []

    def __init__(self) -> None:
        self.image_encoder = object()
        self.unet = object()
        self.vae = object()
        self.scheduler = None
        self.device = None
        self.dtype = None
        self.calls = []

    @classmethod
    def from_pretrained(cls, source, torch_dtype=None, use_safetensors=True):
        pipeline = cls()
        pipeline.source = source
        pipeline.dtype = torch_dtype
        pipeline.use_safetensors = use_safetensors
        pipeline.scheduler = SimpleNamespace(config={"beta_start": 0.1})
        cls.created.append(pipeline)
        return pipeline

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        frames = [Image.new("RGB", (48, 32), color="white") for _ in range(kwargs["num_frames"])]
        return SimpleNamespace(frames=[frames])


def build_model_spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task="image2video",
        pipeline_cls=SVDPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    )


def test_base_svd_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("svd")

    assert spec.id == "svd"
    assert spec.task == "image2video"
    assert spec.pipeline_cls.__name__ == "SVDPipeline"
    assert spec.pipeline_cls.__module__ == "omnirt.models.svd.pipeline"


def test_svd_base_uses_base_defaults(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (48, 32), color="navy").save(image_path)

    monkeypatch.setattr(SVDPipeline, "_diffusers_pipeline_cls", lambda self: FakeSVDDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.svd.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="image2video",
        model="svd",
        backend="cuda",
        inputs={"image": str(image_path)},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = SVDPipeline(runtime=FakeVideoRuntime(), model_spec=build_model_spec("svd"))

    result = pipeline.run(request)

    created = FakeSVDDiffusersPipeline.created[-1]
    output_path = Path(result.outputs[0].path)

    assert output_path.exists()
    assert created.source == DEFAULT_SVD_MODEL_SOURCE
    assert created.calls[-1]["num_frames"] == 14
    assert result.metadata.config_resolved["num_frames"] == 14
