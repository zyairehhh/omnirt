from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
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


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="svd-xt",
        task="image2video",
        pipeline_cls=SVDPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 14, "dtype": "fp16"},
    )


def test_svd_pipeline_exports_mp4(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (48, 32), color="navy").save(image_path)

    monkeypatch.setattr(SVDPipeline, "_diffusers_pipeline_cls", lambda self: FakeSVDDiffusersPipeline)
    monkeypatch.setattr("omnirt.models.svd.pipeline.build_scheduler", lambda config: {"name": "euler"})

    request = GenerateRequest(
        task="image2video",
        model="svd-xt",
        backend="cuda",
        inputs={"image": str(image_path), "num_frames": 4, "fps": 6},
        config={"output_dir": str(tmp_path), "seed": 5},
    )
    pipeline = SVDPipeline(runtime=FakeVideoRuntime(), model_spec=build_model_spec())

    result = pipeline.run(request)

    assert len(result.outputs) == 1
    output_path = Path(result.outputs[0].path)
    assert output_path.exists()
    assert output_path.suffix == ".mp4"
    assert result.outputs[0].num_frames == 4
    assert result.metadata.memory["peak_mb"] == 16.0
    assert [entry.module for entry in result.metadata.backend_timeline] == ["image_encoder", "unet", "vae"]

    created = FakeSVDDiffusersPipeline.created[-1]
    assert created.calls[-1]["num_frames"] == 4
    assert created.calls[-1]["fps"] == 6
    assert created.scheduler == {"name": "euler"}


def test_svd_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (48, 32), color="navy").save(image_path)

    request = GenerateRequest(
        task="image2video",
        model="svd-xt",
        backend="cuda",
        inputs={"image": str(image_path)},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = SVDPipeline(runtime=FakeVideoRuntime(), model_spec=build_model_spec())

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)
