from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import DependencyUnavailableError, GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.wan.components import DEFAULT_WAN2_2_I2V_MODEL_SOURCE, DEFAULT_WAN2_2_T2V_MODEL_SOURCE
from omnirt.models.wan.pipeline import WanPipeline


class FakeWanRuntime(BackendRuntime):
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
        return {"peak_mb": 20.0}

    def available_memory_gb(self):
        return 32.0


class FakeWanTextToVideoPipeline:
    created = []

    def __init__(self) -> None:
        self.text_encoder = object()
        self.transformer = object()
        self.transformer_2 = object()
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
        frames = [Image.new("RGB", (48, 32), color="white") for _ in range(kwargs["num_frames"])]
        return SimpleNamespace(frames=[frames])


class FakeWanImageToVideoPipeline(FakeWanTextToVideoPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = object()


def build_model_spec(model_id: str, task: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        task=task,
        pipeline_cls=WanPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 20, "dtype": "bf16"},
    )


def test_wan_models_are_registered() -> None:
    ensure_registered()

    t2v = get_model("wan2.2-t2v-14b")
    i2v = get_model("wan2.2-i2v-14b")

    assert t2v.task == "text2video"
    assert i2v.task == "image2video"


def test_wan_text2video_pipeline_exports_mp4(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(WanPipeline, "_diffusers_pipeline_cls", lambda self: FakeWanTextToVideoPipeline)

    request = GenerateRequest(
        task="text2video",
        model="wan2.2-t2v-14b",
        backend="cuda",
        inputs={"prompt": "a glass whale flying through clouds", "num_frames": 8, "fps": 12},
        config={"output_dir": str(tmp_path), "seed": 4},
    )
    pipeline = WanPipeline(runtime=FakeWanRuntime(), model_spec=build_model_spec("wan2.2-t2v-14b", "text2video"))

    result = pipeline.run(request)

    output_path = Path(result.outputs[0].path)
    assert output_path.exists()
    assert output_path.suffix == ".mp4"
    assert result.outputs[0].num_frames == 8
    assert result.metadata.memory["peak_mb"] == 20.0

    created = FakeWanTextToVideoPipeline.created[-1]
    assert created.source == DEFAULT_WAN2_2_T2V_MODEL_SOURCE
    assert created.calls[-1]["prompt"] == "a glass whale flying through clouds"


def test_wan_image2video_pipeline_uses_image_source(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (48, 32), color="navy").save(image_path)

    monkeypatch.setattr(WanPipeline, "_diffusers_pipeline_cls", lambda self: FakeWanImageToVideoPipeline)

    request = GenerateRequest(
        task="image2video",
        model="wan2.2-i2v-14b",
        backend="cuda",
        inputs={"prompt": "turn this sketch into a moving cityscape", "image": str(image_path), "num_frames": 6},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = WanPipeline(runtime=FakeWanRuntime(), model_spec=build_model_spec("wan2.2-i2v-14b", "image2video"))

    result = pipeline.run(request)

    created = FakeWanImageToVideoPipeline.created[-1]
    assert Path(result.outputs[0].path).exists()
    assert created.source == DEFAULT_WAN2_2_I2V_MODEL_SOURCE
    assert "image" in created.calls[-1]


def test_wan_pipeline_raises_clear_error_without_diffusers(tmp_path) -> None:
    request = GenerateRequest(
        task="text2video",
        model="wan2.2-t2v-14b",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={"output_dir": str(tmp_path)},
    )
    pipeline = WanPipeline(runtime=FakeWanRuntime(), model_spec=build_model_spec("wan2.2-t2v-14b", "text2video"))
    pipeline._diffusers_pipeline_cls = lambda: (_ for _ in ()).throw(
        DependencyUnavailableError("diffusers with Wan support is required for Wan execution.")
    )

    with pytest.raises(DependencyUnavailableError):
        pipeline.run(request)
