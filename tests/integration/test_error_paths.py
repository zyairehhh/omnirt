from pathlib import Path

import pytest

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec
from omnirt.core.types import AdapterRef, GenerateRequest, InsufficientMemoryError, WeightFormatError
from omnirt.core.weight_loader import WeightLoader
from omnirt.models.sdxl.pipeline import SDXLPipeline


class LowMemoryRuntime(BackendRuntime):
    name = "cuda"
    device_name = "cpu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def available_memory_gb(self):
        return 10.0


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=SDXLPipeline,
        default_backend="auto",
        resource_hint={"min_vram_gb": 12, "dtype": "fp16"},
    )


def test_low_memory_raises_insufficient_memory() -> None:
    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "hello"},
        config={},
    )
    pipeline = SDXLPipeline(runtime=LowMemoryRuntime(), model_spec=build_model_spec())

    with pytest.raises(InsufficientMemoryError):
        pipeline.run(request)


def test_bad_weight_suffix_raises_clear_error(tmp_path: Path) -> None:
    weight_path = tmp_path / "bad.bin"
    weight_path.write_bytes(b"bad")

    with pytest.raises(WeightFormatError):
        WeightLoader.validate_path(str(weight_path))


def test_bad_lora_suffix_fails_during_pipeline_init(tmp_path: Path) -> None:
    bad_lora = tmp_path / "style.bin"
    bad_lora.write_bytes(b"bad")

    with pytest.raises(WeightFormatError):
        SDXLPipeline(
            runtime=LowMemoryRuntime(),
            model_spec=build_model_spec(),
            adapters=[AdapterRef(kind="lora", path=str(bad_lora), scale=1.0)],
        )
