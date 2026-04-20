from __future__ import annotations

from pathlib import Path

from omnirt.core.registry import ModelCapabilities, ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.executors.modular import ModularExecutor


class FakeRuntime:
    name = "cuda"
    device_name = "cpu"

    def __init__(self) -> None:
        self.backend_timeline = []
        self.to_device_calls = []

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self):
        return {}

    def synchronize(self) -> None:
        return None

    def prepare_pipeline(self, pipeline, *, model_spec, config):
        return pipeline

    def wrap_module(self, module, tag: str):
        self.backend_timeline.append({"module": tag})
        return module

    def to_device(self, tensor_or_module, dtype=None):
        self.to_device_calls.append({"target": tensor_or_module, "dtype": dtype})
        return tensor_or_module


class FakeComponentsManager:
    pass


class FakeModularPipeline:
    created = []

    def __init__(self, source: str, *, components_manager=None, device_map=None) -> None:
        self.source = source
        self.components_manager = components_manager
        self.device_map = device_map
        self.unet = object()
        self.vae = object()
        self.text_encoder = object()

    @classmethod
    def from_pretrained(cls, source: str, *, components_manager=None, device_map=None):
        instance = cls(source, components_manager=components_manager, device_map=device_map)
        cls.created.append(instance)
        return instance

    def load_components(self, *, torch_dtype=None):
        self.torch_dtype = torch_dtype

    def __call__(self, **kwargs):
        from types import SimpleNamespace
        from PIL import Image

        return SimpleNamespace(images=[Image.new("RGB", (64, 64), color="orange")])


def test_sdxl_dual_gpu_device_map_passes_through(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ModularExecutor, "_diffusers_api", lambda self: (FakeModularPipeline, FakeComponentsManager))

    executor = ModularExecutor()
    runtime = FakeRuntime()
    spec = ModelSpec(
        id="sdxl-base-1.0",
        task="text2image",
        pipeline_cls=object,
        execution_mode="modular",
        modular_pretrained_id="dummy/sdxl",
        capabilities=ModelCapabilities(required_inputs=("prompt",), artifact_kind="image"),
    )

    executor.load(
        runtime=runtime,
        model_spec=spec,
        config={"dtype": "fp16", "device_map": {"unet": 0, "vae": 1}},
        adapters=None,
    )
    result = executor.run(
        GenerateRequest(
            task="text2image",
            model="sdxl-base-1.0",
            backend="cuda",
            inputs={"prompt": "dual gpu"},
            config={"output_dir": str(tmp_path), "seed": 1},
        )
    )

    assert FakeModularPipeline.created[-1].device_map == {"unet": 0, "vae": 1}
    assert runtime.to_device_calls == []
    assert Path(result.outputs[0].path).exists()
