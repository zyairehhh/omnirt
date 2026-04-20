from __future__ import annotations

import types

from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelSpec
from omnirt.core.types import GenerateRequest
from omnirt.middleware import quantization as quantization_module
from omnirt.middleware.quantization import QuantizationMiddleware, apply_quantization_runtime
from omnirt.middleware.tea_cache import TeaCacheMiddleware, apply_tea_cache_runtime


class DummyRuntime:
    name = "cpu-stub"

    def available_memory_gb(self):
        return 128.0


class DummyComponent:
    def __init__(self) -> None:
        self.quantize_calls = []
        self.casting_calls = []
        self.tea_cache_calls = []

    def quantize(self, *, mode=None, backend=None, config=None):
        self.quantize_calls.append({"mode": mode, "backend": backend, "config": dict(config or {})})

    def enable_layerwise_casting(self, *, storage_dtype=None, compute_dtype=None):
        self.casting_calls.append({"storage_dtype": storage_dtype, "compute_dtype": compute_dtype})

    def enable_teacache(self, *, ratio=None, interval=None, enabled=None):
        self.tea_cache_calls.append({"ratio": ratio, "interval": interval, "enabled": enabled})


class DummyPipelineObject:
    def __init__(self) -> None:
        self.unet = DummyComponent()
        self.vae = DummyComponent()
        self.transformer = DummyComponent()

    def quantize(self, *, mode=None, backend=None, config=None):
        self.pipeline_quantize = {"mode": mode, "backend": backend, "config": dict(config or {})}

    def enable_layerwise_casting(self, *, storage_dtype=None, compute_dtype=None):
        self.pipeline_casting = {"storage_dtype": storage_dtype, "compute_dtype": compute_dtype}

    def enable_teacache(self, *, ratio=None, interval=None, enabled=None):
        self.pipeline_tea_cache = {"ratio": ratio, "interval": interval, "enabled": enabled}


class DummyBasePipeline(BasePipeline):
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


def test_runtime_acceleration_helpers_apply_to_pipeline_and_components() -> None:
    pipeline = DummyPipelineObject()
    config = {
        "cache": "tea_cache",
        "quantization": "int8",
        "quantization_backend": "torchao",
        "enable_layerwise_casting": True,
        "layerwise_casting_storage_dtype": "fp8_e4m3fn",
        "layerwise_casting_compute_dtype": "bf16",
        "enable_tea_cache": True,
        "tea_cache_ratio": 0.2,
        "tea_cache_interval": 2,
    }

    apply_quantization_runtime(pipeline, config=config)
    apply_tea_cache_runtime(pipeline, config=config)

    assert pipeline.pipeline_quantize["mode"] == "int8"
    assert pipeline.pipeline_casting["compute_dtype"] == "bf16"
    assert pipeline.pipeline_tea_cache["interval"] == 2
    assert pipeline.unet.quantize_calls[0]["backend"] == "torchao"
    assert pipeline.transformer.tea_cache_calls[0]["ratio"] == 0.2


def test_runtime_acceleration_middleware_marks_components() -> None:
    components = {"unet": DummyComponent()}
    config = {"quantization": "fp8", "enable_tea_cache": True}

    QuantizationMiddleware().apply(components, runtime=DummyRuntime(), config=config)
    TeaCacheMiddleware().apply(components, runtime=DummyRuntime(), config=config)

    assert components["unet"].quantize_calls[0]["mode"] == "fp8"
    assert components["unet"].tea_cache_calls[0]["enabled"] is True


def test_base_pipeline_apply_pipeline_optimizations_invokes_runtime_acceleration() -> None:
    pipeline = DummyPipelineObject()
    base = DummyBasePipeline(runtime=DummyRuntime(), model_spec=ModelSpec(id="dummy", task="text2image", pipeline_cls=DummyBasePipeline))
    optimized, placement_managed = base.apply_pipeline_optimizations(
        pipeline,
        config={
            "quantization": "nf4",
            "enable_layerwise_casting": True,
            "enable_tea_cache": True,
            "tea_cache_ratio": 0.4,
        },
    )

    assert optimized is pipeline
    assert placement_managed is False
    assert pipeline.unet.quantize_calls[0]["mode"] == "nf4"
    assert pipeline.unet.casting_calls[0]["storage_dtype"] is None
    assert pipeline.unet.tea_cache_calls[0]["ratio"] == 0.4


def test_runtime_acceleration_uses_torchao_when_available(monkeypatch) -> None:
    calls = []

    def fake_apply_dynamic_quant(model=None, target=None, **kwargs):
        calls.append({"model": model, "target": target, "kwargs": kwargs})

    monkeypatch.setattr(
        quantization_module.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(apply_dynamic_quant=fake_apply_dynamic_quant)
        if name == "torchao.quantization.quant_api"
        else None,
    )

    pipeline = DummyPipelineObject()
    apply_quantization_runtime(pipeline, config={"quantization": "int8", "quantization_backend": "torchao"})

    assert calls
