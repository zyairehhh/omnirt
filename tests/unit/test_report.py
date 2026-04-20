from types import SimpleNamespace

from omnirt.backends.base import BackendRuntime
from omnirt.core.base_pipeline import BasePipeline
from omnirt.core.registry import ModelSpec
from omnirt.core.types import Artifact, GenerateRequest
from omnirt.telemetry.report import build_run_report


def test_build_run_report_copies_request_and_artifacts() -> None:
    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "hi"},
        config={"seed": 42},
    )
    artifact = Artifact(kind="image", path="out.png", mime="image/png", width=1024, height=1024)

    report = build_run_report(
        run_id="run-1",
        request=request,
        backend_name="cuda",
        timings={"decode_ms": 1.2},
        memory={"peak_mb": 12.5},
        backend_timeline=[],
        config_resolved={"seed": 42, "width": 1024},
        artifacts=[artifact],
        error=None,
    )

    assert report.run_id == "run-1"
    assert report.config_resolved["seed"] == 42
    assert report.config_resolved["width"] == 1024
    assert report.artifacts[0].path == "out.png"
    assert report.memory["peak_mb"] == 12.5


class DummyRuntime(BackendRuntime):
    name = "dummy"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        return SimpleNamespace()

    def _compile(self, module, tag):
        return module


class DummyPipeline(BasePipeline):
    def prepare_conditions(self, req: GenerateRequest):
        return {"width": req.config.get("width", 512)}

    def prepare_latents(self, req: GenerateRequest, conditions):
        return {"steps": req.config.get("num_inference_steps", 30)}

    def resolve_run_config(self, req: GenerateRequest, conditions, latents):
        return {
            "width": conditions["width"],
            "num_inference_steps": latents["steps"],
        }

    def denoise_loop(self, latents, conditions, config):
        return {"artifact": None}

    def decode(self, latents):
        return latents

    def export(self, raw, req):
        return []


def test_pipeline_run_report_uses_resolved_config() -> None:
    pipeline = DummyPipeline(
        runtime=DummyRuntime(),
        model_spec=ModelSpec(id="dummy", task="text2image", pipeline_cls=DummyPipeline),
    )
    request = GenerateRequest(
        task="text2image",
        model="dummy",
        config={},
        inputs={"prompt": "hello"},
    )

    result = pipeline.run(request)

    assert result.metadata.config_resolved == {"width": 512, "num_inference_steps": 30}
