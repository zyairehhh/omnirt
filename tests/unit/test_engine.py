from __future__ import annotations

import time

from omnirt.engine import OmniEngine
from omnirt.core.registry import ModelCapabilities, clear_registry, get_model, register_model
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.executors.events import emit_event


def _build_result(request: GenerateRequest) -> GenerateResult:
    return GenerateResult(
        outputs=[
            Artifact(
                kind="image",
                path="/tmp/out.png",
                mime="image/png",
                width=64,
                height=64,
            )
        ],
        metadata=RunReport(
            run_id="run-1",
            task=request.task,
            model=request.model,
            backend=request.backend,
        ),
    )


def test_engine_reuses_cached_executor() -> None:
    clear_registry()
    captured = {"init_calls": 0}

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            captured["init_calls"] += 1

        def run(self, req):
            return _build_result(req)

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1)
    runtime = type("Runtime", (), {"name": "cpu-stub"})()
    spec = get_model("dummy-image", task="text2image")
    request = GenerateRequest(
        task="text2image",
        model="dummy-image",
        backend="cpu-stub",
        inputs={"prompt": "hello"},
        config={},
    )

    first = engine.run_sync(request, model_spec=spec, runtime=runtime)
    second = engine.run_sync(request, model_spec=spec, runtime=runtime)

    assert captured["init_calls"] == 1
    assert first.metadata.execution_mode == "legacy_call"
    assert first.metadata.job_id
    assert first.metadata.schema_version == "0.4.0"
    assert second.metadata.stream_events

    clear_registry()


def test_engine_submit_runs_background_job() -> None:
    clear_registry()

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            return _build_result(req)

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1)
    spec = get_model("dummy-image", task="text2image")
    request = GenerateRequest(
        task="text2image",
        model="dummy-image",
        backend="cpu-stub",
        inputs={"prompt": "async"},
        config={},
    )

    job = engine.submit(
        request,
        model_spec=spec,
        runtime=type("Runtime", (), {"name": "cpu-stub"})(),
    )
    resolved = engine.wait(job.id, timeout_s=2.0)

    assert resolved is not None
    assert resolved.state == "succeeded"
    assert resolved.result is not None
    assert resolved.execution_mode == "legacy_call"

    clear_registry()


def test_engine_batches_compatible_modular_requests(monkeypatch) -> None:
    clear_registry()
    captured: dict[str, object] = {}

    @register_model(
        id="dummy-modular",
        task="text2image",
        execution_mode="modular",
        capabilities=ModelCapabilities(required_inputs=("prompt",), supported_config=("seed",), supports_batching=True),
    )
    class DummyPipeline:
        pass

    class FakeExecutor:
        def __init__(self) -> None:
            self.components = {}

        def load(self, **kwargs) -> None:
            pass

        def release(self) -> None:
            pass

        def run(self, request, *, event_callback=None, cache=None):
            prompts = request.inputs["prompt"]
            captured["prompts"] = prompts
            emit_event(event_callback, "stage_start", "denoise", data={"prompt_count": len(prompts)})
            return GenerateResult(
                outputs=[
                    Artifact(kind="image", path=f"/tmp/{index}.png", mime="image/png", width=64, height=64)
                    for index, _prompt in enumerate(prompts)
                ],
                metadata=RunReport(
                    run_id="batch-run",
                    task=request.task,
                    model=request.model,
                    backend="cpu-stub",
                    execution_mode="modular",
                ),
            )

    engine = OmniEngine(max_concurrency=1, pipeline_cache_size=1, batch_window_ms=50, max_batch_size=2)
    spec = get_model("dummy-modular", task="text2image")
    runtime = type("Runtime", (), {"name": "cpu-stub"})()
    monkeypatch.setattr(
        engine,
        "_build_executor",
        lambda **kwargs: FakeExecutor(),
    )

    first_job = engine.submit(
        GenerateRequest(
            task="text2image",
            model="dummy-modular",
            backend="cpu-stub",
            inputs={"prompt": "hello"},
            config={"seed": 1},
        ),
        model_spec=spec,
        runtime=runtime,
    )
    second_job = engine.submit(
        GenerateRequest(
            task="text2image",
            model="dummy-modular",
            backend="cpu-stub",
            inputs={"prompt": "world"},
            config={"seed": 2},
        ),
        model_spec=spec,
        runtime=runtime,
    )

    first = engine.wait(first_job.id, timeout_s=2.0)
    second = engine.wait(second_job.id, timeout_s=2.0)

    assert captured["prompts"] == ["hello", "world"]
    assert first is not None and second is not None
    assert first.result.metadata.batch_size == 2
    assert second.result.metadata.batch_size == 2
    assert first.result.metadata.batch_group_id == second.result.metadata.batch_group_id
    assert first.result.outputs[0].path == "/tmp/0.png"
    assert second.result.outputs[0].path == "/tmp/1.png"

    clear_registry()
