from __future__ import annotations

from omnirt.core.registry import ModelCapabilities, ModelSpec
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.dispatch.batcher import RequestBatcher
from omnirt.dispatch.queue import JobWorkItem


def _make_item(*, job_id: str, prompt: str, seed: int | None = None, width: int = 1024) -> JobWorkItem:
    spec = ModelSpec(
        id="dummy-modular",
        task="text2image",
        pipeline_cls=object,
        execution_mode="modular",
        capabilities=ModelCapabilities(required_inputs=("prompt",), supports_batching=True),
    )
    request = GenerateRequest(
        task="text2image",
        model="dummy-modular",
        backend="cpu-stub",
        inputs={"prompt": prompt},
        config={"seed": seed, "width": width},
    )
    runtime = type("Runtime", (), {"name": "cpu-stub"})()
    return JobWorkItem(job_id=job_id, request=request, model_spec=spec, runtime=runtime)


def test_request_batcher_matches_compatible_modular_requests() -> None:
    batcher = RequestBatcher(batch_window_ms=50, max_batch_size=4)
    first = _make_item(job_id="job-1", prompt="hello", seed=1)
    second = _make_item(job_id="job-2", prompt="world", seed=2)
    different_width = _make_item(job_id="job-3", prompt="world", seed=3, width=768)

    assert batcher.matches(first, second) is True
    assert batcher.matches(first, different_width) is False


def test_request_batcher_combines_and_splits_results() -> None:
    batcher = RequestBatcher(batch_window_ms=50, max_batch_size=4)
    items = [
        _make_item(job_id="job-1", prompt="hello", seed=7),
        _make_item(job_id="job-2", prompt="world", seed=8),
    ]

    group = batcher.create_group(items)

    assert group is not None
    assert group.request.inputs["prompt"] == ["hello", "world"]
    assert group.request.config["seed"] == [7, 8]
    assert group.request.config["use_result_cache"] is False

    result = GenerateResult(
        outputs=[
            Artifact(kind="image", path="/tmp/0.png", mime="image/png", width=64, height=64),
            Artifact(kind="image", path="/tmp/1.png", mime="image/png", width=64, height=64),
        ],
        metadata=RunReport(run_id="run-1", task="text2image", model="dummy-modular", backend="cpu-stub"),
    )

    split = batcher.split_result(result, items, batch_group_id="batch-1")

    assert len(split) == 2
    assert split[0].outputs[0].path == "/tmp/0.png"
    assert split[1].outputs[0].path == "/tmp/1.png"
    assert split[0].metadata.batch_size == 2
    assert split[0].metadata.batch_group_id == "batch-1"
