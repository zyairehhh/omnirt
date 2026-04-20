from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

import omnirt.api as api_module
from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import Artifact, GenerateResult, RunReport
from omnirt.server import create_app


def _dummy_result(task: str, model: str, backend: str) -> GenerateResult:
    return GenerateResult(
        outputs=[
            Artifact(
                kind="image",
                path="/tmp/server.png",
                mime="image/png",
                width=128,
                height=128,
            )
        ],
        metadata=RunReport(
            run_id="server-run",
            task=task,
            model=model,
            backend=backend,
        ),
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_registry()
    yield
    clear_registry()


def test_server_generate_and_job_routes(monkeypatch) -> None:
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("width", "height", "num_images_per_prompt"),
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            return _dummy_result(req.task, req.model, req.backend)

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(default_backend="cpu-stub", max_concurrency=1, pipeline_cache_size=1)
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    sync_response = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "hello"},
            "config": {},
        },
    )
    assert sync_response.status_code == 200
    assert sync_response.json()["metadata"]["execution_mode"] == "legacy_call"

    async_response = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "hello"},
            "config": {},
            "async_run": True,
        },
    )
    assert async_response.status_code == 200
    job_id = async_response.json()["id"]

    final_payload = None
    for _ in range(50):
        job_response = client.get(f"/v1/jobs/{job_id}")
        assert job_response.status_code == 200
        payload = job_response.json()
        if payload["state"] == "succeeded":
            final_payload = payload
            break
        time.sleep(0.02)

    assert final_payload is not None
    assert final_payload["result"]["metadata"]["job_id"] == job_id
    assert final_payload["result"]["metadata"]["trace_id"]
    assert final_payload["result"]["metadata"]["worker_id"]


def test_openai_images_generations_route(monkeypatch) -> None:
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("width", "height", "num_images_per_prompt"),
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            return _dummy_result(req.task, req.model, req.backend)

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(default_backend="cpu-stub", max_concurrency=1, pipeline_cache_size=1)
    client = TestClient(app)

    response = client.post(
        "/v1/images/generations",
        json={"model": "dummy-image", "prompt": "hello", "size": "512x512", "n": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == 1
    assert payload["data"][0]["url"] == "/tmp/server.png"


def test_openai_routes_inherit_default_request_config(monkeypatch) -> None:
    captured = {}

    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("width", "height", "num_images_per_prompt", "device_map"),
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            captured["config"] = dict(req.config)
            return _dummy_result(req.task, req.model, req.backend)

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(
        default_backend="cpu-stub",
        max_concurrency=1,
        pipeline_cache_size=1,
        default_request_config={"device_map": "balanced"},
    )
    client = TestClient(app)

    response = client.post(
        "/v1/images/generations",
        json={"model": "dummy-image", "prompt": "hello", "size": "512x512", "n": 1},
    )

    assert response.status_code == 200
    assert captured["config"]["device_map"] == "balanced"


def test_metrics_route_exposes_prometheus_payload(monkeypatch) -> None:
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(
            required_inputs=("prompt",),
            supported_config=("width", "height", "num_images_per_prompt"),
        ),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            result = _dummy_result(req.task, req.model, req.backend)
            result.metadata.timings["denoise_loop_ms"] = 12.0
            result.metadata.memory["peak_mb"] = 6.0
            result.metadata.cache_hits = ["text_embedding"]
            return result

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(default_backend="cpu-stub", max_concurrency=1, pipeline_cache_size=1)
    client = TestClient(app)

    generate = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "hello"},
            "config": {},
        },
    )
    assert generate.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "omnirt_jobs_total" in response.text
    assert "omnirt_stage_duration_seconds_bucket" in response.text
    assert "omnirt_cache_hits_total" in response.text


def test_job_trace_route_exposes_trace_payload(monkeypatch) -> None:
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            return _dummy_result(req.task, req.model, req.backend)

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(default_backend="cpu-stub", max_concurrency=1, pipeline_cache_size=1)
    client = TestClient(app)

    response = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "hello"},
            "config": {},
            "async_run": True,
        },
    )

    assert response.status_code == 200
    job_id = response.json()["id"]

    final_payload = None
    for _ in range(50):
        job_response = client.get(f"/v1/jobs/{job_id}")
        assert job_response.status_code == 200
        payload = job_response.json()
        if payload["state"] == "succeeded":
            final_payload = payload
            break
        time.sleep(0.02)

    assert final_payload is not None
    trace_response = client.get(f"/v1/jobs/{job_id}/trace")
    assert trace_response.status_code == 200
    trace_payload = trace_response.json()
    assert trace_payload["trace_id"] == final_payload["trace_id"]
    assert trace_payload["worker_id"] == final_payload["worker_id"]
    assert trace_payload["state"] == "succeeded"


def test_job_websocket_stream_supports_cancel(monkeypatch) -> None:
    @register_model(
        id="dummy-image",
        task="text2image",
        capabilities=ModelCapabilities(required_inputs=("prompt",)),
    )
    class DummyPipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, req):
            time.sleep(0.1)
            return _dummy_result(req.task, req.model, req.backend)

    monkeypatch.setattr(api_module, "ensure_registered", lambda: None)
    app = create_app(default_backend="cpu-stub", max_concurrency=1, pipeline_cache_size=1)
    client = TestClient(app)

    first = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "first"},
            "config": {},
            "async_run": True,
        },
    )
    second = client.post(
        "/v1/generate",
        json={
            "task": "text2image",
            "model": "dummy-image",
            "backend": "cpu-stub",
            "inputs": {"prompt": "second"},
            "config": {},
            "async_run": True,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    second_job_id = second.json()["id"]

    with client.websocket_connect(f"/v1/jobs/{second_job_id}/stream") as websocket:
        websocket.send_text('{"action":"cancel"}')
        observed = []
        for _ in range(6):
            observed.append(websocket.receive_json())
            if any(message.get("event") == "control_ack" for message in observed):
                break

    assert any(message.get("event") == "control_ack" and message["data"]["cancelled"] is True for message in observed)
    cancelled_job = client.get(f"/v1/jobs/{second_job_id}").json()
    assert cancelled_job["state"] == "cancelled"
