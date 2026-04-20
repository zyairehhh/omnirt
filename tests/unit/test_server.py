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
