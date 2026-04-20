from __future__ import annotations

from omnirt.core.registry import ModelCapabilities, clear_registry, register_model
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, RunReport
from omnirt.engine import Controller, InProcessWorkerClient, OmniEngine, WorkerEndpoint


def test_controller_routes_round_robin_per_model() -> None:
    controller = Controller()
    controller.register_worker(WorkerEndpoint(worker_id="w1", address="grpc://worker-1", models=("sdxl-base-1.0",)))
    controller.register_worker(WorkerEndpoint(worker_id="w2", address="grpc://worker-2", models=("sdxl-base-1.0",)))

    first = controller.route(model="sdxl-base-1.0")
    second = controller.route(model="sdxl-base-1.0")

    assert first is not None and second is not None
    assert first.worker_id == "w1"
    assert second.worker_id == "w2"


def test_controller_filters_by_tags() -> None:
    controller = Controller()
    controller.register_worker(WorkerEndpoint(worker_id="cpu", address="grpc://cpu", tags=("cpu",)))
    controller.register_worker(WorkerEndpoint(worker_id="gpu", address="grpc://gpu", tags=("gpu", "cn-shanghai")))

    selected = controller.route(model="wan2.2-t2v-14b", tags=("gpu",))

    assert selected is not None
    assert selected.worker_id == "gpu"


def test_controller_can_delegate_sync_requests_to_worker_client() -> None:
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
            return GenerateResult(
                outputs=[Artifact(kind="image", path="/tmp/out.png", mime="image/png", width=64, height=64)],
                metadata=RunReport(run_id="worker", task=req.task, model=req.model, backend=req.backend),
            )

    controller = Controller()
    controller.register_worker(WorkerEndpoint(worker_id="worker-a", address="grpc://worker-a", models=("dummy-image",)))
    remote_engine = OmniEngine(max_concurrency=1, worker_id="worker-a")
    coordinator = OmniEngine(
        max_concurrency=1,
        worker_id="coordinator",
        controller=controller,
        worker_clients={"worker-a": InProcessWorkerClient(remote_engine)},
    )

    result = coordinator.run_sync(
        GenerateRequest(task="text2image", model="dummy-image", backend="cpu-stub", inputs={"prompt": "hello"})
    )

    assert result.metadata.model == "dummy-image"
    assert result.metadata.job_id
    clear_registry()
