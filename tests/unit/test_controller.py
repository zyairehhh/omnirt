from __future__ import annotations

from omnirt.engine import Controller, WorkerEndpoint


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
