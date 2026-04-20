"""OmniEngine orchestrates executors, jobs, and worker threads."""

from __future__ import annotations

from hashlib import sha256
import json
import time
import uuid
from typing import Sequence

from omnirt.backends import resolve_backend
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest, GenerateResult, is_generate_result_like
from omnirt.dispatch import BatchGroup, JobQueue, JobWorkItem, RequestBatcher, TERMINAL_JOB_STATES, Worker
from omnirt.engine.job import JobRecord
from omnirt.engine.pipeline_cache import PipelineCache
from omnirt.engine.result_cache import ResultCache
from omnirt.engine.store import InMemoryJobStore
from omnirt.executors import LegacyCallExecutor, ModularExecutor, SubprocessExecutor
from omnirt.executors.events import emit_event
from omnirt.middleware.telemetry import attach_stream_events


class OmniEngine:
    def __init__(
        self,
        *,
        max_concurrency: int = 1,
        pipeline_cache_size: int = 4,
        result_cache_size: int = 256,
        job_store: InMemoryJobStore | None = None,
        batch_window_ms: int = 0,
        max_batch_size: int = 1,
    ) -> None:
        self.store = job_store or InMemoryJobStore()
        self.pipeline_cache = PipelineCache(max_size=pipeline_cache_size)
        self.result_cache = ResultCache(max_items=result_cache_size)
        self.job_queue = JobQueue()
        self.batcher = RequestBatcher(batch_window_ms=batch_window_ms, max_batch_size=max_batch_size)
        self._workers = [
            Worker(
                name=f"omnirt-worker-{index}",
                job_queue=self.job_queue,
                handler=self._handle_work_items,
                batcher=self.batcher,
            )
            for index in range(max(int(max_concurrency), 1))
        ]
        for worker in self._workers:
            worker.start()

    def run_sync(self, request: GenerateRequest, *, model_spec: ModelSpec | None = None, runtime=None):
        spec = model_spec or get_model(request.model, task=request.task)
        selected_runtime = runtime or resolve_backend(request.backend or spec.default_backend)
        job = self._create_job(request, backend_name=getattr(selected_runtime, "name", request.backend or "auto"))
        result = self._execute(job.id, model_spec=spec, runtime=selected_runtime, raise_on_error=True)
        resolved = self.get_job(job.id)
        if resolved is None:
            raise RuntimeError(f"Job {job.id} disappeared.")
        if resolved.result is not None:
            return resolved.result
        return result

    def submit(self, request: GenerateRequest, *, model_spec: ModelSpec | None = None, runtime=None) -> JobRecord:
        spec = model_spec or get_model(request.model, task=request.task)
        selected_runtime = runtime or resolve_backend(request.backend or spec.default_backend)
        job = self._create_job(request, backend_name=getattr(selected_runtime, "name", request.backend or "auto"))
        self.job_queue.put(JobWorkItem(job_id=job.id, request=request, model_spec=spec, runtime=selected_runtime))
        return job

    def cancel(self, job_id: str) -> JobRecord | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        if job.state == "queued":
            job.state = "cancelled"
            job.finished_at_ms = int(time.time() * 1000)
            self.store.save(job)
            self.store.append_event(job_id, emit_event(None, "job_cancelled", "job", data={"job_id": job_id}))
        return self.store.get(job_id)

    def get_job(self, job_id: str) -> JobRecord | None:
        return self.store.get(job_id)

    def _create_job(self, request: GenerateRequest, *, backend_name: str) -> JobRecord:
        job = JobRecord(
            id=str(uuid.uuid4()),
            request=request,
            backend=backend_name,
            enqueued_at_ms=int(time.time() * 1000),
        )
        self.store.create(job)
        self.store.append_event(job.id, emit_event(None, "job_enqueued", "job", data={"job_id": job.id}))
        return self.store.get(job.id)  # type: ignore[return-value]

    def _handle_work_items(self, items: Sequence[JobWorkItem]) -> None:
        group = self.batcher.create_group(items)
        if group is not None:
            self._execute_batch(group)
            return
        for item in items:
            self._execute(item.job_id, model_spec=item.model_spec, runtime=item.runtime)

    def _execute(self, job_id: str, *, model_spec: ModelSpec, runtime, raise_on_error: bool = False):
        job = self.store.get(job_id)
        if job is None or job.state == "cancelled":
            return None
        job.state = "running"
        job.started_at_ms = int(time.time() * 1000)
        self.store.save(job)
        self.store.append_event(job.id, emit_event(None, "job_started", "job", data={"job_id": job.id}))

        def on_event(event) -> None:
            self.store.append_event(job_id, event)

        try:
            executor_entry = self.pipeline_cache.get_or_create(
                self._cache_key(model_spec=model_spec, runtime=runtime, request=job.request),
                lambda: self._build_executor(model_spec=model_spec, runtime=runtime, request=job.request),
            )
            with executor_entry.lock:
                result = executor_entry.value.run(job.request, event_callback=on_event, cache=self.result_cache)
        except Exception as exc:
            job = self.store.get(job_id)
            if job is None:
                return None
            job.state = "failed"
            job.error = str(exc)
            job.execution_mode = getattr(model_spec, "execution_mode", "legacy_call")
            job.finished_at_ms = int(time.time() * 1000)
            self.store.save(job)
            self.store.append_event(
                job.id,
                emit_event(None, "job_failed", "job", data={"job_id": job.id, "error": str(exc)}),
            )
            if raise_on_error:
                raise
            return None

        job = self.store.get(job_id)
        if job is None:
            return result
        job.state = "succeeded"
        job.result = result
        job.error = None
        job.execution_mode = getattr(model_spec, "execution_mode", "legacy_call")
        job.finished_at_ms = int(time.time() * 1000)
        self.store.save(job)
        finished_event = emit_event(None, "job_finished", "job", data={"job_id": job.id})
        self.store.append_event(job.id, finished_event)
        if is_generate_result_like(result):
            latest = self.store.get(job.id)
            if latest is not None:
                self._hydrate_result_metadata(
                    result,
                    job=latest,
                    model_spec=model_spec,
                    executor=executor_entry.value,
                    batch_size=1,
                    batch_group_id=None,
                )
                latest.result = result
                self.store.save(latest)
        return result

    def _execute_batch(self, group: BatchGroup) -> None:
        active_items = [
            item
            for item in group.items
            if (job := self.store.get(item.job_id)) is not None and job.state != "cancelled"
        ]
        if not active_items:
            return
        if len(active_items) == 1:
            item = active_items[0]
            self._execute(item.job_id, model_spec=item.model_spec, runtime=item.runtime)
            return

        model_spec = active_items[0].model_spec
        runtime = active_items[0].runtime
        for item in active_items:
            job = self.store.get(item.job_id)
            if job is None:
                continue
            job.state = "running"
            job.started_at_ms = int(time.time() * 1000)
            self.store.save(job)
            self.store.append_event(
                job.id,
                emit_event(
                    None,
                    "job_started",
                    "job",
                    data={"job_id": job.id, "batch_group_id": group.group_id, "batch_size": len(active_items)},
                ),
            )

        def on_event(event) -> None:
            for item in active_items:
                self.store.append_event(item.job_id, event)

        try:
            executor_entry = self.pipeline_cache.get_or_create(
                self._cache_key(model_spec=model_spec, runtime=runtime, request=group.request),
                lambda: self._build_executor(model_spec=model_spec, runtime=runtime, request=group.request),
            )
            with executor_entry.lock:
                result = executor_entry.value.run(group.request, event_callback=on_event, cache=self.result_cache)
            if not isinstance(result, GenerateResult):
                raise TypeError("Batched execution requires GenerateResult output.")
            split_results = self.batcher.split_result(result, active_items, batch_group_id=group.group_id)
        except Exception as exc:
            for item in active_items:
                job = self.store.get(item.job_id)
                if job is None:
                    continue
                job.state = "failed"
                job.error = str(exc)
                job.execution_mode = getattr(model_spec, "execution_mode", "legacy_call")
                job.finished_at_ms = int(time.time() * 1000)
                self.store.save(job)
                self.store.append_event(
                    job.id,
                    emit_event(None, "job_failed", "job", data={"job_id": job.id, "error": str(exc)}),
                )
            return

        for item, child_result in zip(active_items, split_results):
            job = self.store.get(item.job_id)
            if job is None:
                continue
            job.state = "succeeded"
            job.result = child_result
            job.error = None
            job.execution_mode = getattr(model_spec, "execution_mode", "legacy_call")
            job.finished_at_ms = int(time.time() * 1000)
            self.store.save(job)
            self.store.append_event(
                job.id,
                emit_event(None, "job_finished", "job", data={"job_id": job.id, "batch_group_id": group.group_id}),
            )
            latest = self.store.get(job.id)
            if latest is None:
                continue
            self._hydrate_result_metadata(
                child_result,
                job=latest,
                model_spec=model_spec,
                executor=executor_entry.value,
                batch_size=len(active_items),
                batch_group_id=group.group_id,
            )
            latest.result = child_result
            self.store.save(latest)

    def _build_executor(self, *, model_spec: ModelSpec, runtime, request: GenerateRequest):
        if model_spec.execution_mode == "subprocess":
            executor = SubprocessExecutor()
        elif model_spec.execution_mode == "modular":
            executor = ModularExecutor()
        else:
            executor = LegacyCallExecutor()
        executor.load(
            runtime=runtime,
            model_spec=model_spec,
            config=request.config,
            adapters=request.adapters,
        )
        return executor

    def _cache_key(self, *, model_spec: ModelSpec, runtime, request: GenerateRequest) -> tuple[str, str, str, str]:
        adapter_fingerprint = [
            {"kind": adapter.kind, "path": adapter.path, "scale": adapter.scale}
            for adapter in (request.adapters or [])
        ]
        payload = {
            "config": self._executor_config(request.config),
            "adapters": adapter_fingerprint,
        }
        fingerprint = sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        return (model_spec.id, model_spec.task, getattr(runtime, "name", "unknown"), fingerprint)

    def _executor_config(self, config: dict[str, object]) -> dict[str, object]:
        run_local_keys = {"seed", "output_dir", "use_result_cache"}
        return {key: value for key, value in config.items() if key not in run_local_keys}

    def _hydrate_result_metadata(
        self,
        result: GenerateResult,
        *,
        job: JobRecord,
        model_spec: ModelSpec,
        executor,
        batch_size: int,
        batch_group_id: str | None,
    ) -> None:
        result.metadata.job_id = job.id
        result.metadata.enqueued_at_ms = job.enqueued_at_ms
        result.metadata.queue_wait_ms = job.queue_wait_ms
        result.metadata.execution_mode = getattr(result.metadata, "execution_mode", None) or getattr(
            model_spec, "execution_mode", "legacy_call"
        )
        result.metadata.device_placement = self._resolve_device_placement(executor)
        result.metadata.batch_size = int(batch_size)
        result.metadata.batch_group_id = batch_group_id
        attach_stream_events(result, job.events)

    def _resolve_device_placement(self, executor) -> dict[str, str]:
        placement: dict[str, str] = {}
        for name, component in dict(getattr(executor, "components", {}) or {}).items():
            device = self._component_device(component)
            if device is not None:
                placement[str(name)] = device
        return placement

    def _component_device(self, component) -> str | None:
        device = getattr(component, "device", None)
        if device is not None:
            return str(device)
        parameters = getattr(component, "parameters", None)
        if callable(parameters):
            try:
                first_parameter = next(parameters())
            except (StopIteration, TypeError):
                first_parameter = None
            if first_parameter is not None:
                return str(getattr(first_parameter, "device", ""))
        buffers = getattr(component, "buffers", None)
        if callable(buffers):
            try:
                first_buffer = next(buffers())
            except (StopIteration, TypeError):
                first_buffer = None
            if first_buffer is not None:
                return str(getattr(first_buffer, "device", ""))
        return None

    def is_ready(self) -> bool:
        return True

    def wait(self, job_id: str, *, timeout_s: float = 30.0) -> JobRecord | None:
        deadline = time.time() + max(timeout_s, 0.0)
        while time.time() <= deadline:
            job = self.store.get(job_id)
            if job is None or job.state in TERMINAL_JOB_STATES:
                return job
            time.sleep(0.01)
        return self.store.get(job_id)
