"""Benchmark runner for queued OmniRT requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Iterable, Sequence

from omnirt.core.types import GenerateRequest, GenerateResult
from omnirt.engine import OmniEngine

from .metrics import BenchReport, summarize_latency


@dataclass(frozen=True)
class BenchScenario:
    name: str
    request_template: GenerateRequest
    concurrency: int = 1
    total_requests: int = 10
    warmup: int = 1
    batch_window_ms: int = 0
    max_batch_size: int = 1


@dataclass(frozen=True)
class BenchInvocation:
    latency_ms: float
    ttft_ms: float
    result: GenerateResult


def run_bench(scenario: BenchScenario, *, engine: OmniEngine | None = None) -> BenchReport:
    active_engine = engine or OmniEngine(
        max_concurrency=scenario.concurrency,
        batch_window_ms=scenario.batch_window_ms,
        max_batch_size=scenario.max_batch_size,
    )

    for index in range(max(scenario.warmup, 0)):
        active_engine.run_sync(_request_for_index(scenario.request_template, index))

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(scenario.concurrency, 1)) as pool:
        invocations = list(pool.map(lambda index: _run_once(active_engine, scenario.request_template, index), range(scenario.total_requests)))
    total_duration_s = max(time.perf_counter() - started, 0.0001)
    return _build_report(scenario, invocations, total_duration_s=total_duration_s)


def _run_once(engine: OmniEngine, request_template: GenerateRequest, index: int) -> BenchInvocation:
    request = _request_for_index(request_template, index)
    started = time.perf_counter()
    job = engine.submit(request)
    resolved = engine.wait(job.id, timeout_s=300.0)
    if resolved is None or resolved.result is None:
        raise RuntimeError(f"Bench job {job.id} did not complete successfully.")
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    return BenchInvocation(
        latency_ms=latency_ms,
        ttft_ms=_first_token_latency_ms(resolved.result),
        result=resolved.result,
    )


def _request_for_index(template: GenerateRequest, index: int) -> GenerateRequest:
    payload = template.to_dict()
    config = dict(payload.get("config", {}))
    base_seed = config.get("seed")
    if isinstance(base_seed, int):
        config["seed"] = base_seed + index
    elif base_seed is None:
        config["seed"] = index
    payload["config"] = config
    return GenerateRequest.from_dict(payload)


def _build_report(scenario: BenchScenario, invocations: Sequence[BenchInvocation], *, total_duration_s: float) -> BenchReport:
    latencies = [invocation.latency_ms for invocation in invocations]
    ttfts = [invocation.ttft_ms for invocation in invocations if invocation.ttft_ms > 0]
    execution_mode_breakdown: dict[str, int] = {}
    peak_vram = 0.0
    cache_hit_count = 0

    for invocation in invocations:
        metadata = invocation.result.metadata
        execution_mode = metadata.execution_mode or "unknown"
        execution_mode_breakdown[execution_mode] = execution_mode_breakdown.get(execution_mode, 0) + 1
        peak_vram = max(peak_vram, _peak_memory_value(metadata.memory))
        if metadata.cache_hits:
            cache_hit_count += 1

    return BenchReport(
        scenario=scenario.name,
        total_requests=scenario.total_requests,
        concurrency=scenario.concurrency,
        warmup=scenario.warmup,
        total_duration_s=round(total_duration_s, 3),
        throughput_rps=round(float(scenario.total_requests) / total_duration_s, 3),
        latency_ms=summarize_latency(latencies),
        ttft_ms=summarize_latency(ttfts),
        peak_vram=round(peak_vram, 3),
        cache_hit_ratio=round(cache_hit_count / max(len(invocations), 1), 3),
        execution_mode_breakdown=execution_mode_breakdown,
    )


def _first_token_latency_ms(result: GenerateResult) -> float:
    metadata = result.metadata
    if metadata.enqueued_at_ms is None:
        return 0.0
    timestamps = [
        event.timestamp_ms
        for event in metadata.stream_events
        if event.event not in {"job_enqueued", "job_finished"}
    ]
    if not timestamps:
        return 0.0
    return round(max(min(timestamps) - metadata.enqueued_at_ms, 0), 3)


def _peak_memory_value(memory: dict[str, object]) -> float:
    numeric = [float(value) for value in memory.values() if isinstance(value, (int, float))]
    return max(numeric, default=0.0)
