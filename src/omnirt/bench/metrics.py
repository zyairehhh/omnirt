"""Bench metric aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, Sequence


def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(min(ratio, 1.0), 0.0) * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class BenchReport:
    scenario: str
    total_requests: int
    concurrency: int
    warmup: int
    total_duration_s: float
    throughput_rps: float
    latency_ms: Dict[str, float]
    ttft_ms: Dict[str, float]
    peak_vram: float
    cache_hit_ratio: float
    execution_mode_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario,
            "total_requests": self.total_requests,
            "concurrency": self.concurrency,
            "warmup": self.warmup,
            "total_duration_s": self.total_duration_s,
            "throughput_rps": self.throughput_rps,
            "latency_ms": dict(self.latency_ms),
            "ttft_ms": dict(self.ttft_ms),
            "peak_vram": self.peak_vram,
            "cache_hit_ratio": self.cache_hit_ratio,
            "execution_mode_breakdown": dict(self.execution_mode_breakdown),
        }


def summarize_latency(values: Iterable[float]) -> Dict[str, float]:
    ordered = [float(value) for value in values]
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "p50": round(percentile(ordered, 0.50), 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "p99": round(percentile(ordered, 0.99), 3),
    }
