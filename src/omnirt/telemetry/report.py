"""Run report construction helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from omnirt.core.types import Artifact, BackendTimelineEntry, GenerateRequest, RunReport


def build_run_report(
    *,
    run_id: str,
    request: GenerateRequest,
    backend_name: str,
    timings: Dict[str, float],
    memory: Dict[str, float],
    backend_timeline: Iterable[BackendTimelineEntry],
    artifacts: Iterable[Artifact],
    error: Optional[str],
) -> RunReport:
    return RunReport(
        run_id=run_id,
        task=request.task,
        model=request.model,
        backend=backend_name,
        timings=dict(timings),
        memory=dict(memory),
        backend_timeline=list(backend_timeline),
        config_resolved=dict(request.config),
        artifacts=list(artifacts),
        error=error,
    )
