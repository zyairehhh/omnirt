"""Base pipeline execution skeleton."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional

from omnirt.core.adapters import AdapterManager
from omnirt.core.registry import ModelSpec
from omnirt.core.types import Artifact, GenerateRequest, GenerateResult, InsufficientMemoryError
from omnirt.telemetry.log import get_logger
from omnirt.telemetry.report import build_run_report


class BasePipeline(ABC):
    """Shared execution contract for all pipelines."""

    def __init__(self, *, runtime: Any, model_spec: ModelSpec, adapters: Optional[Iterable[Any]] = None) -> None:
        self.runtime = runtime
        self.model_spec = model_spec
        self.adapter_manager = AdapterManager()
        self.adapters = list(adapters or [])
        self.logger = get_logger()
        self.last_report = None
        self.components: Dict[str, Any] = {}

        if self.adapters:
            self.adapter_manager.load_all(self.adapters)

    @abstractmethod
    def prepare_conditions(self, req: GenerateRequest) -> Any:
        raise NotImplementedError

    @abstractmethod
    def prepare_latents(self, req: GenerateRequest, conditions: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def denoise_loop(self, latents: Any, conditions: Any, config: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latents: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def export(self, raw: Any, req: GenerateRequest) -> List[Artifact]:
        raise NotImplementedError

    def resolve_output_dir(self, req: GenerateRequest) -> Path:
        output_dir = Path(req.config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def ensure_resource_budget(self) -> None:
        minimum = self.model_spec.resource_hint.get("min_vram_gb")
        if minimum is None:
            return

        available = self.runtime.available_memory_gb()
        if available is None or available >= float(minimum):
            return

        raise InsufficientMemoryError(
            model=self.model_spec.id,
            estimated_gb=float(minimum),
            available_gb=float(available),
            hint="use smaller model or upgrade hardware; offload is planned for v0.2",
        )

    def run(self, req: GenerateRequest) -> GenerateResult:
        run_id = str(uuid.uuid4())
        timings: Dict[str, float] = {}
        outputs: List[Artifact] = []
        self.runtime.backend_timeline = []
        self.runtime.reset_memory_stats()
        self.ensure_resource_budget()

        def timed(stage: str, fn: Any) -> Any:
            started = time.perf_counter()
            self.logger.info("stage.start", extra={"stage": stage, "run_id": run_id, "model": req.model})
            try:
                return fn()
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                timings[f"{stage}_ms"] = round(elapsed_ms, 3)
                self.logger.info(
                    "stage.end",
                    extra={"stage": stage, "run_id": run_id, "model": req.model, "elapsed_ms": elapsed_ms},
                )

        try:
            conditions = timed("prepare_conditions", lambda: self.prepare_conditions(req))
            latents = timed("prepare_latents", lambda: self.prepare_latents(req, conditions))
            denoised = timed("denoise_loop", lambda: self.denoise_loop(latents, conditions, req.config))
            raw = timed("decode", lambda: self.decode(denoised))
            outputs = timed("export", lambda: self.export(raw, req))
            report = build_run_report(
                run_id=run_id,
                request=req,
                backend_name=self.runtime.name,
                timings=timings,
                memory=self.runtime.memory_stats(),
                backend_timeline=self.runtime.backend_timeline,
                artifacts=outputs,
                error=None,
            )
            self.last_report = report
            return GenerateResult(outputs=outputs, metadata=report)
        except Exception as exc:
            report = build_run_report(
                run_id=run_id,
                request=req,
                backend_name=self.runtime.name,
                timings=timings,
                memory=self.runtime.memory_stats(),
                backend_timeline=self.runtime.backend_timeline,
                artifacts=outputs,
                error=str(exc),
            )
            self.last_report = report
            self.logger.error(
                "run.failed",
                extra={"run_id": run_id, "model": req.model, "backend": self.runtime.name, "error": report.error},
            )
            raise
