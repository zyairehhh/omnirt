from __future__ import annotations

from omnirt.telemetry.prometheus import PrometheusMetrics


def test_prometheus_metrics_render_core_series() -> None:
    metrics = PrometheusMetrics()
    metrics.observe_job(task="text2image", model="sdxl-base-1.0", execution_mode="modular", state="succeeded")
    metrics.observe_stage_duration(stage="denoise_loop", model="sdxl-base-1.0", seconds=0.12)
    metrics.observe_cache_hit(cache_type="text_embedding")
    metrics.set_queue_depth(priority="default", depth=2)
    metrics.set_vram_peak_bytes(device="cuda:0", bytes_value=1024)

    payload = metrics.render()

    assert "# TYPE omnirt_jobs_total counter" in payload
    assert 'omnirt_jobs_total{execution_mode="modular",model="sdxl-base-1.0",state="succeeded",task="text2image"} 1.0' in payload
    assert "# TYPE omnirt_stage_duration_seconds histogram" in payload
    assert 'omnirt_cache_hits_total{cache_type="text_embedding"} 1.0' in payload
    assert 'omnirt_queue_depth{priority="default"} 2.0' in payload
    assert 'omnirt_vram_peak_bytes{device="cuda:0"} 1024.0' in payload
