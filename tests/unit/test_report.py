from omnirt.core.types import Artifact, GenerateRequest
from omnirt.telemetry.report import build_run_report


def test_build_run_report_copies_request_and_artifacts() -> None:
    request = GenerateRequest(
        task="text2image",
        model="sdxl-base-1.0",
        backend="cuda",
        inputs={"prompt": "hi"},
        config={"seed": 42},
    )
    artifact = Artifact(kind="image", path="out.png", mime="image/png", width=1024, height=1024)

    report = build_run_report(
        run_id="run-1",
        request=request,
        backend_name="cuda",
        timings={"decode_ms": 1.2},
        memory={"peak_mb": 12.5},
        backend_timeline=[],
        artifacts=[artifact],
        error=None,
    )

    assert report.run_id == "run-1"
    assert report.config_resolved["seed"] == 42
    assert report.artifacts[0].path == "out.png"
    assert report.memory["peak_mb"] == 12.5
