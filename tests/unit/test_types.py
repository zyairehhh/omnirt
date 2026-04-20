from pathlib import Path

from omnirt.core.types import GenerateRequest, GenerateResult, RunReport


def test_generate_request_round_trip() -> None:
    payload = {
        "task": "text2image",
        "model": "sdxl-base-1.0",
        "backend": "cuda",
        "inputs": {"prompt": "hello"},
        "config": {"seed": 7},
        "adapters": [{"kind": "lora", "path": "/tmp/demo.safetensors", "scale": 0.75}],
    }

    request = GenerateRequest.from_dict(payload)

    assert request.to_dict() == payload


def test_generate_request_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "request.yaml"
    config_path.write_text(
        "task: text2image\nmodel: sdxl-base-1.0\ninputs:\n  prompt: hello\nconfig:\n  seed: 1\n",
        encoding="utf-8",
    )

    request = GenerateRequest.from_file(config_path)

    assert request.task == "text2image"
    assert request.inputs["prompt"] == "hello"


def test_generate_result_round_trip() -> None:
    report = RunReport(run_id="1", task="text2image", model="sdxl-base-1.0", backend="cuda")
    result = GenerateResult(outputs=[], metadata=report)

    assert GenerateResult.from_dict(result.to_dict()) == result

