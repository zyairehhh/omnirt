import json

from omnirt.cli.main import build_parser, main, request_from_args


def test_build_parser_accepts_generate_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate", "--config", "request.yaml"])

    assert args.command == "generate"
    assert args.config == "request.yaml"


def test_request_from_args_builds_text2image_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
            "--num-inference-steps",
            "20",
            "--seed",
            "9",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "text2image"
    assert request.model == "sdxl-base-1.0"
    assert request.inputs["prompt"] == "hello"
    assert request.config["num_inference_steps"] == 20
    assert request.config["seed"] == 9


def test_main_emits_json(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "request.yaml"
    config_path.write_text(
        "task: text2image\nmodel: sdxl-base-1.0\nbackend: cuda\ninputs:\n  prompt: hello\nconfig: {}\n",
        encoding="utf-8",
    )

    def fake_generate(request, backend=None):
        class Result:
            def to_dict(self):
                return {"outputs": [], "metadata": {"run_id": "1", "task": request.task, "model": request.model}}

        return Result()

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(["generate", "--config", str(config_path), "--json"])
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["metadata"]["model"] == "sdxl-base-1.0"


def test_main_supports_direct_cli_arguments(monkeypatch, capsys) -> None:
    def fake_generate(request, backend=None):
        class Result:
            def to_dict(self):
                return {"outputs": [], "metadata": {"run_id": "1", "task": request.task, "model": request.model}}

        return Result()

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
            "--json",
        ]
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["metadata"]["task"] == "text2image"


def test_request_from_args_builds_image2video_request() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--task",
            "image2video",
            "--model",
            "svd-xt",
            "--image",
            "frame.png",
            "--num-frames",
            "12",
            "--fps",
            "8",
            "--frame-bucket",
            "96",
        ]
    )

    request = request_from_args(args, parser)

    assert request.task == "image2video"
    assert request.inputs["image"] == "frame.png"
    assert request.inputs["num_frames"] == 12
    assert request.config["frame_bucket"] == 96


def test_main_prints_clean_omnirt_errors(monkeypatch, capsys) -> None:
    from omnirt.core.types import BackendUnavailableError

    def fake_generate(request, backend=None):
        raise BackendUnavailableError("cuda missing")

    monkeypatch.setattr("omnirt.cli.main.generate", fake_generate)

    exit_code = main(
        [
            "generate",
            "--task",
            "text2image",
            "--model",
            "sdxl-base-1.0",
            "--prompt",
            "hello",
            "--backend",
            "cuda",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "error: cuda missing" in captured.err
