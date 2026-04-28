from pathlib import Path

from omnirt.backends.base import BackendRuntime
from omnirt.core.registry import ModelSpec, get_model
from omnirt.core.types import GenerateRequest
from omnirt.models import ensure_registered
from omnirt.models.flashhead.pipeline import FlashHeadPipeline


class FakeAscendRuntime(BackendRuntime):
    name = "ascend"
    device_name = "npu"

    def is_available(self) -> bool:
        return True

    def capabilities(self):
        raise NotImplementedError

    def _compile(self, module, tag):
        return module

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> dict:
        return {"peak_mb": 64.0}

    def available_memory_gb(self):
        return 128.0


class FakeCudaRuntime(FakeAscendRuntime):
    name = "cuda"
    device_name = "cuda"


def build_model_spec() -> ModelSpec:
    return ModelSpec(
        id="soulx-flashhead-1.3b",
        task="audio2video",
        pipeline_cls=FlashHeadPipeline,
        default_backend="ascend",
        resource_hint={"min_vram_gb": 48, "vram_scope": "aggregate", "dtype": "bf16"},
    )


def test_flashhead_model_is_registered() -> None:
    ensure_registered()

    spec = get_model("soulx-flashhead-1.3b")

    assert spec.task == "audio2video"
    assert spec.default_backend == "ascend"
    assert spec.capabilities.default_config["sample_steps"] == 2
    assert "vae_2d_split" in spec.capabilities.supported_config
    assert "latent_carry" in spec.capabilities.supported_config


def test_flashhead_pipeline_launches_external_script(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-FlashHead"
    ckpt_dir = repo_path / "models" / "SoulX-FlashHead-1_3B"
    wav2vec_dir = repo_path / "models" / "wav2vec2-base-960h"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    (repo_path / "generate_video.py").write_text("print('stub')\n", encoding="utf-8")
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    env_script = tmp_path / "set_env.sh"
    env_script.write_text("export ASCEND=1\n", encoding="utf-8")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)

    captured = {}

    def fake_run(cmd, check, cwd, env, stdout, stderr, text):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        shell_command = cmd[-1]
        output_path = Path(shell_command.split("--save_file", 1)[1].split()[0].strip("'\""))
        output_path.write_bytes(b"video")

        class Completed:
            stdout = "ok"

        return Completed()

    monkeypatch.setattr("omnirt.launcher.base.subprocess.run", fake_run)
    monkeypatch.setattr("omnirt.models.flashhead.pipeline.probe_video_file", lambda path: (512, 512, 250))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashhead-1.3b",
        backend="ascend",
        inputs={"image": str(image_path), "audio": str(audio_path)},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashHead-1_3B",
            "wav2vec_dir": "models/wav2vec2-base-960h",
            "output_dir": str(tmp_path / "outputs"),
            "ascend_env_script": str(env_script),
            "python_executable": str(python_executable),
            "launcher": "torchrun",
            "nproc_per_node": 4,
            "visible_devices": "2,3,4,5",
            "model_type": "pro",
            "audio_encode_mode": "stream",
            "sample_steps": 2,
            "vae_2d_split": True,
            "latent_carry": False,
            "npu_fusion_attention": True,
            "profile": True,
        },
    )

    pipeline = FlashHeadPipeline(runtime=FakeAscendRuntime(), model_spec=build_model_spec())
    result = pipeline.run(request)

    assert Path(result.outputs[0].path).exists()
    assert result.outputs[0].width == 512
    assert result.outputs[0].height == 512
    assert result.outputs[0].num_frames == 250
    assert captured["cmd"][:2] == ["bash", "-lc"]
    assert "torch.distributed.run" in captured["cmd"][2]
    assert "--model_type pro" in captured["cmd"][2]
    assert "--audio_encode_mode stream" in captured["cmd"][2]
    assert captured["cwd"] == str(repo_path)
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "2,3,4,5"
    assert captured["env"]["FLASHHEAD_DEVICE"] == "npu"
    assert captured["env"]["FLASHHEAD_SAMPLE_STEPS"] == "2"
    assert captured["env"]["FLASHHEAD_VAE_2D_SPLIT"] == "1"
    assert captured["env"]["FLASHHEAD_LATENT_CARRY"] == "0"
    assert captured["env"]["FLASHHEAD_NPU_FUSION_ATTENTION"] == "1"
    assert captured["env"]["FLASHHEAD_PROFILE"] == "1"


def test_flashhead_pipeline_defaults_device_from_runtime(tmp_path, monkeypatch) -> None:
    repo_path = tmp_path / "SoulX-FlashHead"
    ckpt_dir = repo_path / "models" / "SoulX-FlashHead-1_3B"
    wav2vec_dir = repo_path / "models" / "wav2vec2-base-960h"
    repo_path.mkdir()
    ckpt_dir.mkdir(parents=True)
    wav2vec_dir.mkdir(parents=True)
    (repo_path / "generate_video.py").write_text("print('stub')\n", encoding="utf-8")
    image_path = tmp_path / "speaker.png"
    audio_path = tmp_path / "voice.wav"
    image_path.write_bytes(b"fake")
    audio_path.write_bytes(b"fake")
    python_executable = tmp_path / "python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)

    captured = {}

    def fake_launch(self, *, cwd, command, env, env_script):
        del self
        del cwd, env_script
        save_file = Path(command[command.index("--save_file") + 1])
        save_file.write_bytes(b"video")
        captured["env"] = env

        class Completed:
            stdout = "ok"

        return Completed()

    monkeypatch.setattr("omnirt.launcher.base.Launcher.launch", fake_launch)
    monkeypatch.setattr("omnirt.models.flashhead.pipeline.probe_video_file", lambda path: (512, 512, 250))

    request = GenerateRequest(
        task="audio2video",
        model="soulx-flashhead-1.3b",
        backend="cuda",
        inputs={"image": str(image_path), "audio": str(audio_path)},
        config={
            "repo_path": str(repo_path),
            "ckpt_dir": "models/SoulX-FlashHead-1_3B",
            "wav2vec_dir": "models/wav2vec2-base-960h",
            "output_dir": str(tmp_path / "outputs"),
            "python_executable": str(python_executable),
            "launcher": "python",
        },
    )

    pipeline = FlashHeadPipeline(runtime=FakeCudaRuntime(), model_spec=build_model_spec())
    pipeline.run(request)

    assert captured["env"]["FLASHHEAD_DEVICE"] == "cuda"
