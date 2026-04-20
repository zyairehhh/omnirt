from pathlib import Path

import pytest

from scripts.prepare_modelscope_snapshot import build_modelscope_git_url, build_modelscope_resolve_url


def test_build_modelscope_git_url() -> None:
    assert (
        build_modelscope_git_url("ai-modelscope/stable-video-diffusion-img2vid-xt")
        == "https://www.modelscope.cn/models/ai-modelscope/stable-video-diffusion-img2vid-xt.git"
    )


def test_build_modelscope_resolve_url() -> None:
    assert (
        build_modelscope_resolve_url(
            "ai-modelscope/stable-video-diffusion-img2vid-xt",
            "master",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        )
        == "https://www.modelscope.cn/models/ai-modelscope/stable-video-diffusion-img2vid-xt/resolve/master/unet/diffusion_pytorch_model.fp16.safetensors"
    )


@pytest.mark.parametrize("repo_id", ["", "bad-format", "/"])
def test_build_modelscope_git_url_rejects_invalid_repo_id(repo_id: str) -> None:
    with pytest.raises(ValueError):
        build_modelscope_git_url(repo_id)


def test_build_modelscope_resolve_url_rejects_empty_file_path() -> None:
    with pytest.raises(ValueError):
        build_modelscope_resolve_url("ai-modelscope/stable-video-diffusion-img2vid-xt", "master", "")
