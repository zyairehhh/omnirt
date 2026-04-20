import pytest

from scripts.prepare_modelers_snapshot import build_modelers_git_url


def test_build_modelers_git_url_accepts_owner_and_model() -> None:
    assert (
        build_modelers_git_url("MindSpore-Lab/SDXL_Base1_0")
        == "https://modelers.cn/MindSpore-Lab/SDXL_Base1_0.git"
    )


def test_build_modelers_git_url_rejects_invalid_repo_id() -> None:
    with pytest.raises(ValueError):
        build_modelers_git_url("SDXL_Base1_0")
