from pathlib import Path

from PIL import Image
import pytest

from omnirt.core.parity import average_video_metrics, file_sha256, latent_statistics, psnr


def test_latent_statistics_threshold_shape() -> None:
    cuda = [[0.0, 0.1], [0.2, 0.3]]
    ascend = [[0.01, 0.11], [0.19, 0.31]]

    left = latent_statistics(cuda)
    right = latent_statistics(ascend)

    assert abs(left["mean"] - right["mean"]) < 0.05
    assert abs(left["std"] - right["std"]) < 0.05


def test_image_psnr_matches_initial_threshold() -> None:
    left = Image.new("L", (16, 16), color=128)
    right = Image.new("L", (16, 16), color=130)

    assert psnr(left, right) >= 28


def test_video_psnr_mean_matches_initial_threshold() -> None:
    reference = [Image.new("L", (16, 16), color=100), Image.new("L", (16, 16), color=120)]
    candidate = [Image.new("L", (16, 16), color=101), Image.new("L", (16, 16), color=121)]

    metrics = average_video_metrics(reference, candidate)

    assert metrics["psnr_mean"] >= 26


def test_golden_hash_manifest_round_trip(tmp_path: Path) -> None:
    artifact = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color="white").save(artifact)

    assert len(file_sha256(str(artifact))) == 64
