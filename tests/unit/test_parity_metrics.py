from pathlib import Path

from omnirt.core.parity import average_video_metrics, file_sha256, latent_statistics, psnr, ssim


def test_latent_statistics_reports_expected_keys() -> None:
    stats = latent_statistics([[0, 1], [2, 3]])

    assert set(stats) == {"mean", "std", "p50", "p95"}
    assert round(stats["mean"], 4) == 1.5


def test_image_similarity_metrics_are_high_for_close_images() -> None:
    left = [[0, 10], [20, 30]]
    right = [[0, 12], [18, 30]]

    assert psnr(left, right) > 35
    assert ssim(left, right) > 0.95


def test_average_video_metrics_aggregates_frame_scores() -> None:
    frames_a = [[[0, 1], [2, 3]], [[10, 11], [12, 13]]]
    frames_b = [[[0, 1], [2, 3]], [[10, 12], [12, 13]]]

    metrics = average_video_metrics(frames_a, frames_b)

    assert metrics["psnr_mean"] > 35
    assert metrics["ssim_mean"] > 0.95


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    file_path = tmp_path / "artifact.bin"
    file_path.write_bytes(b"omnirt")

    assert file_sha256(str(file_path)) == file_sha256(str(file_path))
