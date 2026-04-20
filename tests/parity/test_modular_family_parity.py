from __future__ import annotations

from PIL import Image
import pytest

from omnirt.core.parity import check_image_parity, check_video_parity


def _solid_image(value: int) -> Image.Image:
    return Image.new("L", (32, 32), color=value)


@pytest.mark.parametrize(
    ("family", "reference_color", "candidate_color"),
    [
        ("sdxl", 120, 121),
        ("flux", 90, 91),
        ("flux2", 150, 151),
    ],
)
def test_modular_family_image_parity_thresholds(family: str, reference_color: int, candidate_color: int) -> None:
    metrics = check_image_parity(_solid_image(reference_color), _solid_image(candidate_color), min_psnr=40.0, min_ssim=0.99)

    assert metrics["ok"] is True, family


def test_wan_video_family_parity_thresholds() -> None:
    reference = [_solid_image(100), _solid_image(140)]
    candidate = [_solid_image(101), _solid_image(141)]

    metrics = check_video_parity(reference, candidate, min_psnr_mean=40.0, min_ssim_mean=0.99)

    assert metrics["ok"] is True
