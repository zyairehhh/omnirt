"""Helpers for cross-backend parity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

from omnirt.core.types import DependencyUnavailableError


def _to_array(value):
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyUnavailableError("numpy is required for parity metrics.") from exc
    return np.asarray(value, dtype=np.float32)


def latent_statistics(latents) -> dict:
    array = _to_array(latents)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(_percentile(array, 50)),
        "p95": float(_percentile(array, 95)),
    }


def psnr(reference, candidate) -> float:
    reference_array = _to_array(reference)
    candidate_array = _to_array(candidate)
    mse = float(((reference_array - candidate_array) ** 2).mean())
    if mse == 0.0:
        return float("inf")
    return float(20.0 * _log10(255.0 / (mse ** 0.5)))


def ssim(reference, candidate) -> float:
    ref = _to_array(reference)
    cand = _to_array(candidate)

    c1 = 6.5025
    c2 = 58.5225
    mu_x = ref.mean()
    mu_y = cand.mean()
    sigma_x = ref.var()
    sigma_y = cand.var()
    sigma_xy = ((ref - mu_x) * (cand - mu_y)).mean()
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator) if denominator else 1.0


def average_video_metrics(reference_frames: Sequence[object], candidate_frames: Sequence[object]) -> dict:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError("Video comparisons require frame counts to match.")

    psnr_scores = [psnr(left, right) for left, right in zip(reference_frames, candidate_frames)]
    ssim_scores = [ssim(left, right) for left, right in zip(reference_frames, candidate_frames)]
    return {
        "psnr_mean": float(sum(psnr_scores) / len(psnr_scores)),
        "ssim_mean": float(sum(ssim_scores) / len(ssim_scores)),
    }


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(array, value: int) -> float:
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyUnavailableError("numpy is required for parity metrics.") from exc
    return float(np.percentile(array, value))


def _log10(value: float) -> float:
    try:
        import math
    except ImportError as exc:
        raise DependencyUnavailableError("math is unavailable.") from exc
    return math.log10(value)
