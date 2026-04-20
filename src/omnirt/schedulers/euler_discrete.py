"""Thin wrapper around Diffusers' Euler discrete scheduler."""

from __future__ import annotations

from typing import Any, Dict

from omnirt.core.types import DependencyUnavailableError


def build_scheduler(config: Dict[str, Any]):
    try:
        from diffusers import EulerDiscreteScheduler
    except ImportError as exc:
        raise DependencyUnavailableError("diffusers is required to build the EulerDiscreteScheduler.") from exc

    scheduler_config = config.get("scheduler_config")
    if scheduler_config:
        return EulerDiscreteScheduler.from_config(scheduler_config)
    return EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        interpolation_type="linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        timestep_spacing="leading",
        use_karras_sigmas=False,
    )

