"""Path helpers for OmniRT-managed model runtimes."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def omnirt_home() -> Path:
    value = os.environ.get("OMNIRT_HOME")
    if value and value.strip():
        return Path(value).expanduser().resolve()
    return (project_root() / ".omnirt").resolve()


def set_omnirt_home(value: str | Path | None) -> None:
    """Override the runtime home for this process."""
    if value is None:
        return
    os.environ["OMNIRT_HOME"] = str(Path(value).expanduser().resolve())


def runtime_state_dir(name: str, device: str) -> Path:
    return omnirt_home() / "runtimes" / name / device


def expand_path_template(value: str, *, name: str, device: str) -> Path:
    text = (
        value.replace("${OMNIRT_HOME}", str(omnirt_home()))
        .replace("${OMNIRT_ROOT}", str(project_root()))
        .replace("${name}", name)
        .replace("${device}", device)
    )
    return Path(text).expanduser().resolve()
