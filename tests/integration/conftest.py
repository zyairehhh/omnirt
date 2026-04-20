from pathlib import Path
import importlib.util
import os

import pytest


def require_module(name: str, reason: str) -> None:
    if not importlib.util.find_spec(name):
        pytest.skip(reason)


def require_local_model_dir(*env_vars: str) -> str:
    if not env_vars:
        raise ValueError("At least one environment variable name is required.")

    for env_var in env_vars:
        model_source = os.getenv(env_var)
        if not model_source:
            continue

        path = Path(model_source).expanduser()
        if not path.exists():
            pytest.skip(f"{env_var} does not exist locally: {path}")
        if not path.is_dir():
            pytest.skip(f"{env_var} must point to a local model directory: {path}")
        return str(path)

    joined = ", ".join(env_vars)
    pytest.skip(f"None of the model directories are set: {joined}")
