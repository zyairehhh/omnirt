import importlib.util
import os

from omnirt.api import generate


def test_sdxl_cuda_smoke(tmp_path) -> None:
    try:
        import torch
    except ImportError:
        import pytest

        pytest.skip("torch is unavailable")

    if not importlib.util.find_spec("diffusers"):
        import pytest

        pytest.skip("diffusers is unavailable")
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is unavailable")

    model_source = os.getenv("OMNIRT_SDXL_MODEL_SOURCE")
    if not model_source:
        import pytest

        pytest.skip("OMNIRT_SDXL_MODEL_SOURCE is not set")

    result = generate(
        {
            "task": "text2image",
            "model": "sdxl-base-1.0",
            "backend": "cuda",
            "inputs": {"prompt": "a red lighthouse on a cliff"},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "width": 512,
                "height": 512,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".png")
