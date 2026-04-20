import importlib.util
import os

from omnirt.api import generate


def test_sdxl_ascend_smoke(tmp_path) -> None:
    if not importlib.util.find_spec("torch_npu"):
        import pytest

        pytest.skip("torch_npu is unavailable")
    if not importlib.util.find_spec("diffusers"):
        import pytest

        pytest.skip("diffusers is unavailable")

    model_source = os.getenv("OMNIRT_SDXL_MODEL_SOURCE")
    if not model_source:
        import pytest

        pytest.skip("OMNIRT_SDXL_MODEL_SOURCE is not set")

    result = generate(
        {
            "task": "text2image",
            "model": "sdxl-base-1.0",
            "backend": "ascend",
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
