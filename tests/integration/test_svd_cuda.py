import importlib.util
import os

from PIL import Image

from omnirt.api import generate


def test_svd_cuda_smoke(tmp_path) -> None:
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

    model_source = os.getenv("OMNIRT_SVD_MODEL_SOURCE")
    if not model_source:
        import pytest

        pytest.skip("OMNIRT_SVD_MODEL_SOURCE is not set")

    image_path = tmp_path / "input.png"
    Image.new("RGB", (1024, 576), color="teal").save(image_path)

    result = generate(
        {
            "task": "image2video",
            "model": "svd-xt",
            "backend": "cuda",
            "inputs": {"image": str(image_path), "num_frames": 4, "fps": 7},
            "config": {
                "model_path": model_source,
                "output_dir": str(tmp_path),
                "num_inference_steps": 2,
                "decode_chunk_size": 2,
            },
        }
    )

    assert result.outputs
    assert result.outputs[0].path.endswith(".mp4")
