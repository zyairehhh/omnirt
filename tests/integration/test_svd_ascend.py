from PIL import Image

from tests.integration.conftest import require_local_model_dir, require_module
from omnirt.api import generate


def test_svd_ascend_smoke(tmp_path) -> None:
    require_module("torch_npu", "torch_npu is unavailable")
    require_module("diffusers", "diffusers is unavailable")
    model_source = require_local_model_dir("OMNIRT_SVD_XT_MODEL_SOURCE")

    image_path = tmp_path / "input.png"
    Image.new("RGB", (1024, 576), color="teal").save(image_path)

    result = generate(
        {
            "task": "image2video",
            "model": "svd-xt",
            "backend": "ascend",
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
