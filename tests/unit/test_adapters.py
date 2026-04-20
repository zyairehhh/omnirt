from omnirt.core.adapters import AdapterManager
from omnirt.core.types import AdapterRef, DependencyUnavailableError


class FakeLoraPipeline:
    def __init__(self) -> None:
        self.loras = []
        self.fused = []

    def load_lora_weights(self, path: str) -> None:
        self.loras.append(path)

    def fuse_lora(self, lora_scale: float = 1.0) -> None:
        self.fused.append(lora_scale)


def test_adapter_manager_validates_and_applies_once(tmp_path, monkeypatch) -> None:
    adapter_path = tmp_path / "style.safetensors"
    adapter_path.write_bytes(b"fake")
    monkeypatch.setattr(
        "omnirt.core.weight_loader.WeightLoader.load",
        lambda self, path, device="cpu": (_ for _ in ()).throw(AssertionError("weights should not be eagerly loaded")),
    )

    manager = AdapterManager()
    manager.load_all([AdapterRef(kind="lora", path=str(adapter_path), scale=0.75)])
    pipeline = FakeLoraPipeline()

    manager.apply_to_pipeline(pipeline)

    assert pipeline.loras == [str(adapter_path)]
    assert pipeline.fused == [0.75]


def test_adapter_manager_rejects_pipeline_without_lora_support(tmp_path) -> None:
    adapter_path = tmp_path / "style.safetensors"
    adapter_path.write_bytes(b"fake")

    manager = AdapterManager()
    manager.load_all([AdapterRef(kind="lora", path=str(adapter_path), scale=1.0)])

    try:
        manager.apply_to_pipeline(object())
    except DependencyUnavailableError as exc:
        assert "LoRA" in str(exc)
    else:
        raise AssertionError("Expected DependencyUnavailableError")


def test_adapter_manager_accepts_hf_single_file_ref(monkeypatch, tmp_path) -> None:
    cached = tmp_path / "remote-style.safetensors"
    cached.write_bytes(b"fake")
    monkeypatch.setattr(
        "omnirt.core.weight_loader.WeightLoader.validate_path",
        lambda path: cached,
    )

    manager = AdapterManager()
    manager.load_all([AdapterRef(kind="lora", path="hf://acme/demo/weights/style.safetensors", scale=1.0)])
    pipeline = FakeLoraPipeline()

    manager.apply_to_pipeline(pipeline)

    assert pipeline.loras == [str(cached)]
