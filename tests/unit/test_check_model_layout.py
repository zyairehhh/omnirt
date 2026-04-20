from pathlib import Path

from scripts.check_model_layout import check_layout


def test_check_layout_accepts_minimal_sdxl_directory(tmp_path: Path) -> None:
    for rel in [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/tokenizer_config.json",
        "text_encoder/config.json",
        "text_encoder_2/config.json",
        "unet/config.json",
        "vae/config.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert check_layout("sdxl", tmp_path) == 0


def test_check_layout_rejects_missing_required_paths(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")

    assert check_layout("svd", tmp_path) == 1


def test_check_layout_accepts_minimal_flux2_directory(tmp_path: Path) -> None:
    for rel in [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "vae/config.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    assert check_layout("flux2", tmp_path) == 0
