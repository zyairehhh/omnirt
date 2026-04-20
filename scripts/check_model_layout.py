#!/usr/bin/env python3
"""Validate whether a local model directory matches OmniRT's current Diffusers layout expectations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


LAYOUTS = {
    "sdxl": {
        "required_files": ["model_index.json"],
        "required_dirs": [
            "scheduler",
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "unet",
            "vae",
        ],
        "recommended_files": [
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "vae/config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
        ],
    },
    "svd": {
        "required_files": ["model_index.json"],
        "required_dirs": [
            "feature_extractor",
            "image_encoder",
            "scheduler",
            "unet",
            "vae",
        ],
        "recommended_files": [
            "scheduler/scheduler_config.json",
            "unet/config.json",
            "vae/config.json",
            "image_encoder/config.json",
        ],
    },
    "flux2": {
        "required_files": ["model_index.json"],
        "required_dirs": [
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        ],
        "recommended_files": [
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "vae/config.json",
        ],
    },
    "wan_t2v": {
        "required_files": ["model_index.json"],
        "required_dirs": [
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "transformer_2",
            "vae",
        ],
        "recommended_files": [
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "transformer_2/config.json",
            "vae/config.json",
        ],
    },
    "wan_i2v": {
        "required_files": ["model_index.json"],
        "required_dirs": [
            "image_encoder",
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "transformer_2",
            "vae",
        ],
        "recommended_files": [
            "scheduler/scheduler_config.json",
            "image_encoder/config.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "transformer_2/config.json",
            "vae/config.json",
        ],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether a local model directory is ready for OmniRT.")
    parser.add_argument("--task", choices=sorted(LAYOUTS), required=True, help="Layout profile to validate.")
    parser.add_argument("--model-dir", required=True, help="Local model directory to inspect.")
    return parser


def check_layout(task: str, model_dir: Path) -> int:
    profile = LAYOUTS[task]
    errors: list[str] = []
    warnings: list[str] = []

    if not model_dir.exists():
        print(f"error: model directory does not exist: {model_dir}", file=sys.stderr)
        return 2
    if not model_dir.is_dir():
        print(f"error: model path is not a directory: {model_dir}", file=sys.stderr)
        return 2

    for rel_path in profile["required_files"]:
        if not (model_dir / rel_path).is_file():
            errors.append(f"missing required file: {rel_path}")

    for rel_path in profile["required_dirs"]:
        if not (model_dir / rel_path).is_dir():
            errors.append(f"missing required directory: {rel_path}")

    for rel_path in profile["recommended_files"]:
        if not (model_dir / rel_path).exists():
            warnings.append(f"missing recommended file: {rel_path}")

    print(f"task={task}")
    print(f"model_dir={model_dir}")

    if errors:
        print("status=invalid")
        for item in errors:
            print(f"error: {item}")
    else:
        print("status=ok")

    for item in warnings:
        print(f"warning: {item}")

    if not errors:
        print("hint: this directory looks compatible with OmniRT's current Diffusers-based loader.")
        return 0

    print("hint: if this is a MindSpore or custom-export layout, conversion or a new loader adapter may be required.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return check_layout(args.task, Path(args.model_dir).expanduser().resolve())


if __name__ == "__main__":
    raise SystemExit(main())
