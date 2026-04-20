"""Command line interface for OmniRT."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from omnirt.api import generate
from omnirt.core.types import GenerateRequest, OmniRTError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnirt", description="OmniRT command line interface.")
    subparsers = parser.add_subparsers(dest="command")

    generate_parser = subparsers.add_parser("generate", help="Run a generation request.")
    generate_parser.add_argument("--config", help="Path to a YAML or JSON request file.")
    generate_parser.add_argument("--task", choices=["text2image", "image2video"], help="Task to run.")
    generate_parser.add_argument("--model", help="Model registry id to execute.")
    generate_parser.add_argument("--backend", choices=["auto", "cuda", "ascend"], help="Override backend selection.")
    generate_parser.add_argument("--prompt", help="Prompt for text2image generation.")
    generate_parser.add_argument("--negative-prompt", help="Negative prompt for text2image generation.")
    generate_parser.add_argument("--image", help="Input image for image2video generation.")
    generate_parser.add_argument("--num-frames", type=int, help="Frame count for image2video generation.")
    generate_parser.add_argument("--fps", type=int, help="Frames per second for exported video.")
    generate_parser.add_argument("--frame-bucket", type=int, help="Motion bucket hint for SVD image2video.")
    generate_parser.add_argument("--decode-chunk-size", type=int, help="Decode chunk size for video generation.")
    generate_parser.add_argument("--noise-aug-strength", type=float, help="Noise augmentation for SVD image2video.")
    generate_parser.add_argument("--num-inference-steps", type=int, help="Number of denoising steps.")
    generate_parser.add_argument("--guidance-scale", type=float, help="Classifier-free guidance scale.")
    generate_parser.add_argument("--seed", type=int, help="Random seed.")
    generate_parser.add_argument("--width", type=int, help="Output width for image generation.")
    generate_parser.add_argument("--height", type=int, help="Output height for image generation.")
    generate_parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], help="Computation dtype.")
    generate_parser.add_argument("--output-dir", help="Output directory for saved artifacts.")
    generate_parser.add_argument("--model-path", help="Override the default model source.")
    generate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the GenerateResult as compact JSON to stdout.",
    )

    return parser


def request_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> GenerateRequest:
    if args.config:
        return GenerateRequest.from_file(args.config)

    if not args.task or not args.model:
        parser.error("either --config or both --task and --model are required")

    inputs = {}
    if args.task == "text2image":
        if not args.prompt:
            parser.error("--prompt is required for --task text2image")
        inputs["prompt"] = args.prompt
        if args.negative_prompt:
            inputs["negative_prompt"] = args.negative_prompt
    else:
        if not args.image:
            parser.error("--image is required for --task image2video")
        inputs["image"] = args.image
        if args.num_frames is not None:
            inputs["num_frames"] = args.num_frames
        if args.fps is not None:
            inputs["fps"] = args.fps

    config = {}
    for field in (
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "width",
        "height",
        "dtype",
        "output_dir",
        "frame_bucket",
        "decode_chunk_size",
        "noise_aug_strength",
    ):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.model_path:
        config["model_path"] = args.model_path

    return GenerateRequest(
        task=args.task,
        model=args.model,
        backend=args.backend or "auto",
        inputs=inputs,
        config=config,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "generate":
        parser.print_help()
        return 0

    request = request_from_args(args, parser)
    try:
        result = generate(request, backend=args.backend)
    except (OmniRTError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0
