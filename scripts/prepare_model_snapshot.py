#!/usr/bin/env python3
"""Download a Hugging Face model snapshot into a local directory for offline OmniRT use."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a local Diffusers model snapshot so OmniRT can run without online downloads."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output-dir", required=True, help="Directory where the local snapshot should be stored.")
    parser.add_argument("--revision", help="Optional model revision or commit sha.")
    parser.add_argument("--allow-pattern", action="append", default=[], help="Optional allow pattern passed to snapshot_download. Repeatable.")
    parser.add_argument("--ignore-pattern", action="append", default=[], help="Optional ignore pattern passed to snapshot_download. Repeatable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("error: huggingface_hub is required to prepare local model snapshots.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
    )

    print(local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
