#!/usr/bin/env python3
"""Clone a model repository from Modelers into a local directory for offline OmniRT use."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def build_modelers_git_url(repo_id: str) -> str:
    normalized = repo_id.strip().strip("/")
    if not normalized or "/" not in normalized:
        raise ValueError("repo_id must look like <owner>/<model_name>")
    return f"https://modelers.cn/{normalized}.git"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone a Modelers model repository so OmniRT can use it from a local directory."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Modelers repo id, for example MindSpore-Lab/SDXL_Base1_0",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the local clone should be stored.")
    parser.add_argument("--revision", help="Optional git branch, tag, or commit to checkout after clone.")
    parser.add_argument(
        "--no-shallow",
        action="store_true",
        help="Disable shallow clone and fetch full history.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="If output-dir already contains a git checkout, fetch the requested revision instead of failing.",
    )
    return parser


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if shutil.which("git") is None:
        print("error: git is required to clone Modelers repositories.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_url = build_modelers_git_url(args.repo_id)

    if output_dir.exists():
        if not args.update:
            print(f"error: output directory already exists: {output_dir}", file=sys.stderr)
            print("hint: pass --update to refresh an existing checkout.", file=sys.stderr)
            return 2
        if not (output_dir / ".git").exists():
            print(f"error: output directory exists but is not a git checkout: {output_dir}", file=sys.stderr)
            return 2

        run(["git", "fetch", "--all", "--tags"], cwd=output_dir)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_cmd = ["git", "clone"]
        if not args.no_shallow and not args.revision:
            clone_cmd.extend(["--depth", "1"])
        clone_cmd.extend([repo_url, str(output_dir)])
        run(clone_cmd)

    if args.revision:
        run(["git", "checkout", args.revision], cwd=output_dir)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
