#!/usr/bin/env python3
"""Prepare a ModelScope repository for offline OmniRT use."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def build_modelscope_git_url(repo_id: str) -> str:
    normalized = repo_id.strip().strip("/")
    if not normalized or "/" not in normalized:
        raise ValueError("repo_id must look like <owner>/<model_name>")
    return f"https://www.modelscope.cn/models/{normalized}.git"


def build_modelscope_resolve_url(repo_id: str, revision: str, file_path: str) -> str:
    normalized = repo_id.strip().strip("/")
    relative_path = file_path.strip().lstrip("/")
    if not normalized or "/" not in normalized:
        raise ValueError("repo_id must look like <owner>/<model_name>")
    if not relative_path:
        raise ValueError("file_path must not be empty")
    return f"https://www.modelscope.cn/models/{normalized}/resolve/{revision}/{relative_path}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone a ModelScope model repository and optionally fetch selected large files."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="ModelScope repo id, for example ai-modelscope/stable-video-diffusion-img2vid-xt",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where the local checkout should be stored.")
    parser.add_argument("--revision", default="master", help="Git branch, tag, or commit to checkout.")
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
    parser.add_argument(
        "--download-file",
        action="append",
        default=[],
        help="Relative file path to fetch through the ModelScope resolve endpoint. May be passed multiple times.",
    )
    return parser


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("wget") is not None:
        run(["wget", "-c", "-O", str(destination), url])
        return
    if shutil.which("curl") is not None:
        run(["curl", "-L", "-C", "-", "--output", str(destination), url])
        return
    raise RuntimeError("either wget or curl is required to download ModelScope files")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if shutil.which("git") is None:
        print("error: git is required to clone ModelScope repositories.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_url = build_modelscope_git_url(args.repo_id)

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
        if not args.no_shallow and args.revision in {"master", "main"}:
            clone_cmd.extend(["--depth", "1"])
        clone_cmd.extend([repo_url, str(output_dir)])
        run(clone_cmd)

    run(["git", "checkout", args.revision], cwd=output_dir)

    for relative_path in args.download_file:
        url = build_modelscope_resolve_url(args.repo_id, args.revision, relative_path)
        _download_file(url, output_dir / relative_path)

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
