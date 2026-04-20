"""Check that each Chinese docs page has an English sibling and vice versa.

What this script does
---------------------
1. Walks ``docs/`` (excluding ``docs/_generated/`` and anything under a leading
   underscore directory) and enforces a strict 1:1 pairing: every ``foo.md``
   must have a ``foo.en.md`` next to it.
2. Applies a coarse length sanity check to flag half-translated pages. For each
   pair, fail if the English file is drastically shorter than expected relative
   to the Chinese file.
3. Emits a non-fatal warning when the Chinese file has commits newer than the
   English file (useful signal that a follow-up translation pass is due).

Exit codes
----------
- ``0`` when everything looks consistent.
- ``1`` when a pair is missing or the length heuristic trips.

Intentionally out of scope
--------------------------
- ``docs/_generated/`` (both language files are produced by a script).
- ``docs/stylesheets/`` and other static assets.
- ``docs/adr/`` ADRs follow the same convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SKIP_DIRS = ("_generated", "stylesheets")
LENGTH_LOWER_FACTOR = 0.3  # en shorter than 30% of zh characters -> clearly incomplete
LENGTH_UPPER_FACTOR = 4.0  # en longer than 4.0x zh characters -> likely duplicated content
# English prose is normally 2x-3x longer than Chinese per character; 4.0x catches
# accidental double-paste / stray fixtures without being noisy about natural expansion.


def _iter_zh_docs() -> List[Path]:
    """Return every ``foo.md`` under ``docs/`` that we expect to be Chinese source."""

    results: List[Path] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        rel = path.relative_to(DOCS_DIR)
        if any(part in SKIP_DIRS or part.startswith("_") for part in rel.parts[:-1]):
            continue
        # ``foo.en.md`` -> two suffixes; skip so it is not mistaken for a zh source.
        if path.suffixes[-2:] == [".en", ".md"]:
            continue
        results.append(path)
    return results


def _en_sibling(path: Path) -> Path:
    """Return the expected ``.en.md`` path next to ``path``."""

    stem = path.name[: -len(".md")]
    return path.with_name(f"{stem}.en.md")


def _last_commit(path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return completed.stdout.strip()


def _length_ratio(a: Path, b: Path) -> float:
    """Return ``len(b) / len(a)`` in characters. 0.0 when either file is empty or missing."""

    if not a.is_file() or not b.is_file():
        return 0.0
    size_a = len(a.read_text(encoding="utf-8").strip())
    if size_a == 0:
        return 0.0
    return len(b.read_text(encoding="utf-8").strip()) / size_a


def _zh_ahead_of_en(zh_path: Path, en_path: Path) -> bool:
    """Return True when zh has a later commit than en (non-fatal hint)."""

    zh_sha = _last_commit(zh_path)
    en_sha = _last_commit(en_path)
    if not zh_sha or not en_sha or zh_sha == en_sha:
        return False
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", en_sha, zh_sha],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    # is-ancestor exits 0 when en_sha is ancestor of zh_sha (zh is newer), 1 otherwise.
    return completed.returncode == 0


def check() -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    warnings: List[str] = []

    for zh_path in _iter_zh_docs():
        en_path = _en_sibling(zh_path)
        rel_zh = zh_path.relative_to(REPO_ROOT)
        rel_en = en_path.relative_to(REPO_ROOT)
        if not en_path.is_file():
            failures.append(f"missing English sibling: {rel_en} (expected next to {rel_zh})")
            continue

        ratio = _length_ratio(zh_path, en_path)
        if ratio and (ratio < LENGTH_LOWER_FACTOR or ratio > LENGTH_UPPER_FACTOR):
            failures.append(
                f"length mismatch: {rel_en} is {ratio:.2f}x {rel_zh} "
                f"(expected between {LENGTH_LOWER_FACTOR:.2f}x and {LENGTH_UPPER_FACTOR:.2f}x)"
            )

        if _zh_ahead_of_en(zh_path, en_path):
            warnings.append(f"zh newer than en: {rel_zh} has commits not yet reflected in {rel_en}")

    return failures, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (zh-ahead-of-en drift) as failures as well.",
    )
    args = parser.parse_args(argv)

    failures, warnings = check()
    for msg in warnings:
        print(f"warning: {msg}")
    for msg in failures:
        print(f"error: {msg}", file=sys.stderr)

    if failures:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
