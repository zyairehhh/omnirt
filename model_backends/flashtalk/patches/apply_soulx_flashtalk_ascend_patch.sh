#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${PATCH_DIR}/soulx-flashtalk-ascend-omnirt.patch"

usage() {
  cat <<'USAGE'
Apply OmniRT SoulX-FlashTalk Ascend patch to a Git checkout.

Usage (from OmniRT repository root, default SoulX layout):
  bash model_backends/flashtalk/patches/apply_soulx_flashtalk_ascend_patch.sh .omnirt/model-repos/SoulX-FlashTalk

The argument must be the SoulX repository root (directory containing flash_talk/), as a path relative to the current directory or absolute.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -ne 1 ]]; then
  usage
  [[ $# -eq 1 ]] || exit 2
fi

REPO="$(cd "$1" && pwd)"
if [[ ! -d "$REPO/flash_talk" ]]; then
  echo "error: flash_talk/ not found under: $REPO" >&2
  exit 2
fi
if [[ ! -f "$PATCH_FILE" ]]; then
  echo "error: patch missing: $PATCH_FILE" >&2
  exit 2
fi

if [[ ! -d "$REPO/.git" ]]; then
  echo "error: not a git repository (need .git to apply patch): $REPO" >&2
  exit 2
fi

if git -C "$REPO" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
  echo "patch already applied (reverse check ok), skip"
  exit 0
fi

git -C "$REPO" apply --check "$PATCH_FILE"
git -C "$REPO" apply "$PATCH_FILE"
echo "applied: $PATCH_FILE -> $REPO"
