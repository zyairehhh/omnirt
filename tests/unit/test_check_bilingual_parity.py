"""Tests for scripts/check_bilingual_parity.py."""

from __future__ import annotations

from pathlib import Path

from scripts import check_bilingual_parity as parity


def _make_docs_tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    return docs


def test_pair_matched_exits_clean(monkeypatch, tmp_path: Path) -> None:
    docs = _make_docs_tree(tmp_path)
    zh = docs / "foo.md"
    en = docs / "foo.en.md"
    # Real zh:en character ratios sit around 1:2 to 1:3 for the OmniRT docs.
    # Craft a fixture that matches: ~210 zh chars, ~450 en chars (ratio ~2.1x).
    zh.write_text("中文正文包括一段典型长度的描述，覆盖架构说明与常见用法示例。" * 7, encoding="utf-8")
    en.write_text(
        "English mirror content describing the architecture and examples. " * 7,
        encoding="utf-8",
    )

    monkeypatch.setattr(parity, "DOCS_DIR", docs)
    monkeypatch.setattr(parity, "REPO_ROOT", tmp_path)

    failures, warnings = parity.check()
    assert failures == []
    assert warnings == []


def test_missing_en_sibling_reports_failure(monkeypatch, tmp_path: Path) -> None:
    docs = _make_docs_tree(tmp_path)
    (docs / "foo.md").write_text("中文\n", encoding="utf-8")

    monkeypatch.setattr(parity, "DOCS_DIR", docs)
    monkeypatch.setattr(parity, "REPO_ROOT", tmp_path)

    failures, _warnings = parity.check()
    assert len(failures) == 1
    assert "missing English sibling" in failures[0]
    assert "foo.en.md" in failures[0]


def test_length_ratio_too_small_reports_failure(monkeypatch, tmp_path: Path) -> None:
    docs = _make_docs_tree(tmp_path)
    (docs / "foo.md").write_text("中文内容非常长" * 200, encoding="utf-8")
    (docs / "foo.en.md").write_text("stub\n", encoding="utf-8")

    monkeypatch.setattr(parity, "DOCS_DIR", docs)
    monkeypatch.setattr(parity, "REPO_ROOT", tmp_path)

    failures, _warnings = parity.check()
    assert any("length mismatch" in msg for msg in failures)


def test_generated_dir_is_skipped(monkeypatch, tmp_path: Path) -> None:
    docs = _make_docs_tree(tmp_path)
    generated = docs / "_generated"
    generated.mkdir()
    # Only a zh file, no sibling; must not trip because _generated is skipped.
    (generated / "models.md").write_text("auto\n", encoding="utf-8")

    monkeypatch.setattr(parity, "DOCS_DIR", docs)
    monkeypatch.setattr(parity, "REPO_ROOT", tmp_path)

    failures, warnings = parity.check()
    assert failures == []
    assert warnings == []


def test_main_exits_nonzero_on_failure(monkeypatch, tmp_path: Path) -> None:
    docs = _make_docs_tree(tmp_path)
    (docs / "foo.md").write_text("中文\n", encoding="utf-8")

    monkeypatch.setattr(parity, "DOCS_DIR", docs)
    monkeypatch.setattr(parity, "REPO_ROOT", tmp_path)

    assert parity.main([]) == 1
