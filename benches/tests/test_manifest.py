from __future__ import annotations

from pathlib import Path

import pytest

from benches._shared.errors import InvalidManifestError
from benches._shared.manifest import load_manifest


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_manifest_accepts_valid_rows(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", "path,label\na.wav,human\nb.wav,ai_generated\n")
    entries = load_manifest(manifest)
    assert [entry.label for entry in entries] == ["human", "ai_generated"]
    assert [entry.path.name for entry in entries] == ["a.wav", "b.wav"]


def test_manifest_resolves_relative_paths_against_manifest_dir(tmp_path: Path) -> None:
    nested = tmp_path / "clips"
    nested.mkdir()
    manifest = _write(tmp_path / "manifest.csv", "path,label\nclips/a.wav,human\n")
    entries = load_manifest(manifest)
    assert entries[0].path == (nested / "a.wav").resolve()
    assert entries[0].path.is_absolute()


def test_manifest_accepts_absolute_paths(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    manifest = _write(tmp_path / "manifest.csv", f"path,label\n{audio},human\n")
    assert load_manifest(manifest)[0].path == audio.resolve()


def test_manifest_ignores_extra_columns(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", "path,label,duration\na.wav,human,12.5\n")
    assert load_manifest(manifest)[0].label == "human"


def test_manifest_rejects_bad_label(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", "path,label\na.wav,robot\n")
    with pytest.raises(InvalidManifestError, match="line 2"):
        load_manifest(manifest)


def test_manifest_skips_urls_with_warning(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        "path,label\nhttps://example.com/a.mp3,ai_generated\nhttp://example.com/b.mp3,human\n",
    )
    with pytest.warns(UserWarning, match="URL"):
        entries = load_manifest(manifest)
    assert entries == []


def test_manifest_keeps_local_rows_when_skipping_urls(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path / "manifest.csv",
        "path,label\na.wav,human\nhttps://example.com/b.mp3,ai_generated\n",
    )
    with pytest.warns(UserWarning):
        entries = load_manifest(manifest)
    assert [entry.label for entry in entries] == ["human"]


def test_manifest_rejects_missing_header(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", "file,kind\na.wav,human\n")
    with pytest.raises(InvalidManifestError, match="path"):
        load_manifest(manifest)


def test_manifest_rejects_empty_path(tmp_path: Path) -> None:
    manifest = _write(tmp_path / "manifest.csv", "path,label\n,human\n")
    with pytest.raises(InvalidManifestError, match="line 2"):
        load_manifest(manifest)


def test_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidManifestError, match="Cannot open manifest"):
        load_manifest(tmp_path / "missing.csv")
