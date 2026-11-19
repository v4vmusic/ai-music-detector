from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from benches._shared.constants import LABELS, Label
from benches._shared.errors import InvalidManifestError

__all__ = ["ManifestEntry", "load_manifest"]


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One local audio path with its ground-truth label."""

    path: Path
    label: Label


def load_manifest(path: Path | str) -> list[ManifestEntry]:
    """Load ``path,label`` rows, resolving local paths against the manifest directory.

    URL rows are skipped with a warning; URL fetching is outside bench scope.
    """

    manifest_path = Path(path)
    try:
        handle = manifest_path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise InvalidManifestError(f"Cannot open manifest {manifest_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"path", "label"}.issubset(reader.fieldnames):
            raise InvalidManifestError(
                f"Manifest {manifest_path} must have a header containing 'path' and 'label'"
            )
        entries: list[ManifestEntry] = []
        for line_number, row in enumerate(reader, start=2):
            raw_path = (row.get("path") or "").strip()
            raw_label = (row.get("label") or "").strip()
            if not raw_path:
                raise InvalidManifestError(
                    f"Manifest {manifest_path}, line {line_number}: empty path"
                )
            if raw_label not in LABELS:
                raise InvalidManifestError(
                    f"Manifest {manifest_path}, line {line_number}: invalid label {raw_label!r}; "
                    f"expected one of {LABELS}"
                )
            if urlparse(raw_path).scheme.lower() in {"http", "https"}:
                warnings.warn(
                    f"Manifest {manifest_path}, line {line_number}: skipping URL row {raw_path!r}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            candidate = Path(raw_path)
            resolved = candidate if candidate.is_absolute() else manifest_path.parent / candidate
            entries.append(ManifestEntry(path=resolved.resolve(), label=cast(Label, raw_label)))
        return entries
