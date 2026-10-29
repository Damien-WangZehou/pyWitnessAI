"""Streaming fingerprints for local files."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["file_sha256"]


def file_sha256(path: str | Path) -> str:
    """Hash file bytes without loading a whole model or image into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
