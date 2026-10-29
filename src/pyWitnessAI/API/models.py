"""Checksum-verified downloads for local model assets."""

from __future__ import annotations

from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from ..model_files import file_sha256

__all__ = ["download_model"]


def download_model(path: str | Path, url: str, expected_sha256: str) -> Path:
    """Return a verified cached model, or atomically download and verify it.

    Invalid existing files remain untouched. A failed download leaves neither a
    destination file nor a temporary download behind.
    """
    path = Path(path)
    if path.exists():
        if file_sha256(path) != expected_sha256:
            raise ValueError(f"Model checksum mismatch: {path}. Remove the invalid file and retry.")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with urlopen(url, timeout=60) as response, pending.open("wb") as output:
            shutil.copyfileobj(response, output)
        if file_sha256(pending) != expected_sha256:
            raise ValueError(f"Model download checksum mismatch: {path.name}")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path
