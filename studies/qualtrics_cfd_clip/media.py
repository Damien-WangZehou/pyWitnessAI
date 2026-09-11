from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from pyWitnessAI.model_files import file_sha256


MEDIA_MANIFEST_HASH_VERSION = "ordered_media_manifest_content_sha256_v1"


def resolve_cfd_image_path(
    raw_path: Any,
    *,
    repository_root: str | Path,
    image_root: str | Path,
) -> Path:
    """Resolve a cached CFD path portably and keep it inside image_root."""

    repository = Path(repository_root).resolve()
    allowed_root = Path(image_root).resolve()
    normalised = str(raw_path).replace("\\", "/")
    candidate = Path(normalised)
    if not candidate.is_absolute():
        candidate = repository / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            f"Manifest image path falls outside the configured CFD image root: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Manifest CFD image is missing: {resolved}")
    return resolved


def media_manifest_integrity(
    manifest: pd.DataFrame,
    *,
    repository_root: str | Path,
    image_root: str | Path,
) -> tuple[str, dict[str, Path], dict[str, str]]:
    """Bind ordered candidate IDs to displayed paths and exact image bytes."""

    required = {"image_id", "target_id", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Media manifest is missing required columns: {missing}")

    allowed_root = Path(image_root).resolve()
    records: list[dict[str, str]] = []
    paths: dict[str, Path] = {}
    content_hashes: dict[str, str] = {}
    for row in manifest[["image_id", "target_id", "image_path"]].itertuples(index=False):
        image_id = str(row.image_id)
        target_id = str(row.target_id)
        if image_id in paths:
            raise ValueError("Media manifest contains duplicate image_id values.")
        path = resolve_cfd_image_path(
            row.image_path,
            repository_root=repository_root,
            image_root=allowed_root,
        )
        content_hash = file_sha256(path)
        records.append(
            {
                "image_id": image_id,
                "target_id": target_id,
                "relative_path": path.relative_to(allowed_root).as_posix(),
                "content_sha256": content_hash,
            }
        )
        paths[image_id] = path
        content_hashes[image_id] = content_hash

    canonical = json.dumps(
        {
            "version": MEDIA_MANIFEST_HASH_VERSION,
            "records": records,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), paths, content_hashes


__all__ = [
    "MEDIA_MANIFEST_HASH_VERSION",
    "file_sha256",
    "media_manifest_integrity",
    "resolve_cfd_image_path",
]
