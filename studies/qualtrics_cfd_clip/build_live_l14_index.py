"""Build and verify the frozen Qualtrics CFD White-Male-Neutral index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "studies.qualtrics_cfd_clip"


import numpy as np
import pandas as pd

from .cfd import build_cfd_manifest
from pyWitnessAI.clip import (
    ClipIndex,
    SentenceTransformerClipEncoder,
)
from .media import media_manifest_integrity
from .profile import DEFAULT_PROFILE_PATH, LiveSelectionProfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser(*, repository_root: str | Path | None = None) -> argparse.ArgumentParser:
    root = Path(repository_root) if repository_root is not None else PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Build the frozen ViT-L-14 CFD White-Male-Neutral live index."
    )
    parser.add_argument(
        "--manifest", type=Path,
        help="Existing CFD manifest. Omit to build it from --image-root and --metadata.",
    )
    parser.add_argument(
        "--metadata", type=Path,
        default=root / "data" / "CFD Version 3.0" / "CFD 3.0 Norming Data and Codebook.xlsx",
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "build" / "qualtrics_cfd_clip" / "clip_index_vit_l_14",
    )
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument(
        "--image-root", type=Path,
        default=root / "data" / "CFD Version 3.0" / "Images",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *, repository_root: str | Path | None = None,
) -> int:
    root = Path(repository_root) if repository_root is not None else PROJECT_ROOT
    args = build_parser(repository_root=root).parse_args(argv)
    profile = LiveSelectionProfile.load(args.profile)
    manifest = (
        pd.read_csv(args.manifest, low_memory=False)
        if args.manifest is not None
        else build_cfd_manifest(args.image_root, args.metadata, neutral_only=True)
    )
    # Reject an invalid pool before loading model weights.
    pool = _live_pool(manifest)
    encoder = SentenceTransformerClipEncoder(
        model_name=profile.model_name,
        model_revision=profile.model_revision,
        local_model_path=args.local_model_path,
        device=args.device,
    )
    audit = build_live_index(
        pool, encoder, output_dir=args.output_dir, image_root=args.image_root,
        repository_root=root, source_manifest=args.manifest, profile=profile,
        batch_size=args.batch_size, show_progress=args.show_progress,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


def build_live_index(
    manifest: pd.DataFrame,
    encoder,
    *,
    output_dir: str | Path,
    image_root: str | Path,
    repository_root: str | Path | None = None,
    source_manifest: str | Path | None = None,
    profile: LiveSelectionProfile,
    batch_size: int = 8,
    show_progress: bool = False,
) -> dict[str, object]:
    """Build and audit the 93-identity pool using an already configured encoder.

    The complete model, pool and media contract must match before any files are written.
    """
    root = Path(repository_root) if repository_root is not None else PROJECT_ROOT
    index = ClipIndex.build(
        _live_pool(manifest), encoder, batch_size=batch_size, show_progress=show_progress,
    )
    index.metadata.update({
        "source_manifest": (
            _relative_or_absolute(Path(source_manifest), repository_root=root)
            if source_manifest is not None else "generated_from_cfd_images_and_metadata"
        ),
        "pool_filter": "gender=male;race=White;expression=N",
    })
    media_hash, _, _ = media_manifest_integrity(
        index.manifest, repository_root=root, image_root=Path(image_root),
    )
    index.metadata["media_manifest_sha256"] = media_hash
    serialized = np.ascontiguousarray(index.image_embeddings, dtype=np.float32)
    runtime = ClipIndex(serialized, index.manifest, metadata=index.metadata)
    metadata = {
        **index.metadata,
        "embedding_sha256": hashlib.sha256(serialized.tobytes(order="C")).hexdigest(),
    }
    candidate_hash = hashlib.sha256(
        np.ascontiguousarray(runtime.image_embeddings, dtype=np.float32).tobytes(order="C")
    ).hexdigest()
    pool_rows = [
        f"{row.image_id}\t{row.target_id}"
        for row in runtime.manifest[["image_id", "target_id"]].itertuples(index=False)
    ]
    audit = {
        **metadata,
        "candidate_embedding_sha256": candidate_hash,
        "pool_sha256": hashlib.sha256("\n".join(pool_rows).encode("utf-8")).hexdigest(),
        "media_manifest_sha256": media_hash,
    }
    _verify_profile(audit, profile)
    audit.update({
        "selection_profile_id": profile.profile_id,
        "selection_profile_sha256": profile.profile_sha256,
        "profile_match_verified": True,
    })
    index.save(output_dir)
    return audit


def _live_pool(manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = (
        manifest["gender"].astype(str).str.strip().str.casefold().eq("male")
        & manifest["race"].astype(str).str.strip().str.casefold().eq("white")
        & manifest["expression"].astype(str).str.strip().str.casefold().eq("n")
    )
    pool = manifest.loc[eligible].reset_index(drop=True)
    if len(pool) != 93 or pool["target_id"].astype(str).nunique() != 93:
        raise ValueError(
            "The frozen live profile expects exactly 93 White-Male-Neutral identities; "
            f"found {len(pool)} rows and {pool['target_id'].nunique()} identities."
        )
    return pool


def _verify_profile(audit: dict[str, object], profile: LiveSelectionProfile) -> None:
    expected = {
        "model_name": profile.model_name,
        "model_revision": profile.model_revision,
        "model_snapshot_sha256": profile.model_snapshot_sha256,
        "embedding_sha256": profile.index_sha256,
        "candidate_embedding_sha256": profile.candidate_embedding_sha256,
        "pool_sha256": profile.pool_sha256,
        "media_manifest_sha256": profile.media_manifest_sha256,
    }
    mismatches = {
        key: {"actual": audit.get(key), "expected": value}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Built index does not match the frozen selection profile: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _relative_or_absolute(path: Path, *, repository_root: Path | None = None) -> str:
    resolved = path.resolve()
    root = repository_root if repository_root is not None else PROJECT_ROOT
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
