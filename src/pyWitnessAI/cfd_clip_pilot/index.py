from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .clip_backend import SentenceTransformerClipEncoder, l2_normalise

"""
Image embedding index and text search
"""

@dataclass
class ClipIndex:
    """Exact cosine-search index over CLIP image embeddings."""

    image_embeddings: np.ndarray
    manifest: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.image_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"image_embeddings must be 2D, got shape {embeddings.shape}.")
        if len(embeddings) != len(self.manifest):
            raise ValueError("Embedding count does not match manifest row count.")
        _validate_manifest(self.manifest)
        self.image_embeddings = l2_normalise(embeddings)
        self.manifest = self.manifest.reset_index(drop=True).copy()

    @classmethod
    def build(
        cls,
        manifest: pd.DataFrame,
        encoder: SentenceTransformerClipEncoder,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "ClipIndex":
        _validate_manifest(manifest)

        embeddings = l2_normalise(
            _ensure_2d(
                np.asarray(
                    encoder.encode_images(
                        manifest["image_path"].tolist(),
                        batch_size=batch_size,
                        show_progress=show_progress,
                    ),
                    dtype=np.float32,
                )
            )
        )
        if len(embeddings) != len(manifest):
            raise ValueError("Embedding count does not match manifest row count.")

        metadata = {
            "model_name": encoder.model_name,
            "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
            "image_count": int(len(manifest)),
            "embedding_sha256": _array_sha256(embeddings),
        }
        return cls(embeddings, manifest.reset_index(drop=True).copy(), metadata)

    @classmethod
    def load(cls, index_dir: str | Path) -> "ClipIndex":
        root = Path(index_dir)
        embeddings = np.load(root / "image_embeddings.npy")
        manifest = pd.read_csv(root / "manifest.csv")
        metadata_path = root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if embeddings.ndim != 2:
            raise ValueError(f"Cached embeddings must be 2D, got shape {embeddings.shape}.")
        if len(embeddings) != len(manifest):
            raise ValueError("Cached embedding count does not match cached manifest row count.")
        _validate_manifest(manifest, prefix="Cached manifest")
        expected_hash = metadata.get("embedding_sha256")
        if expected_hash and expected_hash != _array_sha256(embeddings):
            raise ValueError("Cached embedding hash does not match metadata.json.")
        return cls(embeddings, manifest, metadata)

    def save(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        embeddings = np.asarray(self.image_embeddings, dtype=np.float32)
        self.metadata.update(
            {
                "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
                "image_count": int(len(self.manifest)),
                "embedding_sha256": _array_sha256(embeddings),
            }
        )
        np.save(root / "image_embeddings.npy", embeddings)
        self.manifest.to_csv(root / "manifest.csv", index=False)
        (root / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def search_vectors(
        self,
        query_vectors: np.ndarray,
        top_k: int = 50,
        exclude_image_ids: Sequence[set[str]] | None = None,
        exclude_target_ids: Sequence[set[str]] | None = None,
    ) -> pd.DataFrame:
        """Search with pre-encoded text vectors and return long-form results."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        if self.image_embeddings.size == 0:
            raise ValueError("Index is empty.")

        vectors = _ensure_2d(query_vectors).astype(np.float32)
        vectors = l2_normalise(vectors)
        if vectors.shape[1] != self.image_embeddings.shape[1]:
            raise ValueError(
                "Query embedding dimension does not match image embedding dimension: "
                f"{vectors.shape[1]} != {self.image_embeddings.shape[1]}."
            )
        scores = vectors @ self.image_embeddings.T
        n_queries = vectors.shape[0]
        image_exclusions = _normalise_exclusions(exclude_image_ids, n_queries)
        target_exclusions = _normalise_exclusions(exclude_target_ids, n_queries)

        records = []
        image_ids = self.manifest["image_id"].astype(str).tolist()
        target_ids = self.manifest["target_id"].astype(str).tolist()
        image_paths = self.manifest["image_path"].astype(str).tolist()

        for query_index in range(n_queries):
            order = np.argsort(-scores[query_index], kind="mergesort")
            rank = 0
            for image_index in order:
                image_id = image_ids[image_index]
                target_id = target_ids[image_index]
                if image_id in image_exclusions[query_index]:
                    continue
                if target_id in target_exclusions[query_index]:
                    continue
                rank += 1
                result = {
                    "query_index": query_index,
                    "rank": rank,
                    "image_index": int(image_index),
                    "image_id": image_id,
                    "target_id": target_id,
                    "image_path": image_paths[image_index],
                    "clip_score": float(scores[query_index, image_index]),
                }
                result.update(
                    {
                        column: value
                        for column, value in self.manifest.iloc[int(image_index)].to_dict().items()
                        if column not in result
                    }
                )
                records.append(result)
                if rank >= top_k:
                    break

        if records:
            return pd.DataFrame.from_records(records)
        ordered = [
            "query_index",
            "rank",
            "image_index",
            "image_id",
            "target_id",
            "image_path",
            "clip_score",
        ]
        metadata_columns = [column for column in self.manifest.columns if column not in ordered]
        return pd.DataFrame(columns=ordered + metadata_columns)

    def search_texts(
        self,
        texts: Sequence[str],
        encoder: SentenceTransformerClipEncoder,
        top_k: int = 50,
        batch_size: int = 32,
        show_progress: bool = False,
        exclude_image_ids: Sequence[set[str]] | None = None,
        exclude_target_ids: Sequence[set[str]] | None = None,
    ) -> pd.DataFrame:
        vectors = encoder.encode_texts(texts, batch_size=batch_size, show_progress=show_progress)
        return self.search_vectors(
            vectors,
            top_k=top_k,
            exclude_image_ids=exclude_image_ids,
            exclude_target_ids=exclude_target_ids,
        )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _validate_manifest(manifest: pd.DataFrame, *, prefix: str = "Manifest") -> None:
    required = {"image_id", "target_id", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"{prefix} is missing required columns: {missing}")
    for column in sorted(required):
        values = manifest[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{prefix} {column} values must not be empty.")
    if manifest["image_id"].astype(str).duplicated().any():
        raise ValueError(f"{prefix} image_id values must be unique.")


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D embedding array, got shape {array.shape}")
    return array


def _normalise_exclusions(
    exclusions: Sequence[set[str]] | None,
    n_queries: int,
) -> list[set[str]]:
    if exclusions is None:
        return [set() for _ in range(n_queries)]
    if len(exclusions) != n_queries:
        raise ValueError("Exclusion sequence length must match number of queries.")
    return [set(values) for values in exclusions]
