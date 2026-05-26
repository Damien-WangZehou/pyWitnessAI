from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .clip_backend import SentenceTransformerClipEncoder, l2_normalise


@dataclass
class ClipIndex:
    """Exact cosine-search index over CLIP image embeddings."""

    image_embeddings: np.ndarray
    manifest: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        manifest: pd.DataFrame,
        encoder: SentenceTransformerClipEncoder,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> "ClipIndex":
        if "image_path" not in manifest.columns:
            raise ValueError("Manifest must contain an image_path column.")

        embeddings = encoder.encode_images(
            manifest["image_path"].tolist(),
            batch_size=batch_size,
            show_progress=show_progress,
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
        return cls(l2_normalise(embeddings.astype(np.float32)), manifest, metadata)

    def save(self, index_dir: str | Path) -> None:
        root = Path(index_dir)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "image_embeddings.npy", self.image_embeddings.astype(np.float32))
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
        if self.image_embeddings.size == 0:
            raise ValueError("Index is empty.")

        vectors = _ensure_2d(query_vectors).astype(np.float32)
        vectors = l2_normalise(vectors)
        scores = vectors @ self.image_embeddings.T
        n_queries = vectors.shape[0]
        image_exclusions = _normalise_exclusions(exclude_image_ids, n_queries)
        target_exclusions = _normalise_exclusions(exclude_target_ids, n_queries)

        records = []
        image_ids = self.manifest["image_id"].astype(str).tolist()
        target_ids = self.manifest["target_id"].astype(str).tolist()
        image_paths = self.manifest["image_path"].astype(str).tolist()

        for query_index in range(n_queries):
            order = np.argsort(scores[query_index])[::-1]
            rank = 0
            for image_index in order:
                image_id = image_ids[image_index]
                target_id = target_ids[image_index]
                if image_id in image_exclusions[query_index]:
                    continue
                if target_id in target_exclusions[query_index]:
                    continue
                rank += 1
                records.append(
                    {
                        "query_index": query_index,
                        "rank": rank,
                        "image_index": int(image_index),
                        "image_id": image_id,
                        "target_id": target_id,
                        "image_path": image_paths[image_index],
                        "clip_score": float(scores[query_index, image_index]),
                    }
                )
                if rank >= top_k:
                    break

        return pd.DataFrame.from_records(records)

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
