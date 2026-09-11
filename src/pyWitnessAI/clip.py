"""Shared CLIP encoding and exact-search index utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_CLIP_MODEL = "clip-ViT-L-14"
DEFAULT_CLIP_MODEL_REVISION = "1b4b2e899178706d1b9905460ac21de1e0ba86a5"
MODEL_SNAPSHOT_HASH_VERSION = "model_snapshot_tree_sha256_v1"

__all__ = [
    "ClipIndex",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_MODEL_REVISION",
    "MODEL_SNAPSHOT_HASH_VERSION",
    "SentenceTransformerClipEncoder",
    "directory_content_sha256",
    "l2_normalise",
]


@dataclass
class SentenceTransformerClipEncoder:
    """Thin wrapper around sentence-transformers CLIP models."""

    model_name: str = DEFAULT_CLIP_MODEL
    device: str | None = None
    model_revision: str | None = None
    local_model_path: str | Path | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CLIP encoding. "
                "Install it with: python -m pip install 'pyWitnessAI[clip]'"
            ) from exc

        local_path = Path(self.local_model_path).resolve() if self.local_model_path else None
        source = str(local_path or self.model_name)
        if local_path is not None and not local_path.is_dir():
            raise FileNotFoundError(f"Local CLIP model snapshot does not exist: {source}")
        if local_path is not None and self.model_revision:
            if self.model_revision != local_path.name:
                raise ValueError(
                    "model_revision does not match the local snapshot directory name: "
                    f"{self.model_revision!r} != {local_path.name!r}."
                )
        if local_path is not None and re.fullmatch(r"[0-9a-f]{40}", local_path.name):
            self.model_revision = local_path.name
        elif local_path is None and self.model_name == DEFAULT_CLIP_MODEL and self.model_revision is None:
            self.model_revision = DEFAULT_CLIP_MODEL_REVISION

        kwargs = {"device": self.device}
        if self.local_model_path is None and self.model_revision:
            kwargs["revision"] = self.model_revision
        self.model = SentenceTransformer(source, **kwargs)
        self.loaded_model_source = "local_snapshot" if self.local_model_path else "repository"
        self.loaded_model_snapshot_name = local_path.name if local_path is not None else None
        self.model_snapshot_sha256 = (
            directory_content_sha256(local_path) if local_path is not None else ""
        )

    def encode_images(
        self,
        image_paths: Sequence[str | Path],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        embeddings = []
        iterator = _progress(range(0, len(image_paths), batch_size), show_progress, "Encoding images")

        for start in iterator:
            batch_paths = image_paths[start : start + batch_size]
            images = [_load_rgb_image(path) for path in batch_paths]
            batch = self.model.encode(
                images,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.append(batch)

        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalise(np.vstack(embeddings).astype(np.float32))

    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        embeddings = []
        iterator = _progress(range(0, len(texts), batch_size), show_progress, "Encoding texts")

        for start in iterator:
            batch_texts = list(texts[start : start + batch_size])
            batch = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            embeddings.append(batch)

        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalise(np.vstack(embeddings).astype(np.float32))

    def prepare_text_chunks(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
    ) -> tuple[list[str], list[int], int, int]:
        """Split text on the model's token IDs without silent truncation."""

        context_tokens = int(max_tokens or getattr(self.model, "max_seq_length", 0) or 77)
        try:
            first_module = self.model[0]
            processor = first_module.processor
            tokenizer = processor.tokenizer
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(
                "The loaded SentenceTransformer CLIP model does not expose its tokenizer."
            ) from exc
        configured_context = int(
            getattr(
                getattr(getattr(first_module, "model", None), "config", None),
                "max_position_embeddings",
                0,
            )
            or getattr(
                getattr(
                    getattr(getattr(first_module, "model", None), "config", None),
                    "text_config",
                    None,
                ),
                "max_position_embeddings",
                0,
            )
            or 0
        )
        if configured_context:
            context_tokens = min(context_tokens, configured_context)

        special_tokens = int(tokenizer.num_special_tokens_to_add(pair=False))
        content_budget = context_tokens - special_tokens
        if content_budget < 1:
            raise RuntimeError("The CLIP tokenizer has no usable text-token capacity.")

        token_ids = list(tokenizer.encode(text, add_special_tokens=False, truncation=False))
        if not token_ids:
            raise ValueError("text must not be empty after tokenization.")
        full_token_count = len(token_ids) + special_tokens
        if len(token_ids) <= content_budget:
            return [text], [len(token_ids)], full_token_count, context_tokens

        chunks: list[str] = []
        chunk_token_counts: list[int] = []
        start = 0
        while start < len(token_ids):
            end = min(start + content_budget, len(token_ids))
            chunk = ""
            while end > start:
                candidate_ids = token_ids[start:end]
                candidate = tokenizer.decode(
                    candidate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if candidate:
                    encoded_length = len(
                        tokenizer.encode(
                            candidate,
                            add_special_tokens=True,
                            truncation=False,
                        )
                    )
                    if encoded_length <= context_tokens:
                        chunk = candidate
                        break
                end -= 1
            if not chunk:
                raise RuntimeError(
                    "The CLIP tokenizer could not form a non-empty context-safe text chunk."
                )
            chunks.append(chunk)
            chunk_token_counts.append(end - start)
            start = end
        return chunks, chunk_token_counts, full_token_count, context_tokens


def l2_normalise(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def directory_content_sha256(root: str | Path) -> str:
    """Hash a model snapshot's relative file names and exact file contents."""

    source = Path(root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Model snapshot directory does not exist: {source}")
    files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not files:
        raise ValueError(f"Model snapshot directory contains no files: {source}")

    digest = hashlib.sha256()
    digest.update(f"{MODEL_SNAPSHOT_HASH_VERSION}\0".encode("utf-8"))
    for path in files:
        relative = path.relative_to(source).as_posix()
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                file_digest.update(chunk)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


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
            "model_revision": getattr(encoder, "model_revision", None),
            "model_load_source": getattr(encoder, "loaded_model_source", None),
            "model_snapshot_sha256": getattr(encoder, "model_snapshot_sha256", None),
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


def _load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _progress(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc)
