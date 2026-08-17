from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

"""
CLIP Encoder
"""

@dataclass
class SentenceTransformerClipEncoder:
    """Thin wrapper around sentence-transformers CLIP models."""

    model_name: str = "clip-ViT-B-32"
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the CFD CLIP pilot. "
                "Install the cfd-pilot extra or install sentence-transformers."
            ) from exc

        self.model = SentenceTransformer(self.model_name, device=self.device)

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
        """Split text on the model's own token IDs without silent truncation.

        CLIP text encoders have a short context window (77 tokens for the
        model used by this project).  Returning explicit chunks lets callers
        define and record how a longer response is aggregated rather than
        allowing a tokenizer to discard its tail.
        """

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
        for start in range(0, len(token_ids), content_budget):
            chunk_ids = token_ids[start : start + content_budget]
            chunk = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not chunk:
                raise RuntimeError("The CLIP tokenizer produced an empty text chunk.")
            encoded_length = len(
                tokenizer.encode(
                    chunk,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            if encoded_length > context_tokens:
                raise RuntimeError(
                    "A decoded CLIP text chunk exceeds the model context window."
                )
            chunks.append(chunk)
            chunk_token_counts.append(len(chunk_ids))
        return chunks, chunk_token_counts, full_token_count, context_tokens


def l2_normalise(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


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
