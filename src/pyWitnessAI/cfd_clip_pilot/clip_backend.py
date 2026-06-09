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
