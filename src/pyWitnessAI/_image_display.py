"""Render ranked image tables without importing a model or study implementation."""

from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def show_ranked_images(
    results: pd.DataFrame,
    *,
    top_k: int | None = None,
    cols: int = 3,
    image_path_col: str = "image_path",
    rank_col: str = "rank",
    score_col: str = "selector_score",
    id_col: str = "image_id",
    figsize_per_tile: tuple[float, float] = (3.0, 3.5),
    max_image_px: int = 900,
    suptitle: str | None = None,
):
    """Return a figure showing images in table order with ranks and optional scores."""
    if image_path_col not in results:
        raise ValueError(f"results must contain an image path column: {image_path_col}")
    if cols < 1:
        raise ValueError("cols must be >= 1.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1 or None.")
    if max_image_px < 1:
        raise ValueError("max_image_px must be >= 1.")
    candidates = results if top_k is None else results.head(top_k)
    if candidates.empty:
        raise ValueError("No rows to display.")

    import matplotlib.pyplot as plt

    rows = ceil(len(candidates) / cols)
    tile_w, tile_h = figsize_per_tile
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * tile_w, rows * tile_h), squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")

    try:
        for position, (ax, (_, row)) in enumerate(zip(axes.flat, candidates.iterrows()), start=1):
            ax.imshow(_load_display_image(row[image_path_col], max_image_px))
            rank = row.get(rank_col, position)
            identity = row.get(id_col, row.get("image_id", ""))
            score = row.get(score_col)
            title = f"Rank {rank}" if score is None or pd.isna(score) else f"Rank {rank} | {score:.3f}"
            ax.set_title(f"{title}\n{identity}", fontsize=9)
        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
        fig.tight_layout()
    except Exception:
        plt.close(fig)
        raise
    return fig


def _load_display_image(path: str | Path, max_image_px: int) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = None
    try:
        import cv2

        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is not None:
            image = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
    except ImportError:
        pass
    if image is None:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_image_px, max_image_px), Image.Resampling.LANCZOS)
    return image
