from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image, ImageOps

from .clip_backend import SentenceTransformerClipEncoder
from .index import ClipIndex


def resolve_repo_path(path_like: str | Path, root: str | Path | None = None) -> Path:
    """Resolve paths stored in CSV files, which are often relative to the repo root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (Path(root) if root is not None else Path.cwd()) / path


def load_rgb_image_for_display(path_like: str | Path, root: str | Path | None = None) -> Image.Image:
    """Load an image as RGB, preferring OpenCV for CFD JPEG color handling."""
    path = resolve_repo_path(path_like, root=root)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        import cv2 as cv

        image_bgr = cv.imread(str(path), cv.IMREAD_COLOR)
        if image_bgr is not None:
            image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            return Image.fromarray(image_rgb)
    except ImportError:
        pass

    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def show_clip_lineup(
    results: pd.DataFrame,
    target_image_path: str | Path | None = None,
    root: str | Path | None = None,
    top_k: int | None = None,
    cols: int = 3,
    target_position: Literal["left", "top", "none"] = "left",
    image_path_col: str = "image_path",
    rank_col: str = "rank",
    score_col: str = "clip_score",
    id_col: str = "target_id",
    figsize_per_tile: tuple[float, float] = (3.0, 3.5),
    max_image_px: int = 900,
    suptitle: str | None = None,
):
    """Display a target image with CLIP top-k results in a 3-column lineup grid.

    Parameters
    ----------
    results:
        DataFrame returned by ``ClipIndex.search_texts`` or a compatible table.
    target_image_path:
        Optional target/suspect image to show alongside the retrieved faces.
    root:
        Repository root used to resolve relative CSV paths.
    top_k:
        Number of retrieved faces to show. If omitted, all rows in ``results`` are shown.
    cols:
        Number of columns in the retrieved-face lineup grid. Use ``cols=3`` for a
        classic 3-column notebook display.
    target_position:
        ``"left"`` shows the target beside the lineup; ``"top"`` shows it above
        the lineup; ``"none"`` hides it even when ``target_image_path`` is given.
    """
    import matplotlib.pyplot as plt

    if image_path_col not in results.columns:
        raise ValueError(f"results must contain an image path column: {image_path_col}")
    if cols < 1:
        raise ValueError("cols must be >= 1")

    candidates = results.head(top_k).reset_index(drop=True).copy() if top_k else results.reset_index(drop=True).copy()
    if candidates.empty:
        raise ValueError("No rows to display.")

    has_target = target_image_path is not None and target_position != "none"
    n_candidates = len(candidates)
    candidate_rows = math.ceil(n_candidates / cols)
    tile_w, tile_h = figsize_per_tile

    if has_target and target_position == "left":
        fig_w = (cols + 1) * tile_w
        fig_h = max(candidate_rows, 1) * tile_h
        fig = plt.figure(figsize=(fig_w, fig_h))
        grid = fig.add_gridspec(candidate_rows, cols + 1)
        target_ax = fig.add_subplot(grid[:, 0])
        candidate_axes = [
            fig.add_subplot(grid[r, c + 1])
            for r in range(candidate_rows)
            for c in range(cols)
        ]
    elif has_target and target_position == "top":
        fig_w = cols * tile_w
        fig_h = (candidate_rows + 1) * tile_h
        fig = plt.figure(figsize=(fig_w, fig_h))
        grid = fig.add_gridspec(candidate_rows + 1, cols)
        target_ax = fig.add_subplot(grid[0, :])
        candidate_axes = [
            fig.add_subplot(grid[r + 1, c])
            for r in range(candidate_rows)
            for c in range(cols)
        ]
    else:
        fig_w = cols * tile_w
        fig_h = candidate_rows * tile_h
        fig, axes = plt.subplots(candidate_rows, cols, figsize=(fig_w, fig_h))
        target_ax = None
        candidate_axes = _flatten_axes(axes)

    if target_ax is not None and target_image_path is not None:
        _imshow_fit(
            target_ax,
            load_rgb_image_for_display(target_image_path, root=root),
            max_image_px=max_image_px,
        )
        target_ax.set_title("Target", fontsize=11, fontweight="bold")
        target_ax.axis("off")

    for ax in candidate_axes:
        ax.axis("off")

    for idx, row in candidates.iterrows():
        ax = candidate_axes[idx]
        image = load_rgb_image_for_display(row[image_path_col], root=root)
        _imshow_fit(ax, image, max_image_px=max_image_px)
        rank = row.get(rank_col, idx + 1)
        identity = row.get(id_col, row.get("image_id", ""))
        score = row.get(score_col, None)
        if score is None or pd.isna(score):
            title = f"Rank {rank}\n{identity}"
        else:
            title = f"Rank {rank} | {score:.3f}\n{identity}"
        ax.set_title(title, fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


def search_and_show_lineup(
    index: ClipIndex,
    encoder: SentenceTransformerClipEncoder,
    description: str,
    target_image_path: str | Path | None = None,
    root: str | Path | None = None,
    top_k: int = 6,
    cols: int = 3,
    target_position: Literal["left", "top", "none"] = "left",
    batch_size: int = 32,
    show_progress: bool = False,
    show_description_title: bool = True,
):
    """Search a description and immediately display target + top-k retrieved faces."""
    results = index.search_texts(
        [description],
        encoder=encoder,
        top_k=top_k,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    fig = show_clip_lineup(
        results,
        target_image_path=target_image_path,
        root=root,
        top_k=top_k,
        cols=cols,
        target_position=target_position,
        suptitle=description if show_description_title else None,
    )
    return results, fig


def _flatten_axes(axes):
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    return [axes]


def _imshow_fit(ax, image: Image.Image, max_image_px: int) -> None:
    image = image.copy()
    image.thumbnail((max_image_px, max_image_px), Image.Resampling.LANCZOS)
    ax.imshow(image)
    ax.set_aspect("equal")
