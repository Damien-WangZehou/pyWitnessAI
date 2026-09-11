"""Select fillers by eyewitness description or reference-face distance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .FaceSearch import FaceSearch
from .ImageCatalog import ImageCatalog, image_manifest, image_paths

__all__ = ["FillerSelector", "select_by_similarity"]


class FillerSelector(FaceSearch):
    """Select the top five fillers by default from a folder or image catalog.

    Inherited ``from_folder`` supports metadata joins and image validation.
    ``parser``, ``selector`` and ``scoring`` configure independent experiment
    choices; the defaults are raw text, CLIP, and unscaled cosine similarity.
    """

    def __init__(self, pool, *, n: int = 5, **kwargs) -> None:
        self.n = _validate_n(n)
        if isinstance(pool, (str, Path)):
            pool = ImageCatalog.from_folder(pool)
        super().__init__(pool, **kwargs)

    @classmethod
    def from_folder(cls, image_dir, *, n: int = 5, **kwargs):
        n = _validate_n(n)
        selector = super().from_folder(image_dir, **kwargs)
        selector.n = n
        return selector

    def select(self, description, *, n: int | None = None, **kwargs):
        explicit_policy = any(kwargs.get(key) is not None for key in ("policy", "top_k", "min_score"))
        if n is not None and explicit_policy:
            raise ValueError("Pass n or policy/top_k/min_score, not both.")
        if not explicit_policy:
            kwargs["top_k"] = self.n if n is None else _validate_n(n)
        return super().select(description, **kwargs)


def select_by_similarity(
    suspect: Image.Image | str | Path,
    pool,
    *,
    n: int = 5,
    order: str = "most",
    model: str = "VGG-Face",
    backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
) -> pd.DataFrame:
    """Select n faces by ImagesAI distance, preserving candidate metadata.

    ``pool`` accepts the same inputs as ``image_paths``. ``order='most'`` takes
    the smallest distances; ``'least'`` takes the largest. The returned table
    adds ``face_distance`` and can be passed directly to LineupBuilder. All
    candidate distances must be finite; a failed face cannot silently enter a
    least-similar selection. The suspect's own file is excluded automatically.
    Requires the optional ``pyWitnessAI[vision]`` dependencies only when called.
    """
    n = _validate_n(n)
    if order not in {"most", "least"}:
        raise ValueError("order must be 'most' or 'least'.")
    if distance_metric not in {"cosine", "euclidean", "euclidean_l2"}:
        raise ValueError("distance_metric must be 'cosine', 'euclidean', or 'euclidean_l2'.")
    if isinstance(suspect, Image.Image):
        filename = getattr(suspect, "filename", None)
        suspect_path = Path(filename).resolve() if filename else None
    else:
        suspect_path = image_paths([suspect])[0]
    frame = image_manifest(pool)
    if suspect_path is not None:
        frame = frame.loc[frame["image_path"].map(Path) != suspect_path].reset_index(drop=True)
    if len(frame) < n:
        raise ValueError(f"Need {n} distinct fillers; pool contains {len(frame)} after excluding the suspect.")

    # Keep model imports and loading out of ordinary text selection and building.
    from .Images import Images
    from .ImagesAI import ImagesAI

    columns, rows = Images([]), Images([])
    columns.images["suspect"] = _similarity_image(suspect)
    for index, path in enumerate(frame["image_path"]):
        key = str(index)
        rows.images[key] = _similarity_image(path)
        rows.path_to_images[key] = path
    analyzer = ImagesAI(
        column_images=columns, row_images=rows, model=model, backend=backend,
        distance_metric=distance_metric, enforce_detection=enforce_detection,
    )
    analyzer.process()
    # dataframe() rounds distances; retain full precision when deciding ranks.
    distances = analyzer.similarity_matrix.reindex(index=list(rows.images))["suspect"].to_numpy(dtype=float)
    if not np.isfinite(distances).all():
        failed = frame.loc[~np.isfinite(distances), "image_path"].tolist()
        raise ValueError(f"Face comparison failed for {len(failed)} candidate(s): {failed}")
    frame["face_distance"] = distances
    return frame.sort_values("face_distance", ascending=order == "most", kind="stable").head(n).reset_index(drop=True)


def _similarity_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(image) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _validate_n(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer.")
    return n


def __getattr__(name: str):
    # Preserve the historical import without loading its optional vision stack
    # when researchers use the lightweight, configurable selector.
    if name == "CLIPFillerSelector":
        from ._legacy_filler_selector import CLIPFillerSelector
        return CLIPFillerSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
