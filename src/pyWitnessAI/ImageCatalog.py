"""Image manifests: discover files, normalise records, and attach metadata.

Both catalog factories share the same image preparation. Metadata is aligned
by a unique key, so joining it never adds or reorders image rows.

Public types come first, followed by image preparation, metadata alignment,
and the small field/report helpers. No model or scoring code belongs here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import pandas as pd


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
MetadataSource = pd.DataFrame | str | Path
MetadataCardinality = Literal["one_to_one", "many_to_one"]

_CATALOG_COLUMNS = {
    "image_id", "target_id", "image_path", "relative_image_path", "image_name",
    "file_extension", "status", "error", "metadata_matched",
}
_JOIN_KEYS = {
    "image_id": ("image_id",),
    "relative_image_path": ("relative_image_path", "relative_path"),
    "image_path": ("image_path", "path"),
    "image_name": ("image_name", "filename", "file", "name"),
}

__all__ = [
    "CatalogReport",
    "IMAGE_EXTENSIONS",
    "ImageCatalog",
    "MetadataJoinSpec",
    "MetadataSource",
    "discover_images",
    "image_manifest",
    "image_paths",
]


@dataclass(frozen=True)
class MetadataJoinSpec:
    """Describe how optional metadata is joined to discovered images."""

    image_key: str
    metadata_key: str | None = None
    validate: MetadataCardinality = "one_to_one"
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if not str(self.image_key).strip():
            raise ValueError("image_key must not be empty.")
        if self.metadata_key is not None and not str(self.metadata_key).strip():
            raise ValueError("metadata_key must not be empty.")
        if self.validate not in {"one_to_one", "many_to_one"}:
            raise ValueError("validate must be 'one_to_one' or 'many_to_one'.")


@dataclass(frozen=True)
class CatalogReport:
    """Image validity and metadata coverage, counted without dropping images."""

    image_count: int
    ready_count: int
    invalid_count: int
    metadata_provided: bool = False
    metadata_row_count: int = 0
    metadata_matched_image_count: int = 0
    metadata_missing_image_count: int = 0
    metadata_orphan_row_count: int = 0
    image_join_key: str | None = None
    metadata_join_key: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ImageCatalog:
    """Image records and their coverage report, independent of image scoring.

    Use ``from_folder`` to discover files or ``from_manifest`` to prepare a
    DataFrame. ``manifest`` retains all rows; ``ready`` selects usable images.
    The constructor accepts an already prepared manifest.
    """

    REQUIRED_COLUMNS = ("image_id", "image_path", "relative_image_path", "image_name", "status", "error")

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        root: str | Path | None = None,
        report: CatalogReport | None = None,
    ) -> None:
        data = manifest.copy().reset_index(drop=True)
        _validate_columns(data)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in data.columns]
        if missing:
            raise ValueError(f"Image catalog is missing required columns: {missing}")
        if data["image_id"].isna().any() or (data["image_id"].astype(str).str.strip() == "").any():
            raise ValueError("image_id values must not be empty.")
        if data["image_id"].astype(str).duplicated().any():
            duplicates = data.loc[data["image_id"].astype(str).duplicated(), "image_id"].tolist()[:3]
            raise ValueError(f"image_id values must be unique; duplicates include {duplicates}")

        self.manifest = data
        self.root = Path(root).resolve() if root is not None else None
        self.report = report or _catalog_report(data)

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        *,
        recursive: bool = True,
        metadata: MetadataSource | None = None,
        metadata_join: MetadataJoinSpec | None = None,
        extensions: Sequence[str] | None = None,
        validate_images: bool = True,
    ) -> "ImageCatalog":
        """Build a stable manifest from a folder and optionally join metadata."""
        folder = Path(root)
        if not folder.exists():
            raise FileNotFoundError(f"Image folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Image folder is not a directory: {folder}")
        folder = folder.resolve()

        paths = discover_images(folder, recursive=recursive, extensions=extensions)
        if not paths:
            raise FileNotFoundError(f"No supported images found under: {folder}")

        manifest = _prepare_manifest(
            pd.DataFrame({"image_path": paths}), folder, validate_images=validate_images,
        )
        manifest = manifest[[
            "image_id", "image_path", "relative_image_path", "image_name",
            "file_extension", "status", "error", "metadata_matched",
        ]]
        if metadata is None:
            return cls(manifest, root=folder)

        metadata_frame, metadata_base = load_metadata(metadata, default_base=folder)
        merged, report = _join_metadata(
            manifest,
            metadata_frame,
            metadata_base=metadata_base,
            join_spec=metadata_join,
        )
        return cls(merged, root=folder, report=report)

    @classmethod
    def from_manifest(
        cls,
        manifest: pd.DataFrame,
        *,
        base_dir: str | Path | None = None,
        validate_images: bool = False,
    ) -> "ImageCatalog":
        """Prepare image_path records relative to base_dir, keeping supplied fields.

        Validation is opt-in here. Without it, supplied status/error fields
        remain authoritative; missing status fields are based on file existence.
        """
        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        data = _prepare_manifest(manifest, base, validate_images=validate_images)
        report = _catalog_report(
            data, metadata_provided=bool(_metadata_columns(data)), metadata_row_count=None,
        )
        return cls(data, root=base, report=report)

    @property
    def ready(self) -> pd.DataFrame:
        return self.manifest.loc[self.manifest["status"].astype(str) == "ready"].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def __repr__(self) -> str:
        return (
            f"ImageCatalog(images={len(self.manifest)}, ready={len(self.ready)}, "
            f"metadata={self.report.metadata_provided})"
        )


def image_manifest(pool) -> pd.DataFrame:
    """Normalize usable image paths while preserving supplied metadata.

    Accept a folder, paths, ImageCatalog, or DataFrame with ``image_path``.
    Tables with a status column contribute only ``ready`` rows. Relative paths
    use the catalog root, or the working directory for other inputs. Duplicate
    paths keep their first occurrence; folders use recursive discovery order.
    This checks file existence, leaving pixel decoding to the consuming tool.
    """
    base = pool.root if isinstance(pool, ImageCatalog) else None
    if isinstance(pool, ImageCatalog):
        pool = pool.ready
    if isinstance(pool, pd.DataFrame):
        _validate_columns(pool)
        if "image_path" not in pool:
            raise ValueError("Image table must contain an image_path column.")
        if "status" in pool:
            pool = pool.loc[pool["status"].astype(str).eq("ready")]
        frame = pool
    elif isinstance(pool, (str, Path)):
        if isinstance(pool, str) and not pool.strip():
            raise ValueError("Image path must not be empty.")
        path = Path(pool)
        values = discover_images(path) if path.is_dir() else [path]
        frame = pd.DataFrame({"image_path": values})
    elif isinstance(pool, Iterable) and not isinstance(pool, (bytes, Mapping)):
        frame = pd.DataFrame({"image_path": list(pool)})
    else:
        raise TypeError("pool must be a folder, image paths, ImageCatalog, or image_path table.")

    paths, positions = [], []
    seen = set()
    for position, value in enumerate(frame["image_path"]):
        if not isinstance(value, (str, Path)):
            raise TypeError("Each image_path must be a string or Path.")
        if isinstance(value, str) and not value.strip():
            raise ValueError("Image path must not be empty.")
        path = ((base or Path.cwd()) / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        if path not in seen:
            seen.add(path)
            positions.append(position)
            paths.append(str(path))
    result = frame.iloc[positions].copy().reset_index(drop=True)
    result["image_path"] = paths
    return result


def image_paths(pool) -> list[Path]:
    """Return resolved, distinct paths from ``image_manifest(pool)``."""
    return [Path(path) for path in image_manifest(pool)["image_path"]]


def discover_images(
    root: str | Path,
    *,
    recursive: bool = True,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Discover supported image files in deterministic relative-path order."""
    folder = Path(root)
    if not folder.exists():
        return []
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder is not a directory: {folder}")

    allowed = {
        suffix.lower() if str(suffix).startswith(".") else f".{str(suffix).lower()}"
        for suffix in (extensions or IMAGE_EXTENSIONS)
    }
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = [path for path in iterator if path.is_file() and path.suffix.lower() in allowed]
    return sorted(paths, key=lambda path: (
        path.relative_to(folder).as_posix().casefold(), path.relative_to(folder).as_posix(),
    ))


def _prepare_manifest(
    manifest: pd.DataFrame, base: Path, *, validate_images: bool,
) -> pd.DataFrame:
    """Fill image fields without replacing supplied IDs, labels, or metadata."""
    _validate_columns(manifest)
    if "image_path" not in manifest.columns:
        raise ValueError("Manifest must contain an image_path column.")
    data = manifest.copy().reset_index(drop=True)
    paths = data["image_path"].map(lambda value: _resolve_path(value, base))
    data["image_path"] = paths.map(str)
    for column, transform in (
        ("relative_image_path", lambda path: _relative_path(path, base)),
        ("image_name", lambda path: path.name),
        ("file_extension", lambda path: path.suffix.lower()),
    ):
        if column not in data:
            data[column] = paths.map(transform)
    if "image_id" not in data:
        data["image_id"] = data["relative_image_path"].map(_image_id)
    else:
        data["image_id"] = data["image_id"].map(str, na_action="ignore")
    _set_image_status(data, paths, validate=validate_images)
    if "metadata_matched" not in data:
        columns = _metadata_columns(data)
        data["metadata_matched"] = data[columns].apply(
            lambda row: any(not _is_empty_metadata_value(value) for value in row), axis=1,
        ) if columns else False
    return data


def _set_image_status(data: pd.DataFrame, paths: pd.Series, *, validate: bool) -> None:
    """Recheck images on request; otherwise preserve supplied status fields."""
    if not validate and "status" in data and "error" in data:
        return
    statuses = paths.map(lambda path: _image_status(path, validate=validate))
    for index, column in enumerate(("status", "error")):
        if validate or column not in data:
            data[column] = statuses.map(lambda item: item[index])


def _validate_columns(frame: pd.DataFrame) -> None:
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique.")


def _metadata_columns(manifest: pd.DataFrame) -> list[str]:
    return [column for column in manifest.columns if column not in _CATALOG_COLUMNS]


def load_metadata(
    source: MetadataSource,
    *,
    default_base: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load metadata from a DataFrame, CSV/TSV, or Excel workbook."""
    if isinstance(source, pd.DataFrame):
        return source.copy(), Path(default_base or Path.cwd()).resolve()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".xls", ".xlsx"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")
    return frame, path.resolve().parent


def _join_metadata(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    metadata_base: Path,
    join_spec: MetadataJoinSpec | None,
) -> tuple[pd.DataFrame, CatalogReport]:
    """Align metadata to image rows after validating the requested key relation."""
    _validate_columns(metadata)
    if metadata.empty:
        output = manifest.copy()
        output["metadata_matched"] = False
        return output, _catalog_report(output, metadata_provided=True)

    spec = join_spec or _infer_metadata_join(manifest, metadata, metadata_base=metadata_base)
    image_key = spec.image_key
    metadata_key = spec.metadata_key or image_key
    image_values, metadata_values = _metadata_join_values(
        manifest, metadata, spec, metadata_base,
    )
    _validate_join_values(image_values, image_key, unique=spec.validate == "one_to_one", catalog=True)
    _validate_join_values(metadata_values, metadata_key, unique=True, catalog=False)

    payload = metadata.drop(columns=metadata_key)
    payload = _rename_metadata_columns(payload, manifest.columns)
    aligned = payload.set_axis(metadata_values).reindex(image_values).set_axis(manifest.index)
    output = pd.concat([manifest.drop(columns="metadata_matched"), aligned], axis=1)
    output["metadata_matched"] = image_values.isin(metadata_values)
    orphan_rows = (~metadata_values.isin(image_values) & metadata_values.ne("")).sum()
    return output, _catalog_report(
        output,
        metadata_provided=True,
        metadata_row_count=len(metadata),
        metadata_orphan_row_count=int(orphan_rows),
        image_join_key=image_key,
        metadata_join_key=metadata_key,
    )


def _metadata_join_values(
    manifest: pd.DataFrame, metadata: pd.DataFrame, spec: MetadataJoinSpec, metadata_base: Path,
) -> tuple[pd.Series, pd.Series]:
    metadata_key = spec.metadata_key or spec.image_key
    if spec.image_key not in manifest:
        raise ValueError(f"Image join key is missing from catalog: {spec.image_key}")
    if metadata_key not in metadata:
        raise ValueError(f"Metadata join key is missing: {metadata_key}")
    semantic = _join_semantic(spec.image_key)
    left = _normalise_join_values(
        manifest[spec.image_key], semantic=semantic,
        base_dir=Path.cwd(), case_sensitive=spec.case_sensitive,
    )
    right = _normalise_join_values(
        metadata[metadata_key], semantic=semantic,
        base_dir=metadata_base, case_sensitive=spec.case_sensitive,
    )
    return left, right


def _validate_join_values(values: pd.Series, key: str, *, unique: bool, catalog: bool) -> None:
    label = "Catalog" if catalog else "Metadata"
    if catalog and values.eq("").any():
        raise ValueError(f"Catalog join key contains empty values: {key}")
    if unique and not values.is_unique:
        duplicates = values[values.duplicated()].tolist()[:3]
        raise ValueError(f"{label} join key must be unique; duplicates include {duplicates}")


def _rename_metadata_columns(metadata: pd.DataFrame, image_columns: pd.Index) -> pd.DataFrame:
    """Keep original labels when possible and prefix conflicts without losing data."""
    occupied = set(image_columns) | set(metadata.columns)
    renamed = {}
    for column in metadata.columns:
        if column in image_columns:
            name = f"metadata__{column}"
            while name in occupied:
                name = f"metadata__{name}"
            renamed[column] = name
            occupied.add(name)
    return metadata.rename(columns=renamed)


def _infer_metadata_join(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    metadata_base: Path,
) -> MetadataJoinSpec:
    scored: list[tuple[int, int, str, str]] = []
    for priority, (image_key, metadata_keys) in enumerate(_JOIN_KEYS.items()):
        if image_key not in manifest.columns:
            continue
        for metadata_key in metadata_keys:
            if metadata_key not in metadata.columns:
                continue
            spec = MetadataJoinSpec(image_key, metadata_key)
            left, right = _metadata_join_values(manifest, metadata, spec, metadata_base)
            if not left.is_unique or not right.is_unique or left.eq("").any():
                continue
            overlap = len((set(left) & set(right)) - {""})
            if overlap:
                scored.append((overlap, -priority, image_key, metadata_key))

    if not scored:
        raise ValueError(
            "Could not infer a metadata join key with matching values. "
            "Pass metadata_join=MetadataJoinSpec(image_key=..., metadata_key=...)."
        )
    _, _, image_key, metadata_key = max(scored)
    return MetadataJoinSpec(image_key=image_key, metadata_key=metadata_key)


def _normalise_join_values(
    values: pd.Series,
    *,
    semantic: str,
    base_dir: Path,
    case_sensitive: bool,
) -> pd.Series:
    def normalise(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if semantic == "image_path":
            text = str(_resolve_path(text, base_dir)).replace("\\", "/")
        elif semantic == "relative_image_path":
            text = text.replace("\\", "/").removeprefix("./")
        elif semantic == "image_name":
            text = Path(text.replace("\\", "/")).name
        return text if case_sensitive else text.casefold()

    return values.map(normalise)


def _join_semantic(column: str) -> str:
    lowered = str(column).strip().lower()
    for semantic, aliases in _JOIN_KEYS.items():
        if lowered in aliases:
            return semantic
    return "identifier"


def _image_status(path: Path, *, validate: bool) -> tuple[str, str]:
    if not path.is_file():
        return "missing", "Image file does not exist."
    if not validate:
        return "ready", ""
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return "invalid_image", f"{type(exc).__name__}: {exc}"
    return "ready", ""


def _is_empty_metadata_value(value: object) -> bool:
    if isinstance(value, str):
        return not value.strip()
    # Lists and other structured metadata are values, not scalar missing markers.
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _image_id(relative_path: object) -> str:
    text = str(relative_path).replace("\\", "/").removeprefix("./")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"image_{digest}"


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _catalog_report(
    manifest: pd.DataFrame,
    *,
    metadata_provided: bool = False,
    metadata_row_count: int | None = 0,
    metadata_orphan_row_count: int = 0,
    image_join_key: str | None = None,
    metadata_join_key: str | None = None,
) -> CatalogReport:
    ready = int(manifest["status"].astype(str).eq("ready").sum())
    matched = int(manifest.get(
        "metadata_matched", pd.Series(False, index=manifest.index),
    ).fillna(False).astype(bool).sum())
    return CatalogReport(
        image_count=len(manifest),
        ready_count=ready,
        invalid_count=len(manifest) - ready,
        metadata_provided=metadata_provided,
        metadata_row_count=matched if metadata_row_count is None else metadata_row_count,
        metadata_matched_image_count=matched,
        metadata_missing_image_count=len(manifest) - matched if metadata_provided else 0,
        metadata_orphan_row_count=metadata_orphan_row_count,
        image_join_key=image_join_key,
        metadata_join_key=metadata_join_key,
    )
