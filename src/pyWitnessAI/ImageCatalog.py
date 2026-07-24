from __future__ import annotations

import hashlib
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

__all__ = [
    "CatalogReport",
    "IMAGE_EXTENSIONS",
    "ImageCatalog",
    "MetadataJoinSpec",
    "MetadataSource",
    "discover_images",
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
    """Validated, manifest-backed collection of images and optional metadata."""

    REQUIRED_COLUMNS = ("image_id", "image_path", "relative_image_path", "image_name", "status", "error")

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        root: str | Path | None = None,
        report: CatalogReport | None = None,
    ) -> None:
        data = manifest.copy().reset_index(drop=True)
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

        records = []
        for path in paths:
            resolved = path.resolve()
            relative = resolved.relative_to(folder).as_posix()
            status, error = _image_status(resolved, validate=validate_images)
            records.append(
                {
                    "image_id": _image_id(relative),
                    "image_path": str(resolved),
                    "relative_image_path": relative,
                    "image_name": resolved.name,
                    "file_extension": resolved.suffix.lower(),
                    "status": status,
                    "error": error,
                }
            )

        manifest = pd.DataFrame.from_records(records)
        if metadata is None:
            manifest["metadata_matched"] = False
            return cls(manifest, root=folder, report=_catalog_report(manifest))

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
        """Normalise an existing manifest into the common catalog contract."""
        if "image_path" not in manifest.columns:
            raise ValueError("Manifest must contain an image_path column.")

        base = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()
        data = manifest.copy().reset_index(drop=True)
        data["image_path"] = data["image_path"].map(lambda value: str(_resolve_path(value, base)))
        if "relative_image_path" not in data.columns:
            data["relative_image_path"] = data["image_path"].map(
                lambda value: _relative_path(Path(value), base)
            )
        if "image_name" not in data.columns:
            data["image_name"] = data["image_path"].map(lambda value: Path(value).name)
        if "image_id" not in data.columns:
            data["image_id"] = data["relative_image_path"].map(_image_id)
        else:
            if data["image_id"].isna().any():
                raise ValueError("image_id values must not be empty.")
            data["image_id"] = data["image_id"].astype(str)

        if validate_images:
            statuses = data["image_path"].map(lambda value: _image_status(Path(value), validate=True))
            data["status"] = statuses.map(lambda item: item[0])
            data["error"] = statuses.map(lambda item: item[1])
        else:
            path_exists = data["image_path"].map(lambda value: Path(value).is_file())
            data["status"] = data.get("status", path_exists.map({True: "ready", False: "missing"}))
            data["error"] = data.get(
                "error",
                path_exists.map({True: "", False: "Image file does not exist."}),
            )

        catalog_columns = {
            "image_id",
            "target_id",
            "image_path",
            "relative_image_path",
            "image_name",
            "file_extension",
            "status",
            "error",
            "metadata_matched",
        }
        metadata_columns = [column for column in data.columns if column not in catalog_columns]
        metadata_provided = bool(metadata_columns)
        if "metadata_matched" not in data.columns:
            if metadata_columns:
                data["metadata_matched"] = data[metadata_columns].apply(
                    lambda row: any(not _is_empty_metadata_value(value) for value in row),
                    axis=1,
                )
            else:
                data["metadata_matched"] = False
        matched = int(data["metadata_matched"].fillna(False).astype(bool).sum())
        report = _catalog_report(
            data,
            metadata_provided=metadata_provided,
            metadata_row_count=matched,
            metadata_matched_image_count=matched,
            metadata_missing_image_count=len(data) - matched if metadata_provided else 0,
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
    return sorted(paths, key=lambda path: path.relative_to(folder).as_posix().casefold())


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
    if metadata.empty:
        output = manifest.copy()
        output["metadata_matched"] = False
        return output, _catalog_report(
            output,
            metadata_provided=True,
            metadata_row_count=0,
            metadata_missing_image_count=len(output),
        )

    spec = join_spec or _infer_metadata_join(manifest, metadata, metadata_base=metadata_base)
    image_key = spec.image_key
    metadata_key = spec.metadata_key or image_key
    if image_key not in manifest.columns:
        raise ValueError(f"Image join key is missing from catalog: {image_key}")
    if metadata_key not in metadata.columns:
        raise ValueError(f"Metadata join key is missing: {metadata_key}")

    semantic = _join_semantic(image_key)
    left = manifest.copy()
    right = metadata.copy()
    left["__join_key"] = _normalise_join_values(
        left[image_key],
        semantic=semantic,
        base_dir=Path.cwd(),
        case_sensitive=spec.case_sensitive,
    )
    right["__join_key"] = _normalise_join_values(
        right[metadata_key],
        semantic=semantic,
        base_dir=metadata_base,
        case_sensitive=spec.case_sensitive,
    )

    if (left["__join_key"] == "").any():
        raise ValueError(f"Catalog join key contains empty values: {image_key}")
    if spec.validate == "one_to_one" and left["__join_key"].duplicated().any():
        duplicates = left.loc[left["__join_key"].duplicated(), image_key].tolist()[:3]
        raise ValueError(f"Catalog join key must be unique; duplicates include {duplicates}")
    if spec.validate in {"one_to_one", "many_to_one"} and right["__join_key"].duplicated().any():
        duplicates = right.loc[right["__join_key"].duplicated(), metadata_key].tolist()[:3]
        raise ValueError(f"Metadata join key must be unique; duplicates include {duplicates}")

    image_keys = set(left["__join_key"])
    orphan_rows = int((~right["__join_key"].isin(image_keys) & right["__join_key"].ne("")).sum())

    right = right.drop(columns=[metadata_key], errors="ignore")
    rename = {
        column: f"metadata__{column}"
        for column in right.columns
        if column != "__join_key" and column in left.columns
    }
    right = right.rename(columns=rename)

    try:
        output = left.merge(
            right,
            on="__join_key",
            how="left",
            validate=spec.validate,
            indicator="__metadata_merge",
            sort=False,
        )
    except pd.errors.MergeError as exc:
        raise ValueError(f"Metadata join cardinality failed ({spec.validate}): {exc}") from exc

    output["metadata_matched"] = output["__metadata_merge"].eq("both")
    matched = int(output["metadata_matched"].sum())
    output = output.drop(columns=["__join_key", "__metadata_merge"])
    report = _catalog_report(
        output,
        metadata_provided=True,
        metadata_row_count=len(metadata),
        metadata_matched_image_count=matched,
        metadata_missing_image_count=len(output) - matched,
        metadata_orphan_row_count=orphan_rows,
        image_join_key=image_key,
        metadata_join_key=metadata_key,
    )
    return output, report


def _infer_metadata_join(
    manifest: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    metadata_base: Path,
) -> MetadataJoinSpec:
    candidates = (
        ("image_id", ("image_id",)),
        ("relative_image_path", ("relative_image_path", "relative_path")),
        ("image_path", ("image_path", "path")),
        ("image_name", ("image_name", "filename", "file", "name")),
    )
    scored: list[tuple[int, int, str, str]] = []
    for priority, (image_key, metadata_keys) in enumerate(candidates):
        if image_key not in manifest.columns:
            continue
        semantic = _join_semantic(image_key)
        left_values = set(
            _normalise_join_values(
                manifest[image_key],
                semantic=semantic,
                base_dir=Path.cwd(),
                case_sensitive=False,
            )
        ) - {""}
        for metadata_key in metadata_keys:
            if metadata_key not in metadata.columns:
                continue
            right_values = set(
                _normalise_join_values(
                    metadata[metadata_key],
                    semantic=semantic,
                    base_dir=metadata_base,
                    case_sensitive=False,
                )
            ) - {""}
            overlap = len(left_values & right_values)
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
    if lowered in {"image_path", "path"}:
        return "image_path"
    if lowered in {"relative_image_path", "relative_path"}:
        return "relative_image_path"
    if lowered in {"image_name", "filename", "file", "name"}:
        return "image_name"
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
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    try:
        if bool(missing):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(value, str) and not value.strip()


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
    metadata_row_count: int = 0,
    metadata_matched_image_count: int | None = None,
    metadata_missing_image_count: int | None = None,
    metadata_orphan_row_count: int = 0,
    image_join_key: str | None = None,
    metadata_join_key: str | None = None,
) -> CatalogReport:
    ready = int(manifest["status"].astype(str).eq("ready").sum())
    matched = (
        int(manifest.get("metadata_matched", pd.Series(False, index=manifest.index)).fillna(False).sum())
        if metadata_matched_image_count is None
        else metadata_matched_image_count
    )
    missing = (
        len(manifest) - matched
        if metadata_provided and metadata_missing_image_count is None
        else int(metadata_missing_image_count or 0)
    )
    return CatalogReport(
        image_count=len(manifest),
        ready_count=ready,
        invalid_count=len(manifest) - ready,
        metadata_provided=metadata_provided,
        metadata_row_count=metadata_row_count,
        metadata_matched_image_count=matched,
        metadata_missing_image_count=missing,
        metadata_orphan_row_count=metadata_orphan_row_count,
        image_join_key=image_join_key,
        metadata_join_key=metadata_join_key,
    )
