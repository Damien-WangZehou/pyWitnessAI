from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NEUTRAL_EXPRESSION_CODES = {"N", "NEUTRAL"}
KNOWN_EXPRESSION_CODES = {
    "A",
    "AFRAID",
    "ANGRY",
    "D",
    "DISGUSTED",
    "F",
    "FEAR",
    "H",
    "HAPPY",
    "HC",
    "HO",
    "N",
    "NEUTRAL",
    "S",
    "SAD",
    "SURPRISED",
}

COLUMN_ALIASES = {
    "image_id": ("image_id", "image", "file", "filename", "imagefile", "image_file"),
    "target_id": ("target_id", "target", "model", "model_id", "identity", "id"),
    "gender": ("gender", "sex"),
    "race": ("race", "ethnicity", "self_reported_race", "selfreportedrace"),
    "age": ("age", "age_years", "target_age", "estimated_age"),
    "skin_tone": ("skin_tone", "skintone", "skin", "skin_colour", "skin_color"),
    "hair_colour": ("hair_colour", "hair_color", "haircolour", "haircolor"),
    "hair_length": ("hair_length", "hairlength"),
    "facial_hair": ("facial_hair", "facialhair"),
    "face_shape": ("face_shape", "faceshape"),
    "notable_features": ("notable_features", "features", "distinctive_features"),
}


@dataclass(frozen=True)
class ManifestConfig:
    image_dir: Path
    metadata_path: Path | None = None
    neutral_only: bool = True
    max_images: int | None = None


def load_cfd_metadata(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load CFD norming metadata from CSV, TSV, or Excel."""
    metadata_path = Path(path)
    suffix = metadata_path.suffix.lower()

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(metadata_path)
    if suffix in {".tsv"}:
        return pd.read_csv(metadata_path, sep="\t")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(metadata_path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported CFD metadata format: {metadata_path.suffix}")


def build_cfd_manifest(
    image_dir: str | Path,
    metadata_path: str | Path | None = None,
    neutral_only: bool = True,
    max_images: int | None = None,
) -> pd.DataFrame:
    """Create a deterministic image manifest and merge CFD metadata when available."""
    config = ManifestConfig(
        image_dir=Path(image_dir),
        metadata_path=Path(metadata_path) if metadata_path else None,
        neutral_only=neutral_only,
        max_images=max_images,
    )

    records = []
    for path in find_cfd_images(config.image_dir, neutral_only=config.neutral_only):
        stem = path.stem
        records.append(
            {
                "image_id": stem,
                "target_id": target_id_from_image_stem(stem),
                "expression": expression_from_image_stem(stem),
                "image_path": str(path),
            }
        )

    manifest = pd.DataFrame.from_records(records)
    if manifest.empty:
        raise ValueError(f"No CFD images found under {config.image_dir}")

    if config.max_images is not None:
        manifest = manifest.head(config.max_images).copy()

    if config.metadata_path is not None:
        metadata = standardise_cfd_metadata(load_cfd_metadata(config.metadata_path))
        manifest = merge_manifest_metadata(manifest, metadata)

    return manifest.sort_values(["target_id", "image_id"]).reset_index(drop=True)


def find_cfd_images(image_dir: str | Path, neutral_only: bool = True) -> list[Path]:
    """Recursively find CFD images with a stable sort order."""
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"CFD image directory does not exist: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if neutral_only:
        paths = [path for path in paths if is_neutral_cfd_image(path)]
    return sorted(paths, key=lambda p: str(p).lower())


def is_neutral_cfd_image(path: str | Path) -> bool:
    expression = expression_from_image_stem(Path(path).stem)
    return expression in NEUTRAL_EXPRESSION_CODES or expression == ""


def expression_from_image_stem(stem: str) -> str:
    token = stem.replace("_", "-").split("-")[-1].upper()
    return token if token in KNOWN_EXPRESSION_CODES else ""


def target_id_from_image_stem(stem: str) -> str:
    """Remove a CFD expression suffix while keeping the image/identity code intact."""
    parts = stem.replace("_", "-").split("-")
    if parts and parts[-1].upper() in KNOWN_EXPRESSION_CODES:
        return "-".join(parts[:-1])
    return stem


def standardise_cfd_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Add canonical CFD pilot columns while preserving original metadata columns."""
    df = metadata.copy()
    slug_to_column = {_slug_column(col): col for col in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        source = _resolve_column(slug_to_column, aliases)
        if source is not None:
            df[canonical] = df[source]

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].map(_clean_identifier)
    if "target_id" in df.columns:
        df["target_id"] = df["target_id"].map(_clean_identifier)
    elif "image_id" in df.columns:
        df["target_id"] = df["image_id"].map(target_id_from_image_stem)

    if "target_id" in df.columns:
        df["target_id"] = df["target_id"].map(target_id_from_image_stem)

    return df


def merge_manifest_metadata(manifest: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata by image_id first, then target_id as a fallback."""
    metadata_by_key = _metadata_lookup(metadata)
    merged_records = []

    for record in manifest.to_dict(orient="records"):
        meta_record = (
            metadata_by_key.get(str(record.get("image_id", "")))
            or metadata_by_key.get(str(record.get("target_id", "")))
        )
        if meta_record:
            for key, value in meta_record.items():
                if key not in record or pd.isna(record[key]) or record[key] == "":
                    record[key] = value
        merged_records.append(record)

    return pd.DataFrame.from_records(merged_records)


def _metadata_lookup(metadata: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    key_columns = [col for col in ("image_id", "target_id") if col in metadata.columns]

    for record in metadata.to_dict(orient="records"):
        for column in key_columns:
            key = _clean_identifier(record.get(column, ""))
            if key and key not in lookup:
                lookup[key] = record
    return lookup


def _resolve_column(slug_to_column: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        column = slug_to_column.get(_slug_column(alias))
        if column is not None:
            return column
    return None


def _slug_column(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _clean_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    for suffix in IMAGE_EXTENSIONS:
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text
