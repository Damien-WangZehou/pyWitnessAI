from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

import pandas as pd

from .FillerGenerator import FaceDescriptionSchema, FillerGenerator, GeneratedFiller, ImageGenerationBackend

DatasetMatchMode = Literal["exact", "contains"]

DEFAULT_GENERATED_FACE_DATASET_ROOT = Path("./data/generated_filler_benchmark")
SCHEMA_COLUMNS = (
    "gender",
    "race",
    "age",
    "hair",
    "facial_hair",
    "eyes",
    "eyebrows",
    "nose",
    "build",
    "face_shape",
    "forehead",
    "mouth",
    "ears",
    "jaw",
    "teeth",
    "expression",
    "clothing",
    "accessories",
    "other_details",
)

__all__ = [
    "DEFAULT_GENERATED_FACE_DATASET_ROOT",
    "DatasetMatchMode",
    "GeneratedFaceDataset",
    "SCHEMA_COLUMNS",
]


class GeneratedFaceDataset:
    """Manifest-first dataset for generated or manually collected filler faces."""

    def __init__(self, root: str | Path = DEFAULT_GENERATED_FACE_DATASET_ROOT) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.batches_dir = self.root / "batches"
        self.manifest_path = self.root / "manifest.csv"

    def load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=self.manifest_columns())
        manifest = pd.read_csv(self.manifest_path)
        for column in self.manifest_columns():
            if column not in manifest.columns:
                manifest[column] = pd.NA
        return manifest

    def save_manifest(self, manifest: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(self.manifest_path, index=False)

    def manifest_columns(self) -> list[str]:
        return [
            "image_id",
            "image_name",
            "image_path",
            "relative_image_path",
            "batch_id",
            "batch_index",
            "schema_hash",
            "prompt",
            "prompt_hash",
            "provider",
            "model",
            "source",
            "source_image_path",
            "source_image_sha256",
            "source_mode",
            "source_stage",
            "source_label_role",
            "verbal_description",
            "created_at_utc",
            "generated",
            *SCHEMA_COLUMNS,
        ]

    def add_records(self, records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
        if not records:
            return self.load_manifest()

        incoming = pd.DataFrame.from_records(records)
        for column in self.manifest_columns():
            if column not in incoming.columns:
                incoming[column] = pd.NA

        existing = self.load_manifest()
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        combined = combined.drop_duplicates("image_id", keep="last")
        combined = combined.sort_values(["created_at_utc", "image_id"], na_position="last").reset_index(drop=True)
        self.save_manifest(combined)
        return combined

    def select(
        self,
        schema: FaceDescriptionSchema,
        *,
        n: int | None = None,
        exclude_image_ids: set[str] | None = None,
        match: DatasetMatchMode = "exact",
        newest_first: bool = False,
        require_existing: bool = True,
    ) -> pd.DataFrame:
        if match not in {"exact", "contains"}:
            raise ValueError("match must be 'exact' or 'contains'.")

        manifest = self.load_manifest()
        if manifest.empty:
            return manifest

        rows = manifest.copy()
        required = _schema_values(schema)

        for column, value in required.items():
            if column not in rows.columns:
                return rows.iloc[0:0].copy()
            rows = rows.loc[_series_text(rows[column]) == str(value)]

        if match == "exact":
            for column in SCHEMA_COLUMNS:
                if column in required:
                    continue
                if column in rows.columns:
                    rows = rows.loc[_series_text(rows[column]) == ""]

        if exclude_image_ids:
            rows = rows.loc[~rows["image_id"].astype(str).isin(exclude_image_ids)]

        if require_existing and "image_path" in rows.columns:
            rows = rows.loc[rows["image_path"].map(lambda value: Path(str(value)).exists())]

        sort_cols = [column for column in ["created_at_utc", "image_id"] if column in rows.columns]
        if sort_cols:
            rows = rows.sort_values(sort_cols, ascending=[not newest_first] + [True] * (len(sort_cols) - 1))

        if n is not None:
            rows = rows.head(n)
        return rows.reset_index(drop=True)

    def generate_fillers(
        self,
        schema: FaceDescriptionSchema,
        *,
        verbal_description: str,
        n: int,
        model: str | None = None,
        provider: str = "openai",
        backend: ImageGenerationBackend | None = None,
        backend_kwargs: Mapping[str, object] | None = None,
        size: str = "1024x1024",
        quality: str = "medium",
        output_format: str = "png",
        overwrite: bool = False,
        sleep: float = 0.0,
        source_mode: str = "",
        source_stage: str = "",
        source_label_role: str = "",
    ) -> pd.DataFrame:
        if n <= 0:
            return pd.DataFrame(columns=self.manifest_columns())

        self.images_dir.mkdir(parents=True, exist_ok=True)
        generator = FillerGenerator(
            verbal_description,
            n=n,
            output_dir=self.images_dir,
            model=model,
            provider=provider,
            backend=backend,
            backend_kwargs=backend_kwargs,
            size=size,
            quality=quality,
            output_format=output_format,
            overwrite=overwrite,
            sleep=sleep,
            schema=schema,
            naming_strategy="batch",
            write_metadata=False,
        )
        generated = generator.generate()
        rows = self.records_from_generated(
            generated,
            schema=schema,
            verbal_description=verbal_description,
            created_at_utc=generator.created_at_utc,
            source_mode=source_mode,
            source_stage=source_stage,
            source_label_role=source_label_role,
        )
        self.add_records(rows)
        self._write_batch_metadata(generator, rows)
        return pd.DataFrame.from_records(rows)

    def records_from_generated(
        self,
        generated: Sequence[GeneratedFiller],
        *,
        schema: FaceDescriptionSchema,
        verbal_description: str,
        created_at_utc: str,
        source_mode: str = "",
        source_stage: str = "",
        source_label_role: str = "",
    ) -> list[dict[str, object]]:
        schema_row = _schema_row(schema)
        schema_hash = _schema_hash(schema)
        rows = []
        for result in generated:
            image_path = Path(result.path).resolve()
            row = {
                "image_id": result.image_id or image_path.stem,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "relative_image_path": _relative_to_root(image_path, self.root),
                "batch_id": result.batch_id,
                "batch_index": result.index,
                "schema_hash": schema_hash,
                "prompt": result.prompt,
                "prompt_hash": result.prompt_hash or _short_hash(result.prompt),
                "provider": result.provider,
                "model": result.model,
                "source": "generated",
                "source_mode": source_mode,
                "source_stage": source_stage,
                "source_label_role": source_label_role,
                "verbal_description": verbal_description,
                "created_at_utc": created_at_utc,
                "generated": True,
                **schema_row,
            }
            rows.append(row)
        return rows

    def import_images(
        self,
        image_paths: Sequence[str | Path],
        *,
        schema: FaceDescriptionSchema,
        verbal_description: str,
        source: str = "manual",
        source_mode: str = "",
        source_stage: str = "",
        source_label_role: str = "",
        copy_to_images: bool = False,
    ) -> pd.DataFrame:
        created_at_utc = datetime.now(timezone.utc).isoformat()
        schema_row = _schema_row(schema)
        schema_hash = _schema_hash(schema)
        prompt = verbal_description
        source_paths = [Path(path_like).resolve() for path_like in image_paths]
        source_hashes = {path: _file_sha256(path) for path in source_paths if path.exists()}
        existing = self._manifest_with_file_hashes()
        rows = []
        batch_id = _import_batch_id(created_at_utc, schema_hash, source_paths)

        for index, image_path in enumerate(source_paths, start=1):
            file_hash = source_hashes.get(image_path, "")
            if file_hash:
                duplicate = existing.loc[
                    (_series_text(existing["schema_hash"]) == schema_hash)
                    & (_series_text(existing["source_image_sha256"]) == file_hash)
                ]
                if not duplicate.empty:
                    continue

            if copy_to_images:
                self.images_dir.mkdir(parents=True, exist_ok=True)
                image_id = f"gf_{_batch_token(batch_id)}_{index:04d}"
                managed_path = self.images_dir / f"{image_id}{image_path.suffix.lower()}"
                if managed_path.exists() and file_hash and _file_sha256(managed_path) != file_hash:
                    image_id = f"gf_{_batch_token(batch_id)}_{index:04d}_{file_hash[:6]}"
                    managed_path = self.images_dir / f"{image_id}{image_path.suffix.lower()}"
                if not managed_path.exists():
                    shutil.copy2(image_path, managed_path)
                target_path = managed_path.resolve()
                generated = True
            else:
                image_id = f"manual_{_short_hash(str(image_path) + schema_hash, length=12)}"
                target_path = image_path
                generated = False

            rows.append(
                {
                    "image_id": image_id,
                    "image_name": target_path.name,
                    "image_path": str(target_path),
                    "relative_image_path": _relative_to_root(target_path, self.root),
                    "batch_id": batch_id,
                    "batch_index": index,
                    "schema_hash": schema_hash,
                    "prompt": prompt,
                    "prompt_hash": _short_hash(prompt) if prompt else "",
                    "provider": "",
                    "model": "",
                    "source": source,
                    "source_image_path": str(image_path),
                    "source_image_sha256": file_hash,
                    "source_mode": source_mode,
                    "source_stage": source_stage,
                    "source_label_role": source_label_role,
                    "verbal_description": verbal_description,
                    "created_at_utc": created_at_utc,
                    "generated": generated,
                    **schema_row,
                }
            )
        if not rows:
            return pd.DataFrame(columns=self.manifest_columns())

        self.add_records(rows)
        if copy_to_images:
            self._write_import_batch_metadata(
                batch_id=batch_id,
                created_at_utc=created_at_utc,
                rows=rows,
                schema=schema,
                source=source,
                source_mode=source_mode,
                source_stage=source_stage,
                source_label_role=source_label_role,
            )
        return pd.DataFrame.from_records(rows)

    def _write_batch_metadata(self, generator: FillerGenerator, rows: Sequence[Mapping[str, object]]) -> None:
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "batch_id": generator.batch_id,
            "created_at_utc": generator.created_at_utc,
            "provider": generator.provider,
            "backend": generator.backend.__class__.__name__,
            "model": generator.model,
            "n": len(rows),
            "size": generator.size,
            "quality": generator.quality,
            "output_format": generator.output_format,
            "naming_strategy": generator.naming_strategy,
            "schema": asdict(generator.schema),
            "image_ids": [row.get("image_id") for row in rows],
        }
        (self.batches_dir / f"{generator.batch_id}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_import_batch_metadata(
        self,
        *,
        batch_id: str,
        created_at_utc: str,
        rows: Sequence[Mapping[str, object]],
        schema: FaceDescriptionSchema,
        source: str,
        source_mode: str,
        source_stage: str,
        source_label_role: str,
    ) -> None:
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "batch_id": batch_id,
            "created_at_utc": created_at_utc,
            "n": len(rows),
            "source": source,
            "source_mode": source_mode,
            "source_stage": source_stage,
            "source_label_role": source_label_role,
            "provider": "",
            "model": "",
            "schema": asdict(schema),
            "image_ids": [row.get("image_id") for row in rows],
            "source_image_paths": [row.get("source_image_path") for row in rows],
        }
        (self.batches_dir / f"{batch_id}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _manifest_with_file_hashes(self) -> pd.DataFrame:
        manifest = self.load_manifest()
        if manifest.empty:
            return manifest
        if "source_image_sha256" not in manifest.columns:
            manifest["source_image_sha256"] = ""
        missing_hash = _series_text(manifest["source_image_sha256"]) == ""
        for index, row in manifest.loc[missing_hash].iterrows():
            image_path = Path(str(row.get("image_path", "")))
            if image_path.exists():
                manifest.at[index, "source_image_sha256"] = _file_sha256(image_path)
        return manifest


def _schema_values(schema: FaceDescriptionSchema) -> dict[str, str]:
    values = {}
    raw = asdict(schema)
    for column in SCHEMA_COLUMNS:
        value = raw.get(column)
        if column == "other_details":
            value = " | ".join(value or ())
        if value:
            values[column] = str(value)
    return values


def _schema_row(schema: FaceDescriptionSchema) -> dict[str, str]:
    row = {column: "" for column in SCHEMA_COLUMNS}
    row.update(_schema_values(schema))
    return row


def _schema_hash(schema: FaceDescriptionSchema) -> str:
    return _short_hash(_schema_values(schema), length=12)


def _short_hash(value: object, length: int = 8) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_batch_id(created_at_utc: str, schema_hash: str, source_paths: Sequence[Path]) -> str:
    payload = {
        "created_at_utc": created_at_utc,
        "schema_hash": schema_hash,
        "source_paths": [str(path) for path in source_paths],
    }
    return f"batch_{_timestamp_token(created_at_utc)}_{_short_hash(payload)}"


def _batch_token(batch_id: str) -> str:
    return str(batch_id).removeprefix("batch_")


def _timestamp_token(created_at_utc: str) -> str:
    dt = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d_%H%M%S")


def _series_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)
