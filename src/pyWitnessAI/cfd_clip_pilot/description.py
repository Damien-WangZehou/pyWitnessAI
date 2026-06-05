from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


DEFAULT_FIELDS = (
    "age",
    "race",
    "gender",
    "skin_tone",
    "hair_colour",
    "hair_length",
    "facial_hair",
    "face_shape",
    "notable_features",
)


@dataclass(frozen=True)
class DescriptionTemplate:
    """Template settings for CFD metadata-derived proxy descriptions."""

    fields: tuple[str, ...] = DEFAULT_FIELDS
    prefix: str = "A neutral frontal face photograph of"
    fallback_subject: str = "an adult person"


def build_proxy_descriptions(
    manifest: pd.DataFrame,
    unique_by: str = "target_id",
    template: DescriptionTemplate | None = None,
    required_fields: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build one CLIP query per target/image from CFD annotations."""
    if unique_by not in manifest.columns:
        raise ValueError(f"unique_by column is missing from manifest: {unique_by}")

    template = template or DescriptionTemplate()
    data = manifest.copy()
    if required_fields:
        data = _drop_missing_required_fields(data, required_fields)

    queries = (
        data.sort_values(["target_id", "image_id"])
        .drop_duplicates(unique_by, keep="first")
        .copy()
    )
    queries["query_id"] = queries[unique_by].astype(str)
    queries["description"] = [
        render_cfd_description(row, template=template)
        for row in queries.to_dict(orient="records")
    ]
    queries["description_source"] = "cfd_metadata_template"

    keep_first = [
        "query_id",
        "target_id",
        "image_id",
        "description",
        "description_source",
    ]
    ordered = keep_first + [col for col in queries.columns if col not in keep_first]
    return queries[ordered].reset_index(drop=True)


def render_cfd_description(
    row: Mapping[str, object],
    template: DescriptionTemplate | None = None,
) -> str:
    """Render a CLIP-friendly sentence from one CFD metadata row."""
    template = template or DescriptionTemplate()
    subject_parts = []
    detail_parts = []

    age = _age_phrase(row.get("age"))
    if age:
        subject_parts.append(age)

    for field in ("race", "gender"):
        value = _clean_value(row.get(field))
        if value:
            subject_parts.append(value.lower())

    for field in template.fields:
        if field in {"age", "race", "gender"}:
            continue
        value = _clean_detail_value(row.get(field))
        if value:
            label = field.replace("_", " ")
            detail_parts.append(f"{label}: {value.lower()}")

    subject = " ".join(subject_parts + ["person"]) if subject_parts else template.fallback_subject
    sentence = f"{template.prefix} {subject}"
    if detail_parts:
        sentence += " with " + ", ".join(detail_parts)
    return sentence + "."


def _clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unspecified", "unknown"}:
        return ""
    return text


def _age_phrase(value: object) -> str:
    text = _clean_value(value)
    if not text:
        return ""

    try:
        age = float(text)
    except ValueError:
        return text.lower()

    if age.is_integer():
        return f"around {int(age)} years old"
    return f"around {age:.1f} years old"


def _clean_detail_value(value: object) -> str:
    text = _clean_value(value)
    if not text:
        return ""
    try:
        float(text)
    except ValueError:
        return text
    return ""


def _drop_missing_required_fields(
    dataframe: pd.DataFrame,
    required_fields: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    missing_columns = [field for field in required_fields if field not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Required fields are missing from manifest: {missing_columns}")

    mask = pd.Series(True, index=dataframe.index)
    for field in required_fields:
        mask &= dataframe[field].map(_clean_value).astype(bool)
    return dataframe[mask].copy()
