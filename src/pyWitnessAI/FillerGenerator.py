from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

from .FaceAttributeSchema import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    FaceAttributeDefinition,
    FaceAttributeSchema,
)

from .filler_generation_backends import (
    DEFAULT_IMAGE_MODEL,
    ImageGenerationBackend,
    ImageGenerationRequest,
    KNOWN_CLIP_MODEL_PREFIXES,
    OPENAI_IMAGE_MODELS,
    OpenAIImageBackend,
    available_image_generation_models,
    available_image_generation_providers,
    create_image_generation_backend,
    register_image_generation_backend,
)

KNOWN_IMAGE_MODELS = OPENAI_IMAGE_MODELS

__all__ = [
    "FaceDescriptionSchema",
    "FaceAttributeDefinition",
    "FaceAttributeSchema",
    "DEFAULT_FACE_ATTRIBUTE_SCHEMA",
    "FillerGenerator",
    "GeneratedFiller",
    "ImageGenerationBackend",
    "ImageGenerationRequest",
    "OpenAIImageBackend",
    "available_image_generation_models",
    "available_image_generation_providers",
    "register_image_generation_backend",
]


@dataclass(frozen=True)
class FaceDescriptionSchema:
    """Structured facial-description fields used to build generation prompts."""

    original_description: str
    gender: str | None = None
    race: str | None = None
    age: str | None = None
    hair: str | None = None
    facial_hair: str | None = None
    eyes: str | None = None
    eyebrows: str | None = None
    nose: str | None = None
    build: str | None = None
    face_shape: str | None = None
    forehead: str | None = None
    mouth: str | None = None
    ears: str | None = None
    jaw: str | None = None
    teeth: str | None = None
    expression: str | None = None
    clothing: str | None = None
    accessories: str | None = None
    other_details: tuple[str, ...] = field(default_factory=tuple)
    hair_color: str | None = None
    hair_texture: str | None = None
    eyebrow_color: str | None = None
    custom_attributes: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_attributes(
        cls,
        original_description: str,
        attributes: Mapping[str, str],
        *,
        other_details: tuple[str, ...] = (),
    ) -> "FaceDescriptionSchema":
        field_names = {item.name for item in fields(cls)}
        known = {
            key: value
            for key, value in attributes.items()
            if key in field_names and key not in {"original_description", "custom_attributes", "other_details"}
        }
        custom = tuple(
            sorted(
                (str(key), str(value))
                for key, value in attributes.items()
                if value and key not in known
            )
        )
        return cls(
            original_description=original_description,
            custom_attributes=custom,
            other_details=tuple(other_details),
            **known,
        )

    def attribute_values(self) -> dict[str, str]:
        excluded = {"original_description", "custom_attributes", "other_details"}
        values = {
            item.name: str(getattr(self, item.name))
            for item in fields(self)
            if item.name not in excluded and getattr(self, item.name)
        }
        values.update({str(key): str(value) for key, value in self.custom_attributes if value})
        return values

    def with_attributes(self, attributes: Mapping[str, str]) -> "FaceDescriptionSchema":
        values = self.attribute_values()
        values.update({str(key): str(value) for key, value in attributes.items() if value})
        return self.from_attributes(
            self.original_description,
            values,
            other_details=self.other_details,
        )

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "original_description": self.original_description,
            **self.attribute_values(),
        }
        if self.other_details:
            values["other_details"] = list(self.other_details)
        return values

    def subject_phrase(self) -> str:
        subject_terms = []
        if self.age:
            subject_terms.append(self.age)
        if self.race:
            subject_terms.append(self.race)
        if self.gender:
            subject_terms.append(self.gender)
        subject_terms.append("person" if self.age and "adult" in self.age.lower() else "adult person")
        return " ".join(subject_terms)

    def detail_phrases(self) -> list[str]:
        hair = _combine_attribute_labels(
            (self.hair, self.hair_texture, self.hair_color),
            noun="hair",
        )
        eyebrows = _combine_attribute_labels(
            (self.eyebrow_color, self.eyebrows),
            noun="eyebrows",
        )
        detail_fields = [
            hair,
            self.facial_hair,
            self.eyes,
            eyebrows,
            self.nose,
            self.build,
            self.face_shape,
            self.forehead,
            self.mouth,
            self.ears,
            self.jaw,
            self.teeth,
            self.expression,
            self.clothing,
            self.accessories,
        ]
        custom = [value for _, value in self.custom_attributes]
        return [text for text in detail_fields if text] + custom + list(self.other_details)

    def to_generation_prompt(self, variation: str | None = None) -> str:
        details = self.detail_phrases()
        if variation:
            details.append(variation)

        prompt = (
            "Photorealistic synthetic fictional adult face portrait, "
            f"{self.subject_phrase()}"
        )
        if details:
            prompt += " with " + ", ".join(details)
        prompt += (
            ", frontal face, centered crop, shoulders visible, plain light gray studio background, "
            "even soft lighting, neutral camera angle, realistic skin texture, high detail, "
            "not a real person, not a celebrity."
        )
        return prompt


@dataclass(frozen=True)
class GeneratedFiller:
    index: int
    path: str
    prompt: str
    model: str
    provider: str = ""
    image_id: str = ""
    batch_id: str = ""
    prompt_hash: str = ""


class FillerGenerator:
    """Generate synthetic fictional filler faces from a verbal description."""

    def __init__(
        self,
        verbal_description: str,
        n: int = 1,
        output_dir: str | Path = "./fillerGenerated/",
        model: str | None = None,
        *,
        provider: str = "openai",
        backend: ImageGenerationBackend | None = None,
        backend_kwargs: Mapping[str, object] | None = None,
        size: str = "1024x1024",
        quality: str = "medium",
        output_format: str = "png",
        overwrite: bool = False,
        sleep: float = 0.0,
        schema: FaceDescriptionSchema | None = None,
        attribute_schema: FaceAttributeSchema | None = None,
        naming_strategy: Literal["sequential", "batch"] = "sequential",
        batch_id: str | None = None,
        image_id_prefix: str = "gf",
        write_metadata: bool = True,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1.")
        if naming_strategy not in {"sequential", "batch"}:
            raise ValueError("naming_strategy must be 'sequential' or 'batch'.")

        self.verbal_description = verbal_description
        self.n = int(n)
        self.output_dir = Path(output_dir)
        self.backend = backend or create_image_generation_backend(provider, backend_kwargs)
        self.provider = (getattr(self.backend, "provider", None) or provider).strip().lower()
        self.backend_kwargs = dict(backend_kwargs or {})
        self.size = size
        self.quality = quality
        self.output_format = output_format.lower()
        self.overwrite = overwrite
        self.sleep = sleep
        self.attribute_schema = attribute_schema or DEFAULT_FACE_ATTRIBUTE_SCHEMA
        self.schema = schema or self.parse_description(
            verbal_description,
            attribute_schema=self.attribute_schema,
        )
        self.naming_strategy = naming_strategy
        self.image_id_prefix = _slug_token(image_id_prefix or "gf")
        self.write_metadata = write_metadata
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.clip_model: str | None = None
        self.model = self._normalise_model_name(model)
        self.batch_id = batch_id or self._make_batch_id()
        self.results: list[GeneratedFiller] = []

    @staticmethod
    def available_models(provider: str = "openai") -> tuple[str, ...]:
        """Return known image-generation model names for a registered provider."""
        return available_image_generation_models(provider)

    @staticmethod
    def available_providers() -> tuple[str, ...]:
        """Return registered image-generation provider names."""
        return available_image_generation_providers()

    @classmethod
    def parse_description(
        cls,
        description: str,
        *,
        attribute_schema: FaceAttributeSchema | None = None,
    ) -> FaceDescriptionSchema:
        """Parse a free-text facial description into the project schema.

        This is deliberately conservative and rule-based. It extracts common
        eyewitness-style feature phrases without inventing attributes.
        """
        active_schema = attribute_schema or DEFAULT_FACE_ATTRIBUTE_SCHEMA
        return FaceDescriptionSchema.from_attributes(
            original_description=description.strip(),
            attributes=active_schema.parse(description),
        )

    def generation_prompts(self) -> list[str]:
        return [
            self.schema.to_generation_prompt(
                variation=_variation_phrase(index, self.n, include_age=self.schema.age is None)
            )
            for index in range(1, self.n + 1)
        ]

    def generate(self, *, dry_run: bool = False) -> list[GeneratedFiller]:
        """Generate images and save them into output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prompts = self.generation_prompts()

        if dry_run:
            for index, prompt in enumerate(prompts, start=1):
                print(f"[dry-run {index}/{self.n}] {self._image_path(index)}")
                print(prompt)
            return []

        generated: list[GeneratedFiller] = []
        for index, prompt in enumerate(prompts, start=1):
            output_path = self._image_path(index)
            if output_path.exists() and not self.overwrite:
                generated.append(
                    GeneratedFiller(
                        index=index,
                        path=str(output_path),
                        prompt=prompt,
                        model=self.model,
                        provider=self.provider,
                        image_id=self._image_id(index),
                        batch_id=self.batch_id,
                        prompt_hash=_short_hash(prompt),
                    )
                )
                continue

            print(f"[{index}/{self.n}] generating {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.backend.generate(
                ImageGenerationRequest(
                    prompt=prompt,
                    output_path=output_path,
                    model=self.model,
                    size=self.size,
                    quality=self.quality,
                    output_format=self.output_format,
                    index=index,
                    total=self.n,
                )
            )
            generated.append(
                GeneratedFiller(
                    index=index,
                    path=str(output_path),
                    prompt=prompt,
                    model=self.model,
                    provider=self.provider,
                    image_id=self._image_id(index),
                    batch_id=self.batch_id,
                    prompt_hash=_short_hash(prompt),
                )
            )
            if self.sleep:
                time.sleep(self.sleep)

        self.results = generated
        if self.write_metadata:
            self._write_metadata()
        return generated

    def show(self, cols: int = 3, figsize_per_image: tuple[float, float] = (3.0, 3.4)) -> None:
        """Display generated images in a notebook or matplotlib window."""
        image_paths = self.image_paths()
        if not image_paths:
            print(f"No generated images found in {self.output_dir}")
            return

        from PIL import Image
        import matplotlib.pyplot as plt

        rows = (len(image_paths) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * figsize_per_image[0], rows * figsize_per_image[1]),
        )
        axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
        for ax in axes_list:
            ax.axis("off")

        for index, path in enumerate(image_paths):
            with Image.open(path) as image:
                axes_list[index].imshow(image.convert("RGB"))
            axes_list[index].set_title(Path(path).name, fontsize=9)

        fig.tight_layout()
        try:
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except ImportError:
            plt.show()

    def print(self) -> None:
        """Print schema, prompt, model, and known output paths."""
        print(self.summary())

    def summary(self) -> str:
        lines = [
            "FillerGenerator",
            f"  description: {self.verbal_description}",
            f"  n: {self.n}",
            f"  provider: {self.provider}",
            f"  backend: {self.backend.__class__.__name__}",
            f"  image_model: {self.model}",
            f"  naming_strategy: {self.naming_strategy}",
            f"  batch_id: {self.batch_id}",
            f"  output_dir: {self.output_dir}",
        ]
        if self.clip_model:
            lines.append(f"  note: {self.clip_model} is a CLIP retrieval model; using {self.model} for generation.")
        lines.append("  schema:")
        for key, value in self.schema.to_dict().items():
            if value:
                lines.append(f"    {key}: {value}")
        lines.append("  first prompt:")
        lines.append(f"    {self.generation_prompts()[0]}")
        return "\n".join(lines)

    def to_manifest(self) -> list[dict[str, str]]:
        return [
            {
                "index": str(result.index),
                "image_id": result.image_id or Path(result.path).stem,
                "batch_id": result.batch_id or self.batch_id,
                "image_path": result.path,
                "image_name": Path(result.path).name,
                "prompt": result.prompt,
                "prompt_hash": result.prompt_hash or _short_hash(result.prompt),
                "provider": result.provider or self.provider,
                "model": result.model,
                "verbal_description": self.verbal_description,
                "created_at_utc": self.created_at_utc,
            }
            for result in (self.results or self._existing_results())
        ]

    def image_paths(self) -> list[Path]:
        return [path for path in [self._image_path(index) for index in range(1, self.n + 1)] if path.exists()]

    def _normalise_model_name(self, model: str | None) -> str:
        model_text = (model or "").strip()
        default_model = getattr(self.backend, "default_model", None)
        known_models = getattr(self.backend, "known_models", ())

        if not model_text:
            if default_model:
                return default_model
            raise ValueError(f"Provider {self.provider!r} does not define a default model; pass model explicitly.")

        lower = model_text.lower()
        if lower.startswith(KNOWN_CLIP_MODEL_PREFIXES):
            self.clip_model = model_text
            if not default_model:
                raise ValueError(
                    f"{model_text!r} looks like a CLIP retrieval model, but provider {self.provider!r} "
                    "does not define a default image-generation model."
                )
            model_list = ", ".join(known_models) if known_models else default_model
            print(
                f"[FillerGenerator] {model_text!r} looks like a CLIP retrieval model, "
                f"not an image generation model. Using {default_model!r} for image generation. "
                f"Known {self.provider} image models: {model_list}"
            )
            return default_model

        if known_models and model_text not in known_models:
            print(
                f"[FillerGenerator] model={model_text!r}. Known {self.provider} image models include: "
                f"{', '.join(known_models)}. Proceeding anyway."
            )
        return model_text

    def _image_path(self, index: int) -> Path:
        if self.naming_strategy == "batch":
            return self.output_dir / f"{self._image_id(index)}.{self.output_format}"
        return self.output_dir / f"filler_{index:03d}.{self.output_format}"

    def _image_id(self, index: int) -> str:
        if self.naming_strategy == "batch":
            return f"{self.image_id_prefix}_{self._batch_token()}_{index:04d}"
        return f"filler_{index:03d}"

    def _batch_token(self) -> str:
        return self.batch_id.removeprefix("batch_")

    def _make_batch_id(self) -> str:
        timestamp = _timestamp_token(self.created_at_utc)
        payload = {
            "created_at_utc": self.created_at_utc,
            "description": self.verbal_description,
            "schema": self.schema.to_dict(),
            "attribute_schema": self.attribute_schema.to_records(),
            "provider": self.provider,
            "model": self.model,
            "size": self.size,
            "quality": self.quality,
            "output_format": self.output_format,
        }
        return f"batch_{timestamp}_{_short_hash(payload)}"

    def _write_metadata(self) -> None:
        metadata = {
            "verbal_description": self.verbal_description,
            "n": self.n,
            "provider": self.provider,
            "backend": self.backend.__class__.__name__,
            "model": self.model,
            "clip_model": self.clip_model,
            "created_at_utc": self.created_at_utc,
            "batch_id": self.batch_id,
            "naming_strategy": self.naming_strategy,
            "size": self.size,
            "quality": self.quality,
            "output_format": self.output_format,
            "schema": self.schema.to_dict(),
            "attribute_schema": self.attribute_schema.to_records(),
            "results": [asdict(result) for result in self.results],
        }
        (self.output_dir / "generation_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        with (self.output_dir / "generation_manifest.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "index",
                    "image_id",
                    "batch_id",
                    "image_path",
                    "image_name",
                    "prompt",
                    "prompt_hash",
                    "provider",
                    "model",
                    "verbal_description",
                    "created_at_utc",
                ],
            )
            writer.writeheader()
            writer.writerows(self.to_manifest())

    def _existing_results(self) -> list[GeneratedFiller]:
        return [
            GeneratedFiller(
                index=index,
                path=str(self._image_path(index)),
                prompt=prompt,
                model=self.model,
                provider=self.provider,
                image_id=self._image_id(index),
                batch_id=self.batch_id,
                prompt_hash=_short_hash(prompt),
            )
            for index, prompt in enumerate(self.generation_prompts(), start=1)
            if self._image_path(index).exists()
        ]

    def __repr__(self) -> str:
        return (
            f"FillerGenerator(description={self.verbal_description!r}, n={self.n}, "
            f"output_dir={str(self.output_dir)!r}, model={self.model!r})"
        )


def _short_hash(value: object, length: int = 8) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def _timestamp_token(created_at_utc: str) -> str:
    dt = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%d_%H%M%S")


def _slug_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_") or "gf"


def _combine_attribute_labels(values: tuple[str | None, ...], *, noun: str) -> str | None:
    labels = [value for value in values if value]
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    if "bald head" in labels:
        return "bald head"

    stems = []
    for label in labels:
        stem = re.sub(rf"\s+{re.escape(noun)}$", "", label, flags=re.IGNORECASE)
        if stem == "visible" and len(labels) > 1:
            continue
        if stem and stem not in stems:
            stems.append(stem)
    return f"{' '.join(stems)} {noun}" if stems else labels[0]


def _variation_phrase(index: int, total: int, include_age: bool = True) -> str:
    age_values = [24, 28, 32, 36, 40, 44, 48, 52, 56]
    age = age_values[(index - 1) % len(age_values)]
    parts = [f"distinct fictional identity {index} of {total}"]
    if include_age:
        parts.append(f"around {age} years old")
    parts.append("same requested attributes")
    return ", ".join(parts)
