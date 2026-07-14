from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping, Sequence

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
        fields = [
            self.hair,
            self.facial_hair,
            self.eyes,
            self.eyebrows,
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
        return [text for text in fields if text] + list(self.other_details)

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
        self.schema = schema or self.parse_description(verbal_description)
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
    def parse_description(cls, description: str) -> FaceDescriptionSchema:
        """Parse a free-text facial description into the project schema.

        This is deliberately conservative and rule-based. It extracts common
        eyewitness-style feature phrases without inventing attributes.
        """
        text = _normalise_text(description)
        original = description.strip()

        return FaceDescriptionSchema(
            original_description=original,
            gender=_first_match(
                text,
                [
                    (r"\b(dude|guy|man|male|gentleman)\b", "male"),
                    (r"\b(woman|female|lady)\b", "female"),
                    (r"\b(nonbinary|non-binary)\b", "non-binary"),
                ],
            ),
            race=_first_match(
                text,
                [
                    (r"\b(white|caucasian)\b", "White"),
                    (r"\b(asian|east asian|south asian)\b", "Asian"),
                    (r"\b(black|african)\b", "Black"),
                    (r"\b(latino|latina|hispanic)\b", "Latino"),
                    (r"\b(indian)\b", "Indian"),
                    (r"\b(middle eastern|arab)\b", "Middle Eastern"),
                ],
            ),
            age=_parse_age(text),
            hair=_parse_hair(text),
            facial_hair=_parse_facial_hair(text),
            eyes=_parse_eyes(text),
            eyebrows=_first_match(
                text,
                [
                    (r"\b(thick|bushy)\s+eyebrows?\b", "thick eyebrows"),
                    (r"\b(thin|fine)\s+eyebrows?\b", "thin eyebrows"),
                    (r"\b(arched)\s+eyebrows?\b", "arched eyebrows"),
                ],
            ),
            nose=_first_match(
                text,
                [
                    (r"\b(broad|wide)\s+nose\b", "broad nose"),
                    (r"\b(narrow|thin)\s+nose\b", "narrow nose"),
                    (r"\b(long)\s+nose\b", "long nose"),
                    (r"\b(short)\s+nose\b", "short nose"),
                ],
            ),
            build=_first_match(
                text,
                [
                    (r"\b(broad|heavy|stocky)\s+build\b", "broad build"),
                    (r"\b(slim|thin|slender)\s+build\b", "slim build"),
                ],
            ),
            face_shape=_first_match(
                text,
                [
                    (r"\b(round)\s+face\b", "round face"),
                    (r"\b(oval)\s+face\b", "oval face"),
                    (r"\b(narrow|long)\s+face\b", "narrow face"),
                    (r"\b(square)\s+face\b", "square face"),
                ],
            ),
            forehead=_first_match(
                text,
                [
                    (r"\bhigh\s+forehead\b", "high forehead"),
                    (r"\blow\s+forehead\b", "low forehead"),
                ],
            ),
            mouth=_parse_mouth(text),
            ears=_first_match(
                text,
                [
                    (r"\bvisible\s+ears?\b", "visible ears"),
                    (r"\b(covered|hidden)\s+ears?\b", "covered ears"),
                ],
            ),
            jaw=_first_match(
                text,
                [
                    (r"\b(strong|square)\s+jaw(line)?\b", "strong jaw"),
                    (r"\bsoft\s+jaw(line)?\b", "soft jawline"),
                ],
            ),
            teeth=_first_match(
                text,
                [
                    (r"\b(no|without)\s+visible\s+teeth\b", "no visible teeth"),
                    (r"\bvisible\s+teeth\b", "visible teeth"),
                ],
            ),
            expression=_parse_expression(text),
            clothing=_parse_clothing(text),
            accessories=_parse_accessories(text),
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
        for key, value in asdict(self.schema).items():
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
            "schema": asdict(self.schema),
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
            "schema": asdict(self.schema),
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


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


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


def _first_match(text: str, patterns: Sequence[tuple[str, str]]) -> str | None:
    for pattern, value in patterns:
        if re.search(pattern, text):
            return value
    return None


def _parse_age(text: str) -> str | None:
    match = re.search(r"\b(?:around|about)?\s*(\d{2})\s*(?:years?\s*old|yo)?\b", text)
    if match:
        age = int(match.group(1))
        if age < 18:
            raise ValueError("FillerGenerator only supports synthetic adult faces; age must be 18 or older.")
        return f"around {age} years old"
    if re.search(r"\byoung\b", text):
        return "young adult"
    if re.search(r"\bmiddle[- ]aged\b", text):
        return "middle-aged adult"
    if re.search(r"\bold(er)?\b", text):
        return "older adult"
    return None


def _parse_hair(text: str) -> str | None:
    if re.search(r"\bbald\b", text):
        return "bald head"
    parts = []
    length = _first_match(
        text,
        [
            (r"\blong\s+hair\b", "long"),
            (r"\bshort\s+hair\b", "short"),
            (r"\bmedium[- ]length\s+hair\b", "medium-length"),
        ],
    )
    color = _first_match(
        text,
        [
            (r"\bblack\s+hair\b", "black"),
            (r"\bbrown\s+hair\b", "brown"),
            (r"\bblond(e)?\s+hair\b", "blond"),
            (r"\bgray|grey\s+hair\b", "gray"),
            (r"\bred\s+hair\b", "red"),
            (r"\bdark\s+hair\b", "dark"),
            (r"\blight\s+hair\b", "light"),
        ],
    )
    if length:
        parts.append(length)
    if color:
        parts.append(color)
    if parts:
        return " ".join(parts) + " hair"
    if "hair" in text:
        return "visible hair"
    return None


def _parse_facial_hair(text: str) -> str | None:
    if re.search(r"\b(no beard|no facial hair|clean[- ]shaven|clean shaven)\b", text):
        return "no facial hair"
    if re.search(r"\b(full beard|big beard|thick beard)\b", text):
        return "full beard"
    if re.search(r"\bbeard(ed)?\b", text):
        return "beard"
    if re.search(r"\b(moustache|mustache)\b", text):
        return "mustache"
    if re.search(r"\bstubble\b", text):
        return "stubble"
    return None


def _parse_eyes(text: str) -> str | None:
    return _first_match(
        text,
        [
            (r"\bblue\s+eyes?\b", "blue eyes"),
            (r"\bbrown\s+eyes?\b", "brown eyes"),
            (r"\bgreen\s+eyes?\b", "green eyes"),
            (r"\bdark\s+eyes?\b", "dark eyes"),
            (r"\blight\s+eyes?\b", "light eyes"),
        ],
    )


def _parse_mouth(text: str) -> str | None:
    return _first_match(
        text,
        [
            (r"\bthin\s+lips?\b", "thin lips"),
            (r"\bfull\s+lips?\b", "full lips"),
            (r"\bwide\s+mouth\b", "wide mouth"),
        ],
    )


def _parse_expression(text: str) -> str | None:
    return _first_match(
        text,
        [
            (r"\bopen[- ]mouth\s+smile\b", "open-mouth smile"),
            (r"\bclosed[- ]mouth\s+smile\b", "closed-mouth smile"),
            (r"\bsmil(e|ing)\b", "smiling expression"),
            (r"\bneutral\s+expression\b", "neutral expression"),
            (r"\bangry\b", "angry expression"),
            (r"\bsad\b", "sad expression"),
            (r"\bsurprised\b", "surprised expression"),
        ],
    )


def _parse_clothing(text: str) -> str | None:
    return _first_match(
        text,
        [
            (r"\b(gray|grey)\s+(shirt|t-shirt|tee)\b", "gray shirt"),
            (r"\bwhite\s+(shirt|t-shirt|tee)\b", "white shirt"),
            (r"\bblack\s+(shirt|t-shirt|tee)\b", "black shirt"),
            (r"\bsuit\b", "suit"),
            (r"\bhoodie\b", "hoodie"),
        ],
    )


def _parse_accessories(text: str) -> str | None:
    return _first_match(
        text,
        [
            (r"\bglasses\b", "glasses"),
            (r"\bsunglasses\b", "sunglasses"),
            (r"\bhat\b", "hat"),
            (r"\bcap\b", "cap"),
        ],
    )


def _variation_phrase(index: int, total: int, include_age: bool = True) -> str:
    age_values = [24, 28, 32, 36, 40, 44, 48, 52, 56]
    age = age_values[(index - 1) % len(age_values)]
    parts = [f"distinct fictional identity {index} of {total}"]
    if include_age:
        parts.append(f"around {age} years old")
    parts.append("same requested attributes")
    return ", ".join(parts)
