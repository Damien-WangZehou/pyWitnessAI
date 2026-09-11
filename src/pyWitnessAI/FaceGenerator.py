"""Plan face portraits, dispatch one backend, and optionally record a manifest.

Random, text, reference-image, and combined generation share one execution
path. Recorded attributes are requested conditions, not verified image labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
from uuid import uuid4

import pandas as pd

from .Description import (
    DescriptionParser, ParsedDescription,
    SchemaDescriptionParser, create_description_parser,
)
from .FaceAttributeSchema import DEFAULT_FACE_ATTRIBUTE_SCHEMA, FaceAttributeSchema, FaceDescriptionSchema
from .GeneratedFaceDataset import GeneratedFaceDataset
from .face_generation_backend import ImageGenerationBackend, ImageGenerationRequest, create_image_generation_backend

__all__ = ["FaceGenerator", "GeneratedFace"]


@dataclass(frozen=True)
class GeneratedFace:
    """A completed face image and the conditions used to request it."""

    index: int
    path: str
    prompt: str
    model: str
    provider: str = ""
    image_id: str = ""
    batch_id: str = ""
    prompt_hash: str = ""
    description: str = ""
    attributes: dict[str, str] = field(default_factory=dict)


class FaceGenerator:
    """Generate n portraits with a shared style and interchangeable backend.

    Construction and generation_prompts() never contact a provider. Call
    generate() (or the configured instance) to create images. A new instance
    owns a unique batch; repeating generate() reuses that batch's existing files.
    seed controls sampled descriptions, not the provider's image randomness.
    """

    def __init__(
        self,
        generator_backend: str | ImageGenerationBackend = "openai",
        output_dir: str | Path = "./facesGenerated",
        n: int = 1,
        *,
        description: str | ParsedDescription | None = None,
        parser: str | DescriptionParser = "schema",
        reference_image: str | Path | None = None,
        similarity: float = 75.0,
        write_manifest: bool = True,
        attribute_schema: FaceAttributeSchema | None = None,
        seed: int | None = None,
        model: str | None = None,
        size: str = "1024x1024",
        quality: str = "medium",
        output_format: str = "png",
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        if description is not None and not isinstance(description, ParsedDescription):
            if not isinstance(description, str) or not description.strip():
                raise ValueError("description must be a non-empty string, ParsedDescription, or None.")
        self.backend = create_image_generation_backend(generator_backend) if isinstance(generator_backend, str) else generator_backend
        if not isinstance(self.backend, ImageGenerationBackend):
            raise TypeError("generator_backend must be a registered name or ImageGenerationBackend instance.")
        self.output_dir, self.n = Path(output_dir), n
        self.description = description.original_text if isinstance(description, ParsedDescription) else description
        if isinstance(parser, SchemaDescriptionParser) and attribute_schema is not None and parser.schema is not attribute_schema:
            raise ValueError("Use the same attribute_schema as the supplied SchemaDescriptionParser.")
        parser_schema = parser.schema if isinstance(parser, SchemaDescriptionParser) else DEFAULT_FACE_ATTRIBUTE_SCHEMA
        self.attribute_schema = attribute_schema if attribute_schema is not None else parser_schema
        self.parser = create_description_parser(parser, schema=self.attribute_schema) if isinstance(parser, str) and parser == "schema" else create_description_parser(parser)
        if isinstance(description, ParsedDescription):
            self.parsed_description = description
            self.parser_name = "preparsed"
        else:
            self.parsed_description = self.parser.parse(description) if description is not None else ParsedDescription("", "", "")
            self.parser_name = getattr(self.parser, "name", type(self.parser).__name__)
        self.reference_image = Path(reference_image).resolve() if reference_image is not None else None
        self.similarity = _similarity(similarity)
        self.write_manifest = bool(write_manifest)
        self.seed = seed if seed is not None else random.SystemRandom().getrandbits(64)
        self.provider = self.backend.provider
        self.model = model or self.backend.default_model
        if not self.model:
            raise ValueError("The backend has no default model; pass model explicitly.")
        self.size, self.quality, self.output_format = size, quality, output_format.lower()
        if self.output_format not in {"png", "jpeg", "webp"}:
            raise ValueError("output_format must be png, jpeg, or webp.")
        self.created_at_utc = datetime.now(timezone.utc).isoformat()
        self.batch_id = f"batch_{uuid4().hex}"
        self.results: list[GeneratedFace] = []

    @property
    def generation_mode(self) -> str:
        if self.reference_image is not None:
            return "description_reference" if self.description is not None else "reference"
        if self.description is not None:
            return "description"
        return "random_description" if self.write_manifest else "random"

    def generation_prompts(self) -> list[str]:
        """Preview the exact prompts for this batch without creating files."""
        return [prompt for prompt, _, _ in self._plan_faces()]

    def _plan_faces(self) -> list[tuple[str, str, dict[str, str]]]:
        rng = random.Random(self.seed)
        plans = []
        for index in range(1, self.n + 1):
            parsed = self.parsed_description
            if self.generation_mode == "random_description":
                parsed = _sample_description(self.attribute_schema, rng)
            if (self.description is not None or self.write_manifest and self.reference_image is None) and not parsed.model_text.strip():
                raise ValueError("Description parser produced empty model input; choose another parser or provide supported attributes.")
            attributes = parsed.schema_attributes
            prompt = self._portrait_prompt(parsed, attributes, index)
            plans.append((prompt, parsed.model_text, attributes))
        return plans

    def _portrait_prompt(self, parsed: ParsedDescription, attributes: dict[str, str], index: int) -> str:
        schema = FaceDescriptionSchema.from_attributes(parsed.original_text, attributes)
        prompt = schema.to_generation_prompt(variation=f"distinct fictional identity {index} of {self.n}")
        if parsed.model_text and parsed.model_text != "; ".join(attributes.values()):
            prompt += f" Requested description: {parsed.model_text}."
        if self.generation_mode == "random":
            prompt += " Randomly choose this person's facial features; create a new, distinct identity."
        if self.reference_image is not None:
            prompt += (
                f" Use the supplied face image as a reference. Requested resemblance: {self.similarity:g}/100,"
                " where 0 means a distinctly dissimilar face and 100 means the greatest resemblance."
                " This is generation guidance, not a measured similarity score."
                " Keep the portrait style above. Explicit described attributes take priority if they conflict with the reference."
            )
        return prompt

    def generate(self, *, dry_run: bool = False) -> list[GeneratedFace]:
        """Create this batch and record each successful image, if requested."""
        plans = self._plan_faces()
        requests = [self._request(index, prompt) for index, (prompt, _, _) in enumerate(plans, 1)]
        for request in requests:
            self.backend.validate_request(request)
        if dry_run:
            for request in requests:
                print(f"[dry-run {request.index}/{self.n}] {request.output_path}\n{request.prompt}")
            return []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        for request, (_, description, attributes) in zip(requests, plans):
            self._generate_image(request)
            result = GeneratedFace(
                request.index, str(request.output_path), request.prompt, self.model, self.provider,
                self._image_id(request.index), self.batch_id, _prompt_hash(request.prompt), description, attributes,
            )
            self.results.append(result)
            if self.write_manifest:
                GeneratedFaceDataset(self.output_dir).add_records([self._manifest_record(result)])
        return self.results

    def __call__(self, *, dry_run: bool = False) -> list[GeneratedFace]:
        return self.generate(dry_run=dry_run)

    def _request(self, index: int, prompt: str) -> ImageGenerationRequest:
        return ImageGenerationRequest(
            prompt=prompt, output_path=self._image_path(index), model=self.model,
            size=self.size, quality=self.quality, output_format=self.output_format,
            index=index, total=self.n, reference_image=self.reference_image, similarity=self.similarity,
        )

    def _generate_image(self, request: ImageGenerationRequest) -> None:
        existing = request.output_path.exists()
        if not existing:
            try:
                self.backend.generate(request)
            except BaseException:
                # A failed new output is not reusable on a later retry.
                if not existing and request.output_path.is_file():
                    request.output_path.unlink(missing_ok=True)
                raise
        if not request.output_path.is_file() or request.output_path.stat().st_size == 0:
            if not existing and request.output_path.is_file():
                request.output_path.unlink(missing_ok=True)
            raise RuntimeError("The image backend did not write a non-empty output image.")

    def _image_id(self, index: int) -> str:
        return f"face_{self.batch_id.removeprefix('batch_')}_{index:04d}"

    def _image_path(self, index: int) -> Path:
        return self.output_dir / f"{self._image_id(index)}.{self.output_format}"

    def image_paths(self) -> list[Path]:
        return [Path(result.path) for result in self.results]

    def _manifest_record(self, result: GeneratedFace) -> dict[str, object]:
        dataset = GeneratedFaceDataset(self.output_dir)
        schema = FaceDescriptionSchema.from_attributes(result.description, result.attributes)
        record = dataset.records_from_generated(
            [result], schema=schema, verbal_description=result.description,
            created_at_utc=self.created_at_utc, source_mode=self.generation_mode,
        )[0]
        record.update(
            generation_mode=self.generation_mode,
            original_description=self.description or "",
            parser=self.parser_name,
            attribute_label_source="requested" if result.attributes else "unspecified",
            attributes_verified=False,
            attribute_schema_name=self.attribute_schema.name,
            attribute_schema_version=self.attribute_schema.version,
            reference_image_path=str(self.reference_image or ""),
            requested_similarity=self.similarity if self.reference_image is not None else "",
            description_seed=str(self.seed) if self.generation_mode == "random_description" else "",
        )
        return record

    def to_manifest(self) -> pd.DataFrame:
        """Return this batch's records, whether or not disk recording is enabled."""
        return pd.DataFrame([self._manifest_record(result) for result in self.results])

    def __repr__(self) -> str:
        return f"FaceGenerator(mode={self.generation_mode!r}, n={self.n}, backend={self.provider!r}, output_dir={str(self.output_dir)!r})"


def _sample_description(schema: FaceAttributeSchema, rng: random.Random) -> ParsedDescription:
    values = schema.sample(rng=rng)
    if not values:
        raise ValueError("Random manifest generation requires schema attributes with sample_values.")
    return ParsedDescription.from_attributes(values)


def _similarity(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("similarity must be a finite number between 0 and 100.")
    try:
        value = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("similarity must be a finite number between 0 and 100.") from exc
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("similarity must be a finite number between 0 and 100.")
    return value


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
