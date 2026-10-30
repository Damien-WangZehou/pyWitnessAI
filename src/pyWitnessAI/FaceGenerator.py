"""Plan face portraits, dispatch one backend, and optionally record a manifest.

Random, text, reference-image, and combined generation share one execution
path. Recorded attributes are requested conditions, not verified image labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from uuid import uuid4

import pandas as pd

from .Description import (
    DescriptionParser, ParsedDescription,
    SchemaDescriptionParser, create_description_parser,
)
from .FaceAttributeSchema import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA, FaceAttributeSchema, FaceDescriptionSchema,
    _PORTRAIT_INSTRUCTIONS,
)
from .GeneratedFaceDataset import GeneratedFaceDataset
from .face_generation_backend import ImageGenerationBackend, ImageGenerationRequest, create_image_generation_backend
from .prompt_processing import (
    PromptProcessor, PromptProcessingBatchResult, PromptProcessingRequest, PromptProcessingResult,
)

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
    filler_profile: str = ""
    base_prompt: str = ""
    prompt_processing: PromptProcessingResult | None = None


class FaceGenerator:
    """Generate n portraits with a shared style and interchangeable backend.

    Construction and generation_prompts() without process=True never contact a
    provider. Optional conversation processing runs before image generation.
    A new instance owns a unique batch; repeating generate() reuses its files.
    seed controls sampled descriptions, not the provider's image randomness.
    """

    def __init__(
        self,
        generator_backend: str | ImageGenerationBackend = "openai",
        output_dir: str | Path = "./facesGenerated",
        n: int = 1,
        *,
        description: str | ParsedDescription | None = None,
        filler_profiles: Sequence[str] | None = None,
        parser: str | DescriptionParser = "schema",
        reference_image: str | Path | None = None,
        similarity: float = 75.0,
        write_manifest: bool = True,
        attribute_schema: FaceAttributeSchema | None = None,
        seed: int | None = None,
        model: str | None = None,
        size: str = "1024x1024",
        quality: str = "high",
        output_format: str = "png",
        conversation_model: str | None = None,
        prompt_processor: PromptProcessor | None = None,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer.")
        if description is not None and not isinstance(description, ParsedDescription):
            if not isinstance(description, str) or not description.strip():
                raise ValueError("description must be a non-empty string, ParsedDescription, or None.")
        self.filler_profiles: tuple[str, ...] = ()
        if filler_profiles is not None:
            if isinstance(filler_profiles, (str, bytes)) or not isinstance(filler_profiles, Sequence):
                raise ValueError("filler_profiles must be a sequence of n distinct non-empty strings.")
            if len(filler_profiles) != n:
                raise ValueError("filler_profiles must contain exactly n profiles, one per generated portrait.")
            if any(not isinstance(profile, str) or not profile.strip() for profile in filler_profiles):
                raise ValueError("filler_profiles must contain distinct non-empty strings.")
            profiles = tuple(profile.strip() for profile in filler_profiles)
            if len({" ".join(profile.split()).casefold() for profile in profiles}) != n:
                raise ValueError("filler_profiles must contain distinct non-empty strings.")
            if description is None:
                raise ValueError("filler_profiles requires an explicit eyewitness description.")
            self.filler_profiles = profiles
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
        if conversation_model is not None:
            if prompt_processor is not None:
                raise ValueError("Supply either conversation_model or prompt_processor, not both.")
            if not isinstance(conversation_model, str) or not conversation_model.strip():
                raise ValueError("conversation_model must be a non-empty model name or None.")
            from .API.openai_prompt_processor import OpenAIPromptProcessor
            prompt_processor = OpenAIPromptProcessor(model=conversation_model)
        if prompt_processor is not None and not callable(getattr(prompt_processor, "process", None)):
            raise TypeError("prompt_processor must implement process(request).")
        self.prompt_processor = prompt_processor
        self._processed_prompts: dict[int, tuple[PromptProcessingRequest, PromptProcessingResult]] = {}
        self._processed_batch_requests: tuple[PromptProcessingRequest, ...] | None = None
        self._processing_signature: tuple[int, object, bool] | None = None
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

    def generation_prompts(self, *, process: bool = False) -> list[str]:
        """Preview local/cached prompts; process=True may call the text provider.

        Explicit processing caches the final prompts for later generate() calls.
        Neither variant creates image files or calls the image provider.
        """
        return [prompt for prompt, _, _ in self._prepared_plans(process=process)]

    @property
    def prompt_processing_results(self) -> list[PromptProcessingResult]:
        """Successful conversation results, in portrait order, without API calls."""
        return [self._processed_prompts[index][1] for index in sorted(self._processed_prompts)]

    def _plan_faces(self) -> list[tuple[str, str, dict[str, str]]]:
        rng = random.Random(self.seed)
        plans = []
        for index in range(1, self.n + 1):
            parsed = self.parsed_description
            if self.generation_mode == "random_description":
                parsed = _sample_description(self.attribute_schema, rng)
            needs_description = self.description is not None or self.write_manifest and self.reference_image is None
            if self.prompt_processor is None and needs_description and not parsed.model_text.strip():
                raise ValueError("Description parser produced empty model input; choose another parser or provide supported attributes.")
            attributes = parsed.schema_attributes
            prompt = self._portrait_prompt(parsed, attributes, index)
            plans.append((prompt, parsed.model_text, attributes))
        return plans

    def _prepared_plans(self, *, process: bool) -> list[tuple[str, str, dict[str, str]]]:
        if self._processed_batch_requests is not None and self.n != len(self._processed_batch_requests):
            raise ValueError("Generation conditions changed after prompt processing; create a new FaceGenerator.")
        plans = self._plan_faces()
        if self.prompt_processor is None:
            if self._processed_prompts:
                raise ValueError("Generation conditions changed after prompt processing; create a new FaceGenerator.")
            return plans
        batch_process = getattr(self.prompt_processor, "process_batch", None)
        use_batch = bool(self.description and self.description.strip()) and callable(batch_process)
        signature = (id(self.prompt_processor), getattr(self.prompt_processor, "model", None), use_batch)
        if self._processing_signature is not None and signature != self._processing_signature:
            raise ValueError("Generation conditions changed after prompt processing; create a new FaceGenerator.")
        if process:
            # Reject missing references and unsupported inputs before any paid call.
            for index, (prompt, _, _) in enumerate(plans, 1):
                self.backend.validate_request(self._request(index, prompt))
        requests = []
        for index, (base_prompt, description, attributes) in enumerate(plans, 1):
            processing_request = PromptProcessingRequest(
                original_description=self.description if self.description is not None else description,
                parsed_description=description,
                attributes=dict(attributes),
                base_prompt=base_prompt,
                portrait_instructions=_PORTRAIT_INSTRUCTIONS,
                filler_profile=self.filler_profiles[index - 1] if self.filler_profiles else "",
                reference_image=self.reference_image,
                similarity=self.similarity if self.reference_image is not None and self.description is None else None,
                index=index, total=self.n,
                attributes_authoritative=self.parser_name == "preparsed" or self.generation_mode == "random_description",
            )
            cached = self._processed_prompts.get(index)
            if cached is not None and cached[0] != processing_request:
                raise ValueError("Generation conditions changed after prompt processing; create a new FaceGenerator.")
            requests.append(processing_request)
        if self._processed_batch_requests is not None and tuple(requests) != self._processed_batch_requests:
            raise ValueError("Generation conditions changed after prompt processing; create a new FaceGenerator.")
        if use_batch:
            if process and self._processed_batch_requests is None:
                snapshot = deepcopy(tuple(requests))
                batch = batch_process(tuple(requests))
                if not isinstance(batch, PromptProcessingBatchResult):
                    raise TypeError("prompt_processor.process_batch() must return PromptProcessingBatchResult.")
                if len(batch.profiles) != self.n:
                    raise ValueError("Prompt processing must return exactly n supplementary profiles.")
                # Commit the complete plan only after every result has validated.
                completed = {}
                for request, profile in zip(snapshot, batch.profiles):
                    result = PromptProcessingResult(
                        prompt=_assemble_profile_prompt(request, profile),
                        summary=batch.summary, constraints=batch.constraints, conflicts=batch.conflicts,
                        model=batch.model, provider=batch.provider,
                        additional_description=profile, strategy="batch_profiles",
                    )
                    completed[request.index] = (request, result)
                self._processed_prompts = completed
                self._processed_batch_requests = snapshot
                self._processing_signature = signature
            return [
                (self._processed_prompts[index][1].prompt if index in self._processed_prompts else base,
                 description, attributes)
                for index, (base, description, attributes) in enumerate(plans, 1)
            ]
        prepared = []
        for processing_request, (base_prompt, description, attributes) in zip(requests, plans):
            index = processing_request.index
            cached = self._processed_prompts.get(index)
            if cached is None and process:
                result = self.prompt_processor.process(processing_request)
                if not isinstance(result, PromptProcessingResult):
                    raise TypeError("prompt_processor.process() must return PromptProcessingResult.")
                # Keep experiment-wide composition instructions even if a rewrite omits them.
                prompt = result.prompt.strip()
                if _PORTRAIT_INSTRUCTIONS not in prompt:
                    prompt += "\n\n" + _PORTRAIT_INSTRUCTIONS
                result = replace(result, prompt=prompt)
                cached = (deepcopy(processing_request), result)
                self._processed_prompts[index] = cached
                self._processing_signature = signature
            prepared.append((cached[1].prompt if cached is not None else base_prompt, description, attributes))
        return prepared

    def _portrait_prompt(self, parsed: ParsedDescription, attributes: dict[str, str], index: int) -> str:
        schema = FaceDescriptionSchema.from_attributes(parsed.original_text, attributes)
        description = "; ".join(filter(None, (
            parsed.model_text,
            *(value for value in attributes.values() if value not in parsed.model_text),
        )))
        profile = self.filler_profiles[index - 1] if self.filler_profiles else ""
        if profile:
            description += (
                f"\n\nAdditional design requirements for this portrait: {profile}\n"
                "Apply these requirements only to features not specified in the eyewitness description. "
                "All features specified in the eyewitness description take priority"
            )
        subject = {"male": "man", "female": "woman", "non-binary": "person"}.get(
            schema.gender, "man" if self.description is not None else "person",
        )
        if self.reference_image is not None:
            prompt = f"Make a photo of a {subject} based on this image"
            if description:
                prompt += f", with the following features: {description}"
            prompt = prompt.rstrip(". ") + ". " + _PORTRAIT_INSTRUCTIONS
            if self.description is None:
                prompt += (
                    f" Requested resemblance: {self.similarity:g}/100,"
                    " where 0 means a distinctly dissimilar face and 100 means the greatest resemblance."
                    " This is generation guidance, not a measured similarity score."
                )
            return prompt
        if profile:
            return (
                f"Make a photo of an under average-looking {subject}, "
                f"matching the following description: {description}. "
                + _PORTRAIT_INSTRUCTIONS
            )
        prompt = schema.to_generation_prompt(description=description)
        if self.generation_mode == "random":
            prompt += " Randomly choose this person's facial features; create a new, distinct identity."
        return prompt

    def generate(self, *, dry_run: bool = False) -> list[GeneratedFace]:
        """Create this batch and record each successful image, if requested."""
        plans = self._prepared_plans(process=not dry_run)
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
            processed = self._processed_prompts.get(request.index)
            result = GeneratedFace(
                request.index, str(request.output_path), request.prompt, self.model, self.provider,
                self._image_id(request.index), self.batch_id, _prompt_hash(request.prompt), description, attributes,
                filler_profile=self.filler_profiles[request.index - 1] if self.filler_profiles else "",
                base_prompt=processed[0].base_prompt if processed is not None else request.prompt,
                prompt_processing=processed[1] if processed is not None else None,
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
        processing = result.prompt_processing
        label_source = "requested" if result.attributes else "unspecified"
        if result.attributes and processing is not None and self.parser_name != "preparsed" and self.generation_mode != "random_description":
            label_source = "parsed_attribute_hints"
        schema = FaceDescriptionSchema.from_attributes(result.description, result.attributes)
        record = dataset.records_from_generated(
            [result], schema=schema, verbal_description=result.description,
            created_at_utc=self.created_at_utc, source_mode=self.generation_mode,
        )[0]
        record.update(
            generation_mode=self.generation_mode,
            original_description=self.description or "",
            parser=self.parser_name,
            attribute_label_source=label_source,
            attributes_verified=False,
            attribute_schema_name=self.attribute_schema.name,
            attribute_schema_version=self.attribute_schema.version,
            reference_image_path=str(self.reference_image or ""),
            requested_similarity=self.similarity if self.reference_image is not None else "",
            description_seed=str(self.seed) if self.generation_mode == "random_description" else "",
            filler_profile=result.filler_profile,
            base_prompt=result.base_prompt,
            prompt_processing_mode="conversation" if processing is not None else "direct",
            prompt_processing_strategy=processing.strategy if processing is not None else "",
            prompt_processing_additional_description=processing.additional_description if processing is not None else "",
            prompt_processing_provider=processing.provider if processing is not None else "",
            prompt_processing_model=processing.model if processing is not None else "",
            prompt_processing_summary=processing.summary if processing is not None else "",
            prompt_processing_constraints=json.dumps(processing.constraints if processing is not None else (), ensure_ascii=False),
            prompt_processing_conflicts=json.dumps(processing.conflicts if processing is not None else (), ensure_ascii=False),
        )
        return record

    def to_manifest(self) -> pd.DataFrame:
        """Return this batch's records, whether or not disk recording is enabled."""
        return pd.DataFrame([self._manifest_record(result) for result in self.results])

    def __repr__(self) -> str:
        return f"FaceGenerator(mode={self.generation_mode!r}, n={self.n}, backend={self.provider!r}, output_dir={str(self.output_dir)!r})"


def _assemble_profile_prompt(request: PromptProcessingRequest, profile: str) -> str:
    """Keep the full source wording and add one planned appearance locally."""
    # Parser hints must not introduce an appearance constraint of their own.
    gender = request.attributes.get("gender") if request.attributes_authoritative else None
    subject = {"male": "man", "female": "woman", "non-binary": "person"}.get(
        gender, "person",
    )
    if request.reference_image is not None:
        opening = f"Make a photo of a {subject} based on this image, with the following features: "
    else:
        opening = "An unretouched candid photograph of an adult in an everyday street setting. "
    parts = [opening + request.original_description]
    if request.attributes_authoritative and request.attributes:
        parts.append(
            "Explicit experimental controls (these take precedence over conflicting description traits): "
            + json.dumps(request.attributes, ensure_ascii=False, sort_keys=True)
        )
    parts.append(profile)
    if request.reference_image is not None:
        parts.append("Use the reference image only for features compatible with the description, controls, and additions above.")
    parts.append(request.portrait_instructions)
    return "\n\n".join(parts)


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
