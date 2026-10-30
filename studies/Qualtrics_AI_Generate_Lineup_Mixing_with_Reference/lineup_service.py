"""Live, description-specific AI lineups for the 2 x 2 Qualtrics study.

Run ONE uvicorn worker; job state and the no-regeneration rule persist in SQLite.
Private source portraits, raw generations and manifests are never mounted as media.
No API request is made by importing this module or by GET /health.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyWitnessAI.job_runtime import ConflictError, DurableJobManager, QueueFullError
from pyWitnessAI.generation_runtime import CallPolicy, GenerationRuntime, current_runtime, ordered_map, write_json_atomic

STUDY_VERSION = "ai_generated_lineup_2x2_v1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class LineupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    participant_id: str = Field(min_length=8, max_length=250)
    description: str = Field(min_length=1, max_length=2000)
    condition: Literal["mixingFaces", "noMixing"]
    target_lineup: Literal["targetPresent", "targetAbsent"]
    study_version: Literal["ai_generated_lineup_2x2_v1"]

    @field_validator("participant_id", "description")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A nonempty value is required.")
        return value.strip()


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    reference_dir: Path
    suspect_path: Path
    image_model: str | None = None
    max_concurrent_jobs: int = 2
    max_pending_jobs: int = 2
    allowed_origins: tuple[str, ...] = ()
    images_per_job: int = 2
    max_concurrent_images: int = 4
    image_requests_per_minute: float = 14.0
    provider_max_retries: int = 2
    generation_timeout_seconds: float = 300.0

    def __post_init__(self):
        for name in ("max_concurrent_jobs", "max_pending_jobs", "images_per_job", "max_concurrent_images"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not math.isfinite(self.generation_timeout_seconds) or self.generation_timeout_seconds <= 0:
            raise ValueError("generation_timeout_seconds must be finite and positive.")
        CallPolicy(max_concurrent=self.max_concurrent_images, requests_per_minute=self.image_requests_per_minute,
                   max_retries=self.provider_max_retries)

    def create_runtime(self):
        return GenerationRuntime(
            image_policy=CallPolicy(max_concurrent=self.max_concurrent_images,
                                    requests_per_minute=self.image_requests_per_minute,
                                    max_retries=self.provider_max_retries),
            text_policy=CallPolicy(max_concurrent=self.max_concurrent_jobs, max_retries=self.provider_max_retries),
        )

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            storage_dir=Path(os.getenv("LINEUP_STORAGE_DIR", str(HERE / "runtime"))).resolve(),
            reference_dir=Path(os.getenv("REFERENCE_DIR", str(ROOT / "data" / "ReferenceFaces_CFD_WhiteMale"))).resolve(),
            suspect_path=Path(os.getenv("SUSPECT_PATH", str(HERE / "perp_refer_to_AI_generated_image.png"))).resolve(),
            image_model=os.getenv("IMAGE_MODEL") or None,
            max_concurrent_jobs=int(os.getenv("MAX_CONCURRENT_JOBS", "2")),
            max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", os.getenv("MAX_CONCURRENT_JOBS", "2"))),
            allowed_origins=tuple(v.strip() for v in os.getenv("ALLOWED_ORIGINS", "").split(",") if v.strip()),
            images_per_job=int(os.getenv("IMAGES_PER_JOB", "2")),
            max_concurrent_images=int(os.getenv("MAX_CONCURRENT_IMAGES", "4")),
            image_requests_per_minute=float(os.getenv("IMAGE_REQUESTS_PER_MINUTE", "14")),
            provider_max_retries=int(os.getenv("PROVIDER_MAX_RETRIES", "2")),
            generation_timeout_seconds=float(os.getenv("GENERATION_TIMEOUT_SECONDS", "300")),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class JobManager(DurableJobManager):
    """Study rules over the shared durable job and provider-call runtimes."""

    def __init__(self, settings: Settings, pipeline: Callable | None = None):
        self.settings = settings
        self.study_pipeline = pipeline or generate_lineup
        self.runtime = settings.create_runtime()
        (settings.storage_dir / "media").mkdir(parents=True, exist_ok=True)
        (settings.storage_dir / "private").mkdir(exist_ok=True)
        super().__init__(settings.storage_dir / "jobs.sqlite3", self._generate,
                         max_workers=settings.max_concurrent_jobs, max_pending=settings.max_pending_jobs,
                         on_error=self._save_error)

    def submit(self, request: LineupRequest) -> dict:
        return super().submit(request.model_dump(), key=request.participant_id, response_factory=lambda job_id: {
            "lineup_id": job_id, "poll_url": f"/v1/lineups/{job_id}",
            "condition": request.condition, "target_lineup": request.target_lineup, "study_version": STUDY_VERSION,
        })

    def _generate(self, payload, job_id):
        request = LineupRequest.model_validate(payload)
        created = datetime.fromisoformat(self.get(job_id)["created_at"])
        remaining = self.settings.generation_timeout_seconds - (datetime.now(timezone.utc) - created).total_seconds()
        if remaining <= 0:
            raise TimeoutError("The generation deadline expired in the queue.")
        metrics = self.settings.storage_dir / "private" / job_id / "runtime.json"
        with self.runtime.scope(metrics, timeout_seconds=remaining):
            result = self.study_pipeline(request, self.settings, job_id)
            _validate_lineup({**payload, **result})
            return result

    def _save_error(self, job_id, exc):
        error_path = self.settings.storage_dir / "private" / job_id
        error_path.mkdir(parents=True, exist_ok=True)
        (error_path / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")

    def close(self):
        super().close()
        self.runtime.close()


def _validate_lineup(response: dict):
    items = response.get("items", [])
    if len(items) != 6 or [v.get("position") for v in items] != list(range(1, 7)):
        raise ValueError("Exactly six ordered members are required.")
    if len({v["image_id"] for v in items}) != 6 or len({v["image_url"] for v in items}) != 6:
        raise ValueError("Lineup members must have unique identifiers and media URLs.")
    roles = [v.get("role") for v in items]
    expected = 1 if response["target_lineup"] == "targetPresent" else 0
    if roles.count("suspect") != expected or roles.count("filler") != 6 - expected:
        raise ValueError("Lineup roles disagree with target presence.")


def generate_lineup(request: LineupRequest, settings: Settings, lineup_id: str, *, generator_factory=None, standardizer_factory=None) -> dict:
    """Generate fillers with prompt processing, then standardize all lineup members."""
    options = dict(generator_factory=generator_factory, standardizer_factory=standardizer_factory)
    if generator_factory is None and current_runtime() is None:
        runtime = settings.create_runtime()
        try:
            with runtime.scope(settings.storage_dir / "private" / lineup_id / "runtime.json",
                               timeout_seconds=settings.generation_timeout_seconds):
                return _generate_lineup(request, settings, lineup_id, **options)
        finally:
            runtime.close()
    return _generate_lineup(request, settings, lineup_id, **options)


def _generate_lineup(request, settings, lineup_id, *, generator_factory, standardizer_factory):
    # Imports are deferred so schema checks, health checks and fake-provider tests
    # do not need or contact OpenAI and do not download any model.
    if generator_factory is None:
        from pyWitnessAI import FaceGenerator
        from pyWitnessAI.API.openai import OpenAIImageBackend
        backend = OpenAIImageBackend()
        generator_factory = lambda **kwargs: FaceGenerator(generator_backend=backend, **kwargs)
    if standardizer_factory is None:
        from pyWitnessAI import FaceStandardizer
        standardizer_factory = FaceStandardizer
    from pyWitnessAI import RawDescriptionParser

    private = settings.storage_dir / "private" / lineup_id
    media = settings.storage_dir / "media" / lineup_id
    private.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(_digest(request.participant_id + ":" + STUDY_VERSION), 16))
    count = 5 if request.target_lineup == "targetPresent" else 6
    references = []
    if request.condition == "mixingFaces":
        pool = sorted(p for p in settings.reference_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if len(pool) < count:
            raise ValueError("Not enough distinct reference portraits.")
        references = rng.sample(pool, count)
    if request.target_lineup == "targetPresent" and not settings.suspect_path.is_file():
        raise FileNotFoundError("The configured suspect portrait is missing.")

    common = dict(description=request.description, parser=RawDescriptionParser(lowercase=False), size="1024x1024", quality="high", output_format="png", write_manifest=True, conversation_model="gpt-6.1-sol", prompt_processor=None)
    if settings.image_model:
        common["model"] = settings.image_model
    generators = []
    if references:
        for index, reference in enumerate(references, 1):
            generators.append(generator_factory(reference_image=reference, n=1, output_dir=private / f"generated_{index}", max_workers=1, **common))
    else:
        generators.append(generator_factory(reference_image=None, n=count, output_dir=private / "generated", max_workers=settings.images_per_job, **common))
    batches = ordered_map(lambda generator: generator.generate(), generators, max_workers=settings.images_per_job)
    faces = [face for batch in batches for face in batch]
    if len(faces) != count:
        raise ValueError("The provider did not produce the requested number of fillers.")
    members = [(Path(face.path), "filler") for face in faces]
    if request.target_lineup == "targetPresent":
        members.append((settings.suspect_path, "suspect"))
    rng.shuffle(members)
    standardization = {"size": [256, 320], "alignment": "center", "eye_distance": 0.23, "fill_canvas": True, "background": "auto", "remove_background": False}
    standardizer = standardizer_factory(**{**standardization, "size": tuple(standardization["size"])})
    items = []
    standardization_measurements = []
    for position, (source, role) in enumerate(members, 1):
        # All six images follow the same transform and encoding. UUID names avoid
        # leaking a distinct suspect filename to participants.
        image_id = uuid4().hex
        destination = media / f"{image_id}.png"
        portrait = standardizer.transform(source)
        portrait.image.convert("RGB").save(destination, format="PNG")
        items.append({"position": position, "image_id": image_id, "image_url": f"/media/{lineup_id}/{image_id}.png", "role": role})
        standardization_measurements.append({"position": position, "image_id": image_id, "output_eye_distance": portrait.metadata["output_eye_distance"], "eye_distance_clamped": portrait.metadata["eye_distance_clamped"]})
    metadata = {
        "model": str(generators[0].model), "provider": "openai",
        "prompt_mode": "conversation", "prompt_processing": True, "conversation_model": common["conversation_model"],
        "reference_image_ids": [p.name for p in references],
        "description_sha256": _digest(request.description),
        "prompts_sha256": [_digest(face.prompt) for face in faces],
        "standardization": standardization,
    }
    result = {"items": items, "metadata": metadata, "suspect_position": next((v["position"] for v in items if v["role"] == "suspect"), None)}
    manifest = {"request": request.model_dump(), "result": result, "sources_in_display_order": [str(source) for source, _ in members], "generation_prompts": [face.prompt for face in faces], "standardization_measurements": standardization_measurements, "created_at": _now()}
    write_json_atomic(private / "manifest.json", manifest)
    return result


def create_app(settings: Settings | None = None, pipeline: Callable | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(application):
        manager = JobManager(config, pipeline)
        application.state.jobs = manager
        try:
            manager.recover()
            yield
        finally:
            manager.close()

    application = FastAPI(title="AI-generated lineup study", lifespan=lifespan)
    if config.allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(config.allowed_origins), allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
    application.mount("/media", StaticFiles(directory=config.storage_dir / "media", check_dir=False), name="media")

    @application.get("/")
    def index():
        return {"service": "AI-generated lineup study", "status": "ok",
                "health": "/health", "docs": "/docs"}

    @application.get("/healthz", include_in_schema=False)
    @application.get("/health")
    def health():
        return {"status": "ok", "study_version": STUDY_VERSION}

    @application.post("/v1/lineups")
    def submit(request: LineupRequest):
        try:
            return application.state.jobs.submit(request)
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(503, str(exc)) from exc

    @application.get("/v1/lineups/{lineup_id}")
    def get(lineup_id: str):
        result = application.state.jobs.get(lineup_id)
        if result is None:
            raise HTTPException(404, "Unknown lineup.")
        return result

    return application


app = create_app()
