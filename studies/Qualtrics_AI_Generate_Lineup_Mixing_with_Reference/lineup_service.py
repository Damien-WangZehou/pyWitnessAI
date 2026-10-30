"""Live, description-specific AI lineups for the 2 x 2 Qualtrics study.

Run ONE uvicorn worker; job state and the no-regeneration rule persist in SQLite.
Private source portraits, raw generations and manifests are never mounted as media.
No API request is made by importing this module or by GET /health.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import threading
from typing import Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    max_pending_jobs: int = 40
    allowed_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            storage_dir=Path(os.getenv("LINEUP_STORAGE_DIR", str(HERE / "runtime"))).resolve(),
            reference_dir=Path(os.getenv("REFERENCE_DIR", str(ROOT / "data" / "ReferenceFaces_CFD_WhiteMale"))).resolve(),
            suspect_path=Path(os.getenv("SUSPECT_PATH", str(HERE / "perp_refer_to_AI_generated_image.png"))).resolve(),
            image_model=os.getenv("IMAGE_MODEL") or None,
            max_concurrent_jobs=max(1, int(os.getenv("MAX_CONCURRENT_JOBS", "2"))),
            max_pending_jobs=max(1, int(os.getenv("MAX_PENDING_JOBS", "40"))),
            allowed_origins=tuple(v.strip() for v in os.getenv("ALLOWED_ORIGINS", "").split(",") if v.strip()),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ConflictError(ValueError):
    pass


class QueueFullError(RuntimeError):
    pass


class JobManager:
    """Durable one-job-per-participant store, with a bounded generation executor."""

    def __init__(self, settings: Settings, pipeline: Callable | None = None):
        self.settings = settings
        self.pipeline = pipeline or generate_lineup
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="lineup")
        settings.storage_dir.mkdir(parents=True, exist_ok=True)
        (settings.storage_dir / "media").mkdir(exist_ok=True)
        (settings.storage_dir / "private").mkdir(exist_ok=True)
        self.database = settings.storage_dir / "jobs.sqlite3"
        with self._connection() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (
                lineup_id TEXT PRIMARY KEY, participant_hash TEXT UNIQUE NOT NULL,
                request_hash TEXT NOT NULL, request_json TEXT NOT NULL,
                response_json TEXT NOT NULL, status TEXT NOT NULL
            )""")

    def _connection(self):
        return sqlite3.connect(self.database, timeout=30)

    def recover(self):
        """Resume unstarted work; interrupted paid work requires researcher review."""
        with self._lock, self._connection() as con:
            rows = con.execute("SELECT lineup_id,response_json,status FROM jobs WHERE status IN ('queued','generating')").fetchall()
            for lineup_id, raw, state in rows:
                if state == "generating":
                    response = json.loads(raw)
                    response.update(status="error", error="Generation was interrupted. The researcher must review this attempt; it will not be regenerated automatically.", error_code="generation_interrupted", updated_at=_now())
                    con.execute("UPDATE jobs SET status='error',response_json=? WHERE lineup_id=?", (_json(response), lineup_id))
        for lineup_id, _, state in rows:
            if state == "queued":
                self._executor.submit(self._run, lineup_id)

    def submit(self, request: LineupRequest) -> dict:
        payload = request.model_dump()
        participant_hash = _digest(request.participant_id)
        request_json = _json(payload)
        request_hash = _digest(request_json)
        with self._lock, self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            prior = con.execute("SELECT request_hash,response_json FROM jobs WHERE participant_hash=?", (participant_hash,)).fetchone()
            if prior:
                if prior[0] != request_hash:
                    raise ConflictError("This participant already has a lineup with different description or condition. A retry cannot change it.")
                return json.loads(prior[1])
            pending = con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','generating')").fetchone()[0]
            if pending >= self.settings.max_pending_jobs:
                raise QueueFullError("The generation queue is full. Retry later with the same participant ID.")
            lineup_id = uuid4().hex
            response = {
                "status": "queued", "lineup_id": lineup_id,
                "poll_url": f"/v1/lineups/{lineup_id}",
                "condition": request.condition, "target_lineup": request.target_lineup,
                "study_version": STUDY_VERSION, "created_at": _now(), "updated_at": _now(),
            }
            con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)", (lineup_id, participant_hash, request_hash, request_json, _json(response), "queued"))
        self._executor.submit(self._run, lineup_id)
        return response

    def get(self, lineup_id: str) -> dict | None:
        with self._connection() as con:
            row = con.execute("SELECT response_json FROM jobs WHERE lineup_id=?", (lineup_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def _run(self, lineup_id: str):
        with self._lock, self._connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT request_json,response_json,status FROM jobs WHERE lineup_id=?", (lineup_id,)).fetchone()
            if not row or row[2] != "queued":
                return
            request = LineupRequest.model_validate_json(row[0])
            response = json.loads(row[1])
            response.update(status="generating", updated_at=_now())
            con.execute("UPDATE jobs SET status='generating',response_json=? WHERE lineup_id=?", (_json(response), lineup_id))
        try:
            result = self.pipeline(request, self.settings, lineup_id)
            response.update(result, status="ready", updated_at=_now())
            _validate_lineup(response)
        except Exception as exc:
            # Exception text can contain provider/request details. Store privately only.
            error_path = self.settings.storage_dir / "private" / lineup_id
            error_path.mkdir(parents=True, exist_ok=True)
            (error_path / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            response = {k: v for k, v in response.items() if k not in ("items", "metadata", "suspect_position")}
            response.update(status="error", error="The lineup could not be generated. This attempt will not be regenerated automatically; please contact the researcher.", error_code="generation_failed", updated_at=_now())
        with self._lock, self._connection() as con:
            con.execute("UPDATE jobs SET status=?,response_json=? WHERE lineup_id=?", (response["status"], _json(response), lineup_id))

    def close(self):
        self._executor.shutdown(wait=True)


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
    # Imports are deferred so schema checks, health checks and fake-provider tests
    # do not need or contact OpenAI and do not download any model.
    if generator_factory is None:
        from pyWitnessAI import FaceGenerator
        from pyWitnessAI.API.openai import OpenAIImageBackend
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY must be set in the server environment.")
        backend = OpenAIImageBackend(client=OpenAI(max_retries=0, timeout=240.0))
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
            generators.append(generator_factory(reference_image=reference, n=1, output_dir=private / f"generated_{index}", **common))
    else:
        generators.append(generator_factory(reference_image=None, n=count, output_dir=private / "generated", **common))
    faces = [face for generator in generators for face in generator.generate()]
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
    (private / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def create_app(settings: Settings | None = None, pipeline: Callable | None = None) -> FastAPI:
    config = settings or Settings.from_env()
    manager = JobManager(config, pipeline)

    @asynccontextmanager
    async def lifespan(application):
        manager.recover()
        yield
        manager.close()

    application = FastAPI(title="AI-generated lineup study", lifespan=lifespan)
    application.state.jobs = manager
    if config.allowed_origins:
        application.add_middleware(CORSMiddleware, allow_origins=list(config.allowed_origins), allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type"])
    application.mount("/media", StaticFiles(directory=config.storage_dir / "media"), name="media")

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
            return manager.submit(request)
        except ConflictError as exc:
            raise HTTPException(409, str(exc)) from exc
        except QueueFullError as exc:
            raise HTTPException(503, str(exc)) from exc

    @application.get("/v1/lineups/{lineup_id}")
    def get(lineup_id: str):
        result = manager.get(lineup_id)
        if result is None:
            raise HTTPException(404, "Unknown lineup.")
        return result

    return application


app = create_app()
