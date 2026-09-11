"""HTTP adapter for lineup construction and iterative witness feedback.

Use create_app() in a trusted local, single-process service. Input paths are
relative to image_root (or absolute paths inside it). Only exported portraits
are served; research annotations and server paths never enter responses.
Creation POSTs allocate a new resource each time and must not be retried
automatically. Feedback retries use the submitted round_number and choice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4

from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel, ConfigDict, Field
except ImportError as exc:  # pragma: no cover - optional installation boundary
    raise ImportError(
        "The lineup HTTP adapter requires python -m pip install 'pyWitnessAI[api]'."
    ) from exc

from ..ImageCatalog import image_paths
from ..LineupSession import LineupSession, create_lineup


class GenerationOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, min_length=1, max_length=128)
    size: str = Field(default="1024x1024", min_length=1, max_length=32)
    quality: str = Field(default="medium", min_length=1, max_length=32)
    output_format: Literal["png", "jpeg", "webp"] = "png"


class LineupOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    size: int = Field(default=6, strict=True, ge=2, le=24)
    rows: int = Field(default=2, strict=True, ge=1, le=24)
    similarity: float = Field(default=75, ge=0, le=100, allow_inf_nan=False)
    seed: int | None = Field(default=None, strict=True)
    parser: Literal["raw", "lenient", "strict", "schema"] = "raw"
    generation_options: GenerationOptions | None = None


class LineupRequest(LineupOptions):
    suspect: str | None = Field(default=None, min_length=1, max_length=4096)
    description: str | None = Field(default=None, min_length=1, max_length=2000)
    reference_image: str | None = Field(default=None, min_length=1, max_length=4096)
    filler_pool: str | list[str] | None = None
    target: Literal["present", "absent", "unknown"] | None = None
    order: Literal["random", "given"] = "random"


class SessionRequest(LineupOptions):
    description: str = Field(min_length=1, max_length=2000)
    reference_image: str | None = Field(default=None, min_length=1, max_length=4096)
    max_rounds: int = Field(default=10, strict=True, ge=1, le=50)


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(strict=True, ge=1, le=50)
    choice: int = Field(strict=True, ge=1, le=24)
    confirmed: bool = Field(strict=True)


def create_app(*, output_dir, image_root, generator_backend="openai", allowed_origins=()) -> FastAPI:
    """Create a local adapter without loading an index or model.

    All operations on one session share a lock. Multiple Uvicorn workers or
    separate processes must not write to the same output_dir concurrently.
    allowed_origins explicitly enables browser origins for UI integrations;
    the default permits no cross-origin browser requests.
    """
    destination, inputs = Path(output_dir).resolve(), Path(image_root).resolve()
    if not inputs.is_dir():
        raise ValueError("image_root must be an existing directory.")
    destination.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="pyWitnessAI lineup workflows", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=list(allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"], allow_credentials=False,
    )
    sessions, locks = {}, {}
    registry_lock = Lock()

    def resource_directory(kind: str, resource_id: UUID) -> Path:
        directory = (destination / kind / str(resource_id)).resolve()
        if not directory.is_relative_to(destination):
            raise HTTPException(404, "Resource not found.")
        return directory

    def session_lock(session_id: UUID):
        with registry_lock:
            return locks.setdefault(session_id, Lock())

    def load_session(session_id: UUID):
        if session_id not in sessions:
            directory = resource_directory("sessions", session_id)
            if not directory.is_dir():
                raise HTTPException(404, "Session not found.")
            sessions[session_id] = _invoke(
                LineupSession.load, directory, generator_backend=generator_backend,
            )
        return sessions[session_id]

    @app.post("/v1/lineups", status_code=201)
    def new_lineup(request: LineupRequest):
        options = request.model_dump()
        options["suspect"] = _input_image(request.suspect, inputs)
        options["reference_image"] = _input_image(request.reference_image, inputs)
        options["filler_pool"] = _input_pool(request.filler_pool, inputs)
        lineup_id = uuid4()
        directory = resource_directory("lineups", lineup_id)
        lineup = _invoke(
            create_lineup, **options, generator_backend=generator_backend,
            output_dir=directory / "generated",
        )
        _invoke(lineup.save, directory / "presentation")
        _invoke(
            (directory / "audit.json").write_text,
            json.dumps(lineup.to_dict(include_roles=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"lineup_id": str(lineup_id), **_presentation(
            lineup, f"/v1/lineups/{lineup_id}/images",
        )}

    @app.get("/v1/lineups/{lineup_id}/images/{position}")
    def lineup_image(lineup_id: UUID, position: int):
        directory = resource_directory("lineups", lineup_id) / "presentation"
        try:
            manifest = json.loads((directory / "lineup.json").read_text(encoding="utf-8"))
            size = manifest["size"]
        except (OSError, ValueError, KeyError):
            raise HTTPException(404, "Image not found.") from None
        return _member_image(directory, position, size, destination)

    @app.post("/v1/sessions", status_code=201)
    def new_session(request: SessionRequest):
        options = request.model_dump()
        options["reference_image"] = _input_image(request.reference_image, inputs)
        session_id = uuid4()
        with session_lock(session_id):
            session = _invoke(
                LineupSession, **options, generator_backend=generator_backend,
                output_dir=resource_directory("sessions", session_id),
            )
            _invoke(session.start)
            sessions[session_id] = session
            return _session_presentation(session_id, session)

    @app.get("/v1/sessions/{session_id}")
    def get_session(session_id: UUID):
        with session_lock(session_id):
            return _session_presentation(session_id, load_session(session_id))

    @app.post("/v1/sessions/{session_id}/feedback")
    def feedback(session_id: UUID, request: FeedbackRequest):
        with session_lock(session_id):
            session = load_session(session_id)
            _invoke(session.respond, **request.model_dump(), invalid_status=409)
            return _session_presentation(session_id, session)

    @app.post("/v1/sessions/{session_id}/stop")
    def stop(session_id: UUID):
        with session_lock(session_id):
            session = load_session(session_id)
            _invoke(session.stop, invalid_status=409)
            return _session_presentation(session_id, session)

    @app.get("/v1/sessions/{session_id}/rounds/{round_number}/images/{position}")
    def session_image(session_id: UUID, round_number: int, position: int):
        with session_lock(session_id):
            session = load_session(session_id)
            round_ = next((item for item in session.history if item.number == round_number), None)
            if round_ is None:
                raise HTTPException(404, "Image not found.")
            return _member_image(round_.directory, position, len(round_.lineup), destination)

    return app


def _input_path(value: str, root: Path) -> Path:
    try:
        path = Path(value)
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise ValueError
        return path
    except (OSError, ValueError):
        raise HTTPException(422, "Input paths must exist inside image_root.") from None


def _input_image(value: str | None, root: Path) -> Path | None:
    if value is None:
        return None
    path = _input_path(value, root)
    try:
        with Image.open(path) as opened:
            opened.verify()
    except (OSError, ValueError, SyntaxError):
        raise HTTPException(422, "Input must be a readable image inside image_root.") from None
    return path


def _input_pool(value, root: Path) -> list[Path] | None:
    if value is None:
        return None
    if isinstance(value, str):
        path = _input_path(value, root)
        values = _invoke(image_paths, path) if path.is_dir() else [value]
    else:
        values = value
    return [_input_image(str(item), root) for item in values]


def _invoke(function, *args, invalid_status=422, **kwargs):
    try:
        return function(*args, **kwargs)
    except (ValueError, TypeError):
        detail = "Feedback conflicts with the current round or session state." if invalid_status == 409 else "Invalid lineup or session options."
        raise HTTPException(invalid_status, detail) from None
    except Exception:
        logging.getLogger(__name__).exception("Lineup workflow failed.")
        raise HTTPException(503, "The operation could not finish; inspect the local service and retry feedback if applicable.") from None


def _presentation(lineup, image_prefix: str) -> dict:
    presentation = lineup.to_dict()
    for member in presentation["members"]:
        member.pop("image")
        member["image_url"] = f"{image_prefix}/{member['position']}"
    return presentation


def _session_presentation(session_id: UUID, session: LineupSession) -> dict:
    payload = {"session_id": str(session_id), "status": session.status, "current": None}
    if session.current is not None:
        round_ = session.current
        payload["current"] = {"number": round_.number, **_presentation(
            round_.lineup, f"/v1/sessions/{session_id}/rounds/{round_.number}/images",
        )}
        if round_.feedback is not None:
            payload["current"]["feedback"] = {
                "choice": round_.feedback["choice"], "confirmed": round_.feedback["confirmed"],
            }
    return payload


def _member_image(directory: Path, position: int, size: int, root: Path) -> Response:
    if not 1 <= position <= size:
        raise HTTPException(404, "Image not found.")
    path = (directory / f"member_{position:02d}.png").resolve()
    if not path.is_relative_to(root):
        raise HTTPException(404, "Image not found.")
    try:
        content = path.read_bytes()
    except OSError:
        raise HTTPException(404, "Image not found.") from None
    return Response(content, media_type="image/png", headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    })
