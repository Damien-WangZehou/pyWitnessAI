from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "studies.qualtrics_cfd_clip"

from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, ConfigDict, Field
    from starlette.concurrency import run_in_threadpool
except ImportError as exc:  # pragma: no cover - exercised only before deployment deps are installed
    raise ImportError(
        "The Qualtrics CLIP service requires FastAPI and Uvicorn. "
        "Install them with: python -m pip install 'pyWitnessAI[clip,excel,api]'"
    ) from exc

from .profile import (
    DEFAULT_PROFILE_PATH,
    DEFAULT_STUDY_VERSION,
    SCORING_CONTRACT_VERSION,
)
from .selector import CfdWhiteMaleLineupSelector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServiceSettings:
    index_dir: Path
    repository_root: Path
    image_root: Path
    signing_secret: str
    model_name: str | None
    device: str | None
    image_url_ttl_seconds: int
    excluded_target_ids: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    allowed_origin_regex: str | None
    inference_timeout_seconds: float
    max_concurrent_inference: int
    result_cache_size: int
    model_revision: str | None = None
    local_model_path: Path | None = None
    selection_profile_path: Path = DEFAULT_PROFILE_PATH

    @classmethod
    def from_environment(
        cls,
        *,
        repository_root: str | Path | None = None,
        selection_profile_path: str | Path | None = None,
    ) -> "ServiceSettings":
        """Read deployment settings, with environment overrides taking priority.

        Optional paths override the study defaults; environment values take priority.
        """
        repository_root = Path(os.getenv(
            "CFD_REPOSITORY_ROOT",
            repository_root if repository_root is not None else PROJECT_ROOT,
        )).resolve()
        image_root = Path(
            os.getenv(
                "CFD_IMAGE_ROOT",
                repository_root / "data" / "CFD Version 3.0" / "Images",
            )
        ).resolve()
        index_dir = Path(
            os.getenv(
                "CFD_CLIP_INDEX_DIR",
                repository_root
                / "build"
                / "qualtrics_cfd_clip"
                / "clip_index_vit_l_14",
            )
        ).resolve()
        origin_text = os.getenv(
            "QUALTRICS_ALLOWED_ORIGINS",
            "https://bristolexppsych.eu.qualtrics.com",
        )
        origins = tuple(value.strip() for value in origin_text.split(",") if value.strip())
        excluded_target_ids = tuple(
            value.strip()
            for value in os.getenv("CFD_EXCLUDED_TARGET_IDS", "").split(",")
            if value.strip()
        )
        origin_regex = os.getenv(
            "QUALTRICS_ALLOWED_ORIGIN_REGEX",
            "",
        ).strip()
        return cls(
            index_dir=index_dir,
            repository_root=repository_root,
            image_root=image_root,
            signing_secret=os.getenv("LINEUP_SIGNING_SECRET", ""),
            model_name=os.getenv("CFD_CLIP_MODEL") or None,
            device=os.getenv("CFD_CLIP_DEVICE") or None,
            image_url_ttl_seconds=int(os.getenv("LINEUP_IMAGE_TTL_SECONDS", "7200")),
            excluded_target_ids=excluded_target_ids,
            allowed_origins=origins,
            allowed_origin_regex=origin_regex or None,
            inference_timeout_seconds=float(os.getenv("CLIP_INFERENCE_TIMEOUT_SECONDS", "25")),
            max_concurrent_inference=int(os.getenv("CLIP_MAX_CONCURRENT_INFERENCE", "1")),
            result_cache_size=int(os.getenv("CLIP_RESULT_CACHE_SIZE", "512")),
            model_revision=os.getenv("CFD_CLIP_MODEL_REVISION") or None,
            local_model_path=(
                Path(os.environ["CFD_CLIP_LOCAL_MODEL_PATH"]).resolve()
                if os.getenv("CFD_CLIP_LOCAL_MODEL_PATH")
                else None
            ),
            selection_profile_path=Path(
                os.getenv(
                    "CFD_CLIP_SELECTION_PROFILE",
                    selection_profile_path if selection_profile_path is not None else DEFAULT_PROFILE_PATH,
                )
            ).resolve(),
        )


class LineupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    participant_id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    study_version: str = Field(default=DEFAULT_STUDY_VERSION, min_length=1, max_length=128)


def create_app(
    settings: ServiceSettings | None = None,
    selector: CfdWhiteMaleLineupSelector | None = None,
) -> FastAPI:
    configured = settings or ServiceSettings.from_environment()
    if configured.image_url_ttl_seconds < 900:
        raise ValueError("LINEUP_IMAGE_TTL_SECONDS must be at least 900 seconds.")
    if configured.inference_timeout_seconds <= 0:
        raise ValueError("CLIP_INFERENCE_TIMEOUT_SECONDS must be greater than zero.")
    if configured.max_concurrent_inference < 1:
        raise ValueError("CLIP_MAX_CONCURRENT_INFERENCE must be at least 1.")
    if configured.result_cache_size < 1:
        raise ValueError("CLIP_RESULT_CACHE_SIZE must be at least 1.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active_selector = selector
        if active_selector is None:
            if len(configured.signing_secret) < 16:
                raise RuntimeError(
                    "LINEUP_SIGNING_SECRET must be set to a random value of at least 16 characters."
                )
            active_selector = CfdWhiteMaleLineupSelector.load(
                configured.index_dir,
                signing_secret=configured.signing_secret,
                repository_root=configured.repository_root,
                image_root=configured.image_root,
                model_name=configured.model_name,
                model_revision=configured.model_revision,
                local_model_path=configured.local_model_path,
                selection_profile_path=configured.selection_profile_path,
                device=configured.device,
                excluded_target_ids=configured.excluded_target_ids,
            )
        app.state.selector = active_selector
        app.state.inflight = {}
        app.state.inflight_lock = asyncio.Lock()
        app.state.result_cache = OrderedDict()
        app.state.cleanup_tasks = set()
        yield

    app = FastAPI(
        title="CFD CLIP Qualtrics lineup service",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(configured.allowed_origins),
        allow_origin_regex=configured.allowed_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        active: CfdWhiteMaleLineupSelector = app.state.selector
        return {
            "status": "ok",
            "model_name": active.model_name,
            "index_sha256": active.index_sha256,
            "pool_version": active.pool_version,
            "pool_size": len(active.pool_manifest),
            "lineup_size": active.lineup_size,
            "model_revision": getattr(active, "model_revision", ""),
            "model_snapshot_sha256": getattr(
                active, "model_snapshot_sha256", ""
            ),
            "model_snapshot_match_verified": getattr(
                active, "model_snapshot_match_verified", False
            ),
            "candidate_embedding_sha256": getattr(
                active, "candidate_embedding_sha256", ""
            ),
            "media_manifest_sha256": getattr(
                active, "media_manifest_sha256", ""
            ),
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "scoring_contract_sha256": getattr(
                active, "scoring_contract_sha256", ""
            ),
            "selection_profile_id": getattr(
                getattr(active, "selection_profile", None), "profile_id", ""
            ),
            "selection_profile_sha256": getattr(
                getattr(active, "selection_profile", None), "profile_sha256", ""
            ),
            "study_version": getattr(
                getattr(active, "selection_profile", None), "study_version", ""
            ),
            "ready": True,
        }

    @app.post("/v1/lineups")
    async def create_lineup(request: LineupRequest) -> JSONResponse:
        active: CfdWhiteMaleLineupSelector = app.state.selector
        active_profile = getattr(active, "selection_profile", None)
        expected_study = getattr(active_profile, "study_version", request.study_version)
        if request.study_version != expected_study:
            raise HTTPException(
                status_code=409,
                detail="Survey study_version does not match the deployed CLIP profile.",
            )
        profile_hash = str(getattr(active_profile, "profile_sha256", ""))
        contract_hash = str(getattr(active, "scoring_contract_sha256", ""))
        request_key = hashlib.sha256(
            "\0".join(
                [
                    request.participant_id.strip(),
                    " ".join(request.description.split()),
                    request.study_version,
                    SCORING_CONTRACT_VERSION,
                    contract_hash,
                    profile_hash,
                ]
            ).encode("utf-8")
        ).hexdigest()

        async def finalise_task(key: str, task: asyncio.Task[Any]) -> None:
            try:
                result = await task
            except Exception:
                result = None
            async with app.state.inflight_lock:
                if app.state.inflight.get(key) is task:
                    app.state.inflight.pop(key, None)
                if result is not None:
                    app.state.result_cache[key] = result
                    app.state.result_cache.move_to_end(key)
                    while len(app.state.result_cache) > configured.result_cache_size:
                        app.state.result_cache.popitem(last=False)

        async with app.state.inflight_lock:
            cached = app.state.result_cache.get(request_key)
            if cached is not None:
                app.state.result_cache.move_to_end(request_key)
                selection_task = None
            else:
                selection_task = app.state.inflight.get(request_key)
                if selection_task is None:
                    if len(app.state.inflight) >= configured.max_concurrent_inference:
                        raise HTTPException(
                            status_code=503,
                            detail="CLIP inference is busy; retry the request.",
                        )
                    selection_task = asyncio.create_task(
                        run_in_threadpool(
                            active.select,
                            request.description,
                            request.participant_id,
                            study_version=request.study_version,
                        )
                    )
                    app.state.inflight[request_key] = selection_task
                    cleanup = asyncio.create_task(finalise_task(request_key, selection_task))
                    app.state.cleanup_tasks.add(cleanup)
                    cleanup.add_done_callback(app.state.cleanup_tasks.discard)
        try:
            if cached is None:
                result = await asyncio.wait_for(
                    asyncio.shield(selection_task),
                    timeout=configured.inference_timeout_seconds,
                )
            else:
                result = cached
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "CLIP inference did not finish within the response window; "
                    "it is still running and a retry will reuse the same task."
                ),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (RuntimeError, FloatingPointError) as exc:
            raise HTTPException(
                status_code=503,
                detail="CLIP selection failed; retry the request.",
            ) from exc

        expires = int(time.time()) + configured.image_url_ttl_seconds
        payload = result.to_dict()
        for item in payload["items"]:
            item["image_url"] = active.signed_image_path(item["image_id"], expires)
        return JSONResponse(
            payload,
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.get("/v1/images/{image_id}")
    async def get_image(
        image_id: str,
        expires: int = Query(...),
        signature: str = Query(..., min_length=64, max_length=64),
    ) -> Response:
        active: CfdWhiteMaleLineupSelector = app.state.selector
        if not active.verify_image_token(image_id, expires, signature):
            raise HTTPException(status_code=403, detail="Invalid or expired image URL.")
        try:
            filename, payload = active.image_bytes(image_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Image not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="Image integrity verification failed.",
            ) from exc
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        remaining = max(0, expires - int(time.time()))
        return Response(
            content=payload,
            media_type=media_type,
            headers={
                "Cache-Control": f"private, max-age={remaining}",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    return app


app = create_app()


def main() -> None:
    """Run the frozen CFD study API with Uvicorn."""

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The experiment service requires Uvicorn. "
            "Install it with: python -m pip install 'pyWitnessAI[clip,excel,api]'"
        ) from exc

    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
