"""Single-process, durable background jobs without HTTP or provider dependencies.

The SQL column names retain the original study database format. The public
interface uses arbitrary idempotency keys and JSON payloads, not study models.
One running manager must own a database; recovery assumes its predecessor has
stopped. Running paid work is never replayed automatically.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from threading import RLock
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ConflictError(ValueError):
    """An idempotency key was reused with different input."""


class QueueFullError(RuntimeError):
    """The limit on queued plus running work has been reached."""


class SQLiteJobStore:
    """Atomic reservations and state transitions, compatible with jobs.sqlite3."""

    def __init__(self, database):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS jobs (
                lineup_id TEXT PRIMARY KEY, participant_hash TEXT UNIQUE NOT NULL,
                request_hash TEXT NOT NULL, request_json TEXT NOT NULL,
                response_json TEXT NOT NULL, status TEXT NOT NULL
            )""")
            con.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")

    @contextmanager
    def connection(self):
        # sqlite3.Connection.__exit__ commits/rolls back, but does NOT close.
        con = sqlite3.connect(self.database, timeout=30)
        try:
            with con:
                yield con
        finally:
            con.close()

    def reserve(self, payload, key, response_factory, max_pending):
        raw = _json(payload)
        fingerprint = _digest(raw)
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            prior = con.execute(
                "SELECT lineup_id,request_hash,response_json FROM jobs WHERE participant_hash=?",
                (_digest(key),),
            ).fetchone()
            if prior:
                if prior[1] != fingerprint:
                    raise ConflictError("This idempotency key already has a job with different input.")
                return prior[0], json.loads(prior[2]), False
            pending = con.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','generating')").fetchone()[0]
            if pending >= max_pending:
                raise QueueFullError("The generation queue is full. Retry later with the same input.")
            job_id = uuid4().hex
            now = utc_now()
            response = {**response_factory(job_id), "status": "queued", "created_at": now, "updated_at": now}
            con.execute(
                "INSERT INTO jobs (lineup_id,participant_hash,request_hash,request_json,response_json,status) VALUES (?,?,?,?,?,?)",
                (job_id, _digest(key), fingerprint, raw, _json(response), "queued"),
            )
        return job_id, response, True

    def get(self, job_id):
        with self.connection() as con:
            row = con.execute("SELECT response_json FROM jobs WHERE lineup_id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def claim(self, job_id):
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT request_json,response_json FROM jobs WHERE lineup_id=? AND status='queued'", (job_id,),
            ).fetchone()
            if row is None:
                return None
            response = json.loads(row[1])
            now = utc_now()
            response.update(status="generating", started_at=now, updated_at=now)
            con.execute("UPDATE jobs SET status='generating',response_json=? WHERE lineup_id=?", (_json(response), job_id))
        return json.loads(row[0]), response

    def finish(self, job_id, response, *, expected="generating"):
        response = dict(response)
        now = utc_now()
        response.update(finished_at=now, updated_at=now)
        end = datetime.fromisoformat(now)
        created = datetime.fromisoformat(response["created_at"])
        started = datetime.fromisoformat(response.get("started_at", now))
        response["timing"] = {
            "queue_seconds": max(0, (started - created).total_seconds()),
            "run_seconds": max(0, (end - started).total_seconds()),
            "total_seconds": max(0, (end - created).total_seconds()),
        }
        with self.connection() as con:
            # A stale worker must not overwrite a recovery decision.
            cursor = con.execute(
                "UPDATE jobs SET status=?,response_json=? WHERE lineup_id=? AND status=?",
                (response["status"], _json(response), job_id, expected),
            )
        return bool(cursor.rowcount)

    def recover(self):
        with self.connection() as con:
            con.execute("BEGIN IMMEDIATE")
            rows = con.execute("SELECT lineup_id,response_json,status FROM jobs WHERE status IN ('queued','generating')").fetchall()
            for job_id, raw, status in rows:
                if status == "generating":
                    response = json.loads(raw)
                    response.update(status="error", error_code="generation_interrupted", updated_at=utc_now(),
                                    error="Generation was interrupted. Researcher review is required; it will not be regenerated automatically.")
                    con.execute("UPDATE jobs SET status='error',response_json=? WHERE lineup_id=?", (_json(response), job_id))
        return [job_id for job_id, _, status in rows if status == "queued"]


class DurableJobManager:
    """Run JSON-in/JSON-out jobs; adapters supply domain validation and errors."""

    def __init__(self, database, pipeline, *, max_workers=2, max_pending=40, on_error=None):
        for name, value in (("max_workers", max_workers), ("max_pending", max_pending)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.store = SQLiteJobStore(database)
        self.database = self.store.database
        self.pipeline = pipeline
        self.max_pending = max_pending
        self.on_error = on_error
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="generation")
        self._lock = RLock()
        self._scheduled = set()
        self._closed = False
        self._recovered = False

    def recover(self):
        with self._lock:
            self._check_open()
            if self._recovered:
                return
            # Never mistake this manager's already-running work for a crash.
            if self._scheduled:
                raise RuntimeError("Recover before submitting jobs.")
            for job_id in self.store.recover():
                self._schedule(job_id)
            self._recovered = True

    def submit(self, payload, *, key, response_factory):
        with self._lock:
            self._check_open()
            job_id, response, created = self.store.reserve(payload, key, response_factory, self.max_pending)
            if created:
                try:
                    self._schedule(job_id)
                except Exception:
                    self.store.finish(job_id, {**response, "status": "error", "error_code": "scheduling_failed",
                                               "error": "The job could not be scheduled."}, expected="queued")
                    raise
        return response

    def _check_open(self):
        if self._closed:
            raise RuntimeError("The job manager is closed.")

    def _schedule(self, job_id):
        if job_id in self._scheduled:
            return
        self._scheduled.add(job_id)
        try:
            future = self._executor.submit(self._run, job_id)
        except Exception:
            self._scheduled.discard(job_id)
            raise
        future.add_done_callback(lambda completed: self._forget(job_id, completed))

    def _forget(self, job_id, future):
        with self._lock:
            self._scheduled.discard(job_id)
        if not future.cancelled() and future.exception() is not None:
            exc = future.exception()
            logging.getLogger(__name__).error("Job %s could not persist its state", job_id,
                                             exc_info=(type(exc), exc, exc.__traceback__))

    def _run(self, job_id):
        claimed = self.store.claim(job_id)
        if claimed is None:
            return
        payload, response = claimed
        try:
            result = self.pipeline(payload, job_id)
            # Validate serialization inside the failure boundary.
            _json(result)
            final = {**response, **result, "status": "ready"}
            for field in ("created_at", "started_at"):
                final[field] = response[field]
        except Exception as exc:
            if self.on_error is not None:
                try:
                    self.on_error(job_id, exc)
                except Exception:
                    logging.getLogger(__name__).exception("Could not save private error details for job %s", job_id)
            final = {**response, "status": "error", "error_code": "generation_failed",
                     "error": "Generation failed. Researcher review is required; it will not be regenerated automatically."}
        self.store.finish(job_id, final)

    def get(self, job_id):
        return self.store.get(job_id)

    def close(self):
        with self._lock:
            self._closed = True
        # Do not hold the lock needed by completion callbacks during shutdown.
        self._executor.shutdown(wait=True)
