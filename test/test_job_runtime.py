"""Durability, admission and restart behavior without an HTTP server or API."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time

import pytest

from pyWitnessAI.job_runtime import ConflictError, DurableJobManager, QueueFullError, SQLiteJobStore


def submit(manager, key="participant", value=1):
    return manager.submit({"value": value}, key=key, response_factory=lambda job: {"lineup_id": job})


def done(manager, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = manager.get(job)
        if result["status"] in {"ready", "error"}:
            return result
        time.sleep(0.005)
    raise AssertionError("Job did not finish")


def test_admission_is_atomic_and_duplicate_requests_bypass_full_queue(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def pipeline(payload, job):
        calls.append(job)
        entered.set()
        assert release.wait(5)
        return {"value": payload["value"]}
    manager = DurableJobManager(tmp_path / "jobs.sqlite3", pipeline, max_workers=1, max_pending=2)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: submit(manager), range(16)))
        assert len({r["lineup_id"] for r in results}) == 1
        assert entered.wait(5)
        second = submit(manager, "second")
        assert second["status"] == "queued"
        assert submit(manager)["lineup_id"] == results[0]["lineup_id"]
        with pytest.raises(QueueFullError):
            submit(manager, "third")
        with pytest.raises(ConflictError):
            submit(manager, value=2)
        release.set()
        final = done(manager, second["lineup_id"])
        assert len(calls) == 2
        assert final["started_at"] <= final["finished_at"]
        timing = final["timing"]
        assert timing["total_seconds"] == pytest.approx(timing["queue_seconds"] + timing["run_seconds"])
    finally:
        release.set()
        manager.close()


def test_recovery_is_repeatable_and_stale_completion_cannot_replace_interruption(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def pipeline(*_):
        entered.set()
        assert release.wait(5)
        return {"items": ["late paid result"]}
    manager = DurableJobManager(tmp_path / "jobs.sqlite3", pipeline)
    manager.recover()
    first = submit(manager)
    assert entered.wait(5)
    manager.recover()  # Does not mark its own active task interrupted.
    assert manager.get(first["lineup_id"])["status"] == "generating"
    recovered = DurableJobManager(manager.database, lambda *_: pytest.fail("Must not replay paid work"))
    try:
        recovered.recover()
        assert recovered.get(first["lineup_id"])["error_code"] == "generation_interrupted"
        release.set()
        manager.close()
        assert recovered.get(first["lineup_id"])["error_code"] == "generation_interrupted"
        assert "items" not in recovered.get(first["lineup_id"])
    finally:
        release.set()
        manager.close()
        recovered.close()


def test_legacy_records_are_reused_without_migration_or_regeneration(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    raw = '{"value":1}'
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    legacy = {"lineup_id": "old", "status": "ready", "created_at": "2026-01-01T00:00:00+00:00"}
    with store.connection() as con:
        con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?)", (
            "old", digest("participant"), digest(raw), raw, json.dumps(legacy), "ready",
        ))
    manager = DurableJobManager(store.database, lambda *_: pytest.fail("Must reuse the old result"))
    try:
        manager.recover()
        assert submit(manager) == legacy
    finally:
        manager.close()


def test_queued_work_survives_restart_and_does_not_run_twice(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite3")
    job, _, _ = store.reserve({"value": 1}, "participant", lambda j: {"lineup_id": j}, 2)
    calls = []
    manager = DurableJobManager(store.database, lambda payload, job: (calls.append(job), {"ok": True})[1])
    try:
        manager.recover()
        manager.recover()
        assert done(manager, job)["ok"]
        assert calls == [job]
    finally:
        manager.close()


def test_error_logging_failure_still_releases_capacity_and_closed_submit_leaves_no_job(tmp_path):
    def fail(*_):
        raise RuntimeError("private provider details")
    manager = DurableJobManager(tmp_path / "jobs.sqlite3", fail, max_pending=1, on_error=fail)
    try:
        first = submit(manager)
        final = done(manager, first["lineup_id"])
        assert final["status"] == "error"
        assert "private provider details" not in json.dumps(final)
        second = submit(manager, "second")
        assert done(manager, second["lineup_id"])["status"] == "error"
    finally:
        manager.close()
    with pytest.raises(RuntimeError, match="closed"):
        submit(manager, "after-close")
    with manager.store.connection() as con:
        assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
