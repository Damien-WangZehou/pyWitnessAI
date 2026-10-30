"""Shared provider-call pacing, concurrency, bounded retries and private metrics.

This module uses only the standard library. Limits are per runtime/process;
share one instance across all jobs using an account. Request pacing does not
estimate or enforce token quotas. Record actual usage to size those quotas.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
from pathlib import Path
import random
from threading import BoundedSemaphore, Lock
import time
from uuid import uuid4


_current = ContextVar("generation_runtime", default=None)


def ordered_map(function, values, *, max_workers=1):
    """Bounded workers, stable output order and a separate context per call."""
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")
    if max_workers == 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="portrait") as executor:
        futures = [executor.submit(copy_context().run, function, value) for value in values]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def current_runtime():
    scope = _current.get()
    return scope[0] if scope is not None else None


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


@dataclass(frozen=True)
class CallPolicy:
    max_concurrent: int = 2
    requests_per_minute: float | None = None
    timeout_seconds: float = 240.0
    max_retries: int = 0
    max_retry_wait_seconds: float = 30.0

    def __post_init__(self):
        for name, minimum in (("max_concurrent", 1), ("max_retries", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        for name in ("timeout_seconds", "max_retry_wait_seconds", "requests_per_minute"):
            value = getattr(self, name)
            if value is None and name == "requests_per_minute":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


class _Gate:
    def __init__(self, policy):
        self.policy = policy
        self.slots = BoundedSemaphore(policy.max_concurrent)
        self.lock = Lock()
        self.next_start = 0.0

    @contextmanager
    def enter(self, deadline):
        remaining = None if deadline is None else max(0, deadline - time.monotonic())
        if not self.slots.acquire(timeout=remaining):
            raise TimeoutError("Generation deadline expired while waiting for a provider slot.")
        try:
            while True:
                with self.lock:
                    now = time.monotonic()
                    delay = max(0, self.next_start - now)
                    if deadline is not None and now + delay >= deadline:
                        raise TimeoutError("Generation deadline expired while waiting for the rate limit.")
                    if delay == 0:
                        interval = 60.0 / self.policy.requests_per_minute if self.policy.requests_per_minute else 0
                        self.next_start = now + interval
                        break
                time.sleep(delay)
            yield
        finally:
            self.slots.release()


def _retry_delay(exc, attempt):
    # Only explicit transient rejection codes are safe to replay here. A read
    # timeout or connection loss may hide a successful, chargeable generation.
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = _field(error, "code") or getattr(exc, "code", None)
    if (status, code) not in {
        (429, "rate_limit_exceeded"), (429, "slow_down"), (503, "server_is_overloaded"),
    }:
        return None
    headers = _field(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("Retry-After") or headers.get("retry-after")
    delay = None
    if value is not None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    if delay is None or not math.isfinite(delay) or delay < 0:
        delay = min(2 ** attempt, 8)
    return delay + random.uniform(0, 0.25)


class GenerationRuntime:
    """One application-wide controller, with separate image and text policies.

    ``scope`` associates calls with a private metrics file and an optional
    whole-operation deadline. The scope is propagated by explicit copy_context
    when callers use worker threads. Explicitly supplied clients remain owned
    by the caller; lazily cached clients are closed by ``close``.
    """

    def __init__(self, *, image_policy=None, text_policy=None):
        self.gates = {"image": _Gate(image_policy or CallPolicy()), "text": _Gate(text_policy or CallPolicy())}
        self._client_lock = Lock()
        self._clients = {}
        self._closed = False

    def client(self, name, factory):
        with self._client_lock:
            if self._closed:
                raise RuntimeError("The generation runtime is closed.")
            if name not in self._clients:
                self._clients[name] = factory()
            return self._clients[name]

    @contextmanager
    def scope(self, metrics_path=None, *, timeout_seconds=None):
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive.")
        started = time.monotonic()
        report = {"started_at": datetime.now(timezone.utc).isoformat(), "calls": []}
        deadline = None if timeout_seconds is None else started + timeout_seconds
        parent = _current.get()
        if parent is not None and parent[0] is self and parent[3] is not None:
            deadline = parent[3] if deadline is None else min(deadline, parent[3])
        token = _current.set((self, report, Lock(), deadline))
        failed = False
        try:
            yield report
        except BaseException:
            failed = True
            raise
        finally:
            _current.reset(token)
            report["elapsed_seconds"] = time.monotonic() - started
            if metrics_path is not None:
                try:
                    write_json_atomic(metrics_path, report)
                except Exception:
                    if not failed:
                        raise
                    logging.getLogger(__name__).exception("Could not save failed-operation metrics")

    def call(self, kind, model, invoke):
        """Invoke ``invoke(timeout_seconds)`` and record metadata, never prompts."""
        if self._closed:
            raise RuntimeError("The generation runtime is closed.")
        gate = self.gates[kind]
        policy = gate.policy
        scope = _current.get()
        scope = scope if scope is not None and scope[0] is self else None
        deadline = scope[3] if scope is not None else None
        retry_wait = 0.0
        for attempt in range(policy.max_retries + 1):
            queued = time.monotonic()
            event = {"kind": kind, "model": model, "attempt": attempt + 1}
            try:
                with gate.enter(deadline):
                    started = time.monotonic()
                    timeout = policy.timeout_seconds
                    if deadline is not None:
                        timeout = min(timeout, deadline - started)
                        if timeout <= 0:
                            raise TimeoutError("Generation deadline expired before dispatch.")
                    event["wait_seconds"] = started - queued
                    try:
                        result = invoke(timeout)
                        event["status"] = "ok"
                        usage = _field(result, "usage")
                        event["usage"] = {
                            name: value for name in ("input_tokens", "output_tokens", "total_tokens")
                            if isinstance(value := _field(usage, name), int) and not isinstance(value, bool)
                        }
                        request_id = _field(result, "_request_id")
                        if isinstance(request_id, str):
                            event["request_id"] = request_id
                        return result
                    finally:
                        event["call_seconds"] = time.monotonic() - started
            except Exception as exc:
                event.update(status="error", error_type=type(exc).__name__)
                status = getattr(exc, "status_code", None)
                if isinstance(status, int):
                    event["http_status"] = status
                request_id = getattr(exc, "request_id", None)
                if isinstance(request_id, str):
                    event["request_id"] = request_id
                delay = _retry_delay(exc, attempt) if attempt < policy.max_retries else None
                if delay is None or retry_wait + delay > policy.max_retry_wait_seconds:
                    raise
                if deadline is not None and time.monotonic() + delay >= deadline:
                    raise
                event["retry_wait_seconds"] = delay
                retry_wait += delay
            finally:
                if scope is not None:
                    with scope[2]:
                        scope[1]["calls"].append(event)
            time.sleep(delay)

    def close(self):
        with self._client_lock:
            self._closed = True
            clients, self._clients = self._clients, {}
        for client in clients.values():
            client.close()
