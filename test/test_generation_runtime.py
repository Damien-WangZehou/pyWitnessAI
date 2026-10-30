"""Provider control is tested entirely with local fakes and virtual time."""

from concurrent.futures import ThreadPoolExecutor
import base64
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pandas as pd
from PIL import Image
import pytest

from pyWitnessAI.API.openai import OpenAIImageBackend, OPENAI_IMAGE_MODELS
from pyWitnessAI.API.openai_prompt_processor import OpenAIPromptProcessor
from pyWitnessAI.FaceGenerator import FaceGenerator
from pyWitnessAI.face_generation_backend import ImageGenerationBackend, ImageGenerationRequest
from pyWitnessAI.generation_runtime import CallPolicy, GenerationRuntime, current_runtime, ordered_map
import pyWitnessAI.generation_runtime as runtime_module


class Clock:
    now = 100.0

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime_module.time, "sleep", clock.advance)
    monkeypatch.setattr(runtime_module.random, "uniform", lambda *_: 0)
    return clock


class Rejected(Exception):
    def __init__(self, status=429, code="rate_limit_exceeded", delay="3"):
        super().__init__("private prompt or server details")
        self.status_code = status
        self.body = {"error": {"code": code}}
        self.response = SimpleNamespace(headers={} if delay is None else {"Retry-After": delay})


def test_pacing_retries_usage_and_separate_text_gate(clock, tmp_path):
    runtime = GenerationRuntime(image_policy=CallPolicy(requests_per_minute=30, max_retries=2))
    starts = []
    def invoke(timeout):
        starts.append(clock.now)
        if len(starts) == 1:
            raise Rejected()
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30), _request_id="req-local")
    with runtime.scope(tmp_path / "metrics.json") as report:
        runtime.call("image", "image-model", invoke)
        runtime.call("text", "text-model", lambda _: None)
        runtime.call("image", "image-model", invoke)
    assert starts == [100, 103, 105]
    assert len(report["calls"]) == 4
    assert report["calls"][0]["retry_wait_seconds"] == 3
    assert report["calls"][1]["usage"]["total_tokens"] == 30
    assert report["calls"][2]["wait_seconds"] == 0
    saved = (tmp_path / "metrics.json").read_text()
    assert "private prompt" not in saved
    assert "req-local" in saved
    assert current_runtime() is None


@pytest.mark.parametrize("error", [
    Rejected(code="insufficient_quota"), Rejected(code="billing_hard_limit_reached"),
    Rejected(status=500, code="server_error"), TimeoutError("unknown completion"),
    Rejected(delay="60"),
])
def test_unsafe_or_excessively_delayed_retries_are_not_replayed(clock, error):
    runtime = GenerationRuntime(image_policy=CallPolicy(max_retries=2, max_retry_wait_seconds=10))
    calls = []
    def invoke(_):
        calls.append(True)
        raise error
    with pytest.raises(type(error)):
        runtime.call("image", "model", invoke)
    assert calls == [True]
    assert clock.now == 100


def test_deadline_includes_rate_wait_and_retry_wait(clock):
    runtime = GenerationRuntime(image_policy=CallPolicy(requests_per_minute=6, max_retries=2))
    calls = []
    with runtime.scope(timeout_seconds=5):
        runtime.call("image", "model", lambda timeout: calls.append(timeout))
        with pytest.raises(TimeoutError):
            runtime.call("image", "model", lambda _: pytest.fail("Expired work must not start"))
    assert calls == [5]
    runtime = GenerationRuntime(image_policy=CallPolicy(max_retries=2))
    with runtime.scope(timeout_seconds=2):
        with pytest.raises(Rejected):
            runtime.call("image", "model", lambda _: (_ for _ in ()).throw(Rejected(delay="3")))
    assert clock.now == 100


def test_concurrency_cap_is_shared_by_independent_jobs():
    runtime = GenerationRuntime(image_policy=CallPolicy(max_concurrent=2))
    lock, entered, release = threading.Lock(), threading.Event(), threading.Event()
    active = peak = 0
    def invoke(_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                entered.set()
        try:
            assert release.wait(5)
        finally:
            with lock:
                active -= 1
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(runtime.call, "image", "model", invoke) for _ in range(8)]
        try:
            assert entered.wait(5)
        finally:
            release.set()
        for future in futures:
            future.result()
    assert peak == 2


def test_worker_context_keeps_metrics_with_its_job(tmp_path):
    runtime = GenerationRuntime()
    def job(number):
        with runtime.scope(tmp_path / f"{number}.json"):
            return ordered_map(lambda value: runtime.call("image", str(number), lambda _: value), range(4), max_workers=2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(job, [1, 2])) == [list(range(4)), list(range(4))]
    for number in (1, 2):
        report = json.loads((tmp_path / f"{number}.json").read_text())
        assert len(report["calls"]) == 4
        assert {call["model"] for call in report["calls"]} == {str(number)}


def test_image_edit_retry_reopens_reference_and_disables_sdk_retry(clock, tmp_path):
    streams, options = [], []
    class Client:
        def __init__(self):
            self.images = self
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self
        def edit(self, image, **_):
            streams.append(image)
            assert image.read() == b"reference"
            if len(streams) == 1:
                raise Rejected(delay="1")
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"image"))])
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference")
    runtime = GenerationRuntime(image_policy=CallPolicy(max_retries=1))
    request = ImageGenerationRequest("prompt", tmp_path / "output.png", "image-model", "1024x1024", "high", "png", 1, 1, reference)
    with runtime.scope():
        OpenAIImageBackend(Client()).generate(request)
    assert request.output_path.read_bytes() == b"image"
    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert options == [{"max_retries": 0, "timeout": 240.0}] * 2


def test_parallel_image_failure_keeps_order_and_reuses_all_successes(tmp_path):
    class Backend(ImageGenerationBackend):
        default_model = "offline"
        def __init__(self):
            self.calls = []
            self.lock = threading.Lock()
            self.failed = False
        def generate(self, request):
            with self.lock:
                self.calls.append(request.index)
                fail = request.index == 2 and not self.failed
                if fail:
                    self.failed = True
            if fail:
                request.output_path.write_bytes(b"partial")
                raise RuntimeError("offline failure")
            Image.new("RGB", (8, 8), (request.index, 100, 20)).save(request.output_path)
    backend = Backend()
    generator = FaceGenerator(backend, tmp_path / "faces", 4, description="Short hair", max_workers=2)
    with pytest.raises(RuntimeError, match="offline failure"):
        generator.generate()
    assert [face.index for face in generator.results] == [1, 3, 4]
    originals = {face.index: Path(face.path).read_bytes() for face in generator.results}
    assert len(pd.read_csv(generator.output_dir / "manifest.csv")) == 3
    results = generator.generate()
    assert [face.index for face in results] == [1, 2, 3, 4]
    assert sorted(backend.calls) == [1, 2, 2, 3, 4]
    assert all(Path(face.path).read_bytes() == originals[face.index] for face in results if face.index != 2)
    rows = pd.read_csv(generator.output_dir / "manifest.csv")
    assert len(rows) == 4 and rows.image_id.is_unique


def test_runtime_clients_are_lazy_shared_and_owned():
    closed, created = [], []
    runtime = GenerationRuntime()
    def factory():
        created.append(True)
        return SimpleNamespace(close=lambda: closed.append(True))
    with ThreadPoolExecutor(max_workers=8) as executor:
        clients = list(executor.map(lambda _: runtime.client("provider", factory), range(16)))
    assert len({id(client) for client in clients}) == 1 and created == [True]
    runtime.close()
    runtime.close()
    assert closed == [True]
    with pytest.raises(RuntimeError, match="closed"):
        runtime.client("provider", factory)


def test_model_names_are_not_accidentally_concatenated():
    assert {"gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-2"} <= set(OPENAI_IMAGE_MODELS)


def test_batch_prompt_and_parallel_images_share_one_client_and_record_usage(tmp_path, monkeypatch):
    options, clients, closed = [], [], []
    class Client:
        def __init__(self):
            self.responses = self.images = self
        def with_options(self, **kwargs):
            options.append(kwargs)
            return self
        def create(self, **kwargs):
            assert kwargs["store"] is False
            return SimpleNamespace(status="completed", model="text-model", output_text=json.dumps({
                "profiles": ["An oval face", "A square face"], "summary": "Profiles", "constraints": [], "conflicts": [],
            }), usage=SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150))
        def generate(self, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(b64_json=base64.b64encode(b"offline image"))],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=20, total_tokens=30))
        def close(self):
            closed.append(True)
    def factory():
        clients.append(Client())
        return clients[-1]
    monkeypatch.setattr(OpenAIImageBackend, "_create_client", staticmethod(factory))
    monkeypatch.setattr(OpenAIPromptProcessor, "_create_client", staticmethod(factory))
    runtime = GenerationRuntime()
    generator = FaceGenerator(output_dir=tmp_path / "faces", description="A man with short hair", n=2,
                              conversation_model="text-model", max_workers=2)
    with runtime.scope(tmp_path / "calls.json"):
        assert len(generator.generate()) == 2
    runtime.close()
    report = json.loads((tmp_path / "calls.json").read_text())
    assert len(clients) == 1 and closed == [True]
    assert [event["kind"] for event in report["calls"]].count("text") == 1
    assert [event["kind"] for event in report["calls"]].count("image") == 2
    assert sum(event["usage"]["total_tokens"] for event in report["calls"]) == 210
    assert all(option["max_retries"] == 0 for option in options)


def test_metrics_write_failure_does_not_hide_original_generation_failure(tmp_path, monkeypatch):
    runtime = GenerationRuntime()
    def fail_write(*_):
        raise OSError("disk full")
    monkeypatch.setattr(runtime_module, "write_json_atomic", fail_write)
    with pytest.raises(RuntimeError, match="generation failed"):
        with runtime.scope(tmp_path / "metrics.json"):
            raise RuntimeError("generation failed")
