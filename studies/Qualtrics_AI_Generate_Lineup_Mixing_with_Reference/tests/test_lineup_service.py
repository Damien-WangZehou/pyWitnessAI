"""Study contracts verified with fake providers; never calls a paid image API."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from pyWitnessAI import FaceStandardizer

MODULE_PATH = Path(__file__).resolve().parents[1] / "lineup_service.py"
spec = importlib.util.spec_from_file_location("study_lineup_service", MODULE_PATH)
service = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = service
spec.loader.exec_module(service)


def settings(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    for i in range(8):
        Image.new("RGB", (320, 400), (i * 20, 60, 120)).save(refs / f"ref{i}.jpg")
    suspect = tmp_path / "suspect.png"
    Image.new("RGB", (512, 512), "red").save(suspect)
    return service.Settings(tmp_path / "runtime", refs, suspect, image_model="test-model", allowed_origins=("https://example.qualtrics.com",))


def request(condition="noMixing", target="targetAbsent", participant="participant_123"):
    return service.LineupRequest(participant_id=participant, description="A MAN with Short Hair.", condition=condition, target_lineup=target, study_version=service.STUDY_VERSION)


def fake_generator_factory(calls):
    class Generator:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs
            self.model = kwargs["model"]
        def generate(self):
            out = self.kwargs["output_dir"]
            out.mkdir(parents=True, exist_ok=True)
            result = []
            for i in range(self.kwargs["n"]):
                p = out / f"fake{i}.png"
                Image.new("RGB", (1024, 1024), (30 + i, 100, 200)).save(p)
                result.append(SimpleNamespace(path=p, prompt="fake processed prompt " + self.kwargs["description"]))
            return result
    return Generator


def measured_standardizer_factory(calls, portraits):
    def factory(**kwargs):
        calls.append(kwargs)
        detections = []

        def detect(image):
            # Alternate between exact eye scaling and the canvas-coverage limit.
            fraction = 0.12 if len(detections) % 2 == 0 else 0.25
            detections.append(image.size)
            width, height = image.size
            return [[width * (0.5 - fraction / 2), height * 0.4],
                    [width * (0.5 + fraction / 2), height * 0.4],
                    [width * 0.5, height * 0.5],
                    [width * 0.45, height * 0.6],
                    [width * 0.55, height * 0.6]]

        processor = FaceStandardizer(detector=detect, **kwargs)

        def transform(source):
            portrait = processor.transform(source)
            portraits.append(portrait)
            return portrait

        return SimpleNamespace(transform=transform)
    return factory


@pytest.mark.parametrize("condition", ["mixingFaces", "noMixing"])
@pytest.mark.parametrize("target", ["targetPresent", "targetAbsent"])
def test_four_conditions_and_reference_isolation(tmp_path, condition, target):
    config = settings(tmp_path)
    calls = []
    standardizer_calls, portraits = [], []
    result = service.generate_lineup(
        request(condition, target), config, "testlineup",
        generator_factory=fake_generator_factory(calls),
        standardizer_factory=measured_standardizer_factory(standardizer_calls, portraits),
    )
    service._validate_lineup({**result, "target_lineup": target})
    count = 5 if target == "targetPresent" else 6
    assert sum(c["n"] for c in calls) == count
    assert all(c["parser"].parse("MiXeD Case").model_text == "MiXeD Case" for c in calls)
    assert all(c["description"] == "A MAN with Short Hair." for c in calls)
    assert all(c["size"] == "1024x1024" for c in calls)
    expected_standardization = {"size": [256, 320], "alignment": "center", "eye_distance": 0.23, "fill_canvas": True, "background": "auto", "remove_background": False}
    assert result["metadata"]["standardization"] == expected_standardization
    assert standardizer_calls == [{**expected_standardization, "size": (256, 320)}]
    assert len(portraits) == 6
    assert all(portrait.mask.getextrema() == (255, 255) for portrait in portraits)
    assert [portrait.metadata["eye_distance_clamped"] for portrait in portraits] == [False, True] * 3
    # The requested distance is 256 * 0.23; canvas coverage raises it to 80
    # for every second portrait rather than introducing padded borders.
    assert [portrait.metadata["output_eye_distance"] for portrait in portraits] == pytest.approx([58.88, 80] * 3)
    if condition == "mixingFaces":
        assert len(calls) == count
        assert len({c["reference_image"] for c in calls}) == count
        assert all(c["reference_image"].parent == config.reference_dir for c in calls)
    else:
        assert len(calls) == 1 and calls[0]["reference_image"] is None
    roles = [i["role"] for i in result["items"]]
    assert roles.count("suspect") == int(target == "targetPresent")
    for item in result["items"]:
        file = config.storage_dir / item["image_url"].lstrip("/")
        with Image.open(file) as image:
            assert image.size == (256, 320) and image.mode == "RGB"
            if item["role"] == "suspect":
                assert image.getpixel((128, 160)) == (255, 0, 0)
    manifest = json.loads((config.storage_dir / "private/testlineup/manifest.json").read_text())
    assert len(manifest["generation_prompts"]) == count
    assert manifest["standardization_measurements"] == [
        {"position": item["position"], "image_id": item["image_id"],
         "output_eye_distance": portrait.metadata["output_eye_distance"],
         "eye_distance_clamped": portrait.metadata["eye_distance_clamped"]}
        for item, portrait in zip(result["items"], portraits)
    ]
    assert result["metadata"]["prompt_processing"] is True
    assert result["metadata"]["prompt_mode"] == "conversation"
    assert manifest["result"]["metadata"] == result["metadata"]


def ready_result(req):
    return {"items": [{"position": n, "image_id": f"id{n}", "image_url": f"/media/job/{n}.png", "role": "suspect" if n == 3 and req.target_lineup == "targetPresent" else "filler"} for n in range(1, 7)], "metadata": {"model": "fake"}}


def wait_done(manager, lineup_id):
    end = time.monotonic() + 5
    while time.monotonic() < end:
        result = manager.get(lineup_id)
        if result["status"] in {"ready", "error"}:
            return result
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def test_concurrent_idempotency_conflicts_and_restart(tmp_path):
    config = settings(tmp_path)
    seen = []
    def pipeline(req, cfg, job):
        seen.append(job)
        return ready_result(req)
    manager = service.JobManager(config, pipeline)
    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: manager.submit(request()), range(24)))
    ids = {r["lineup_id"] for r in results}
    assert len(ids) == 1
    lineup_id = ids.pop()
    assert wait_done(manager, lineup_id)["status"] == "ready"
    assert seen == [lineup_id]
    with pytest.raises(service.ConflictError):
        manager.submit(request(target="targetPresent"))
    manager.close()
    second = service.JobManager(config, pipeline)
    second.recover()
    assert second.submit(request())["status"] == "ready"
    assert seen == [lineup_id]
    second.close()


def test_provider_error_persists_without_regeneration(tmp_path):
    config = settings(tmp_path)
    calls = []
    def pipeline(*args):
        calls.append(True)
        raise RuntimeError("private provider details")
    manager = service.JobManager(config, pipeline)
    first = manager.submit(request())
    failed = wait_done(manager, first["lineup_id"])
    assert failed["status"] == "error"
    assert "private provider" not in json.dumps(failed)
    assert manager.submit(request())["status"] == "error"
    assert len(calls) == 1
    manager.close()


def test_interrupted_generation_is_not_reissued(tmp_path):
    config = settings(tmp_path)
    gate = threading.Event()
    manager = service.JobManager(config, lambda req, *_: (gate.wait(5), ready_result(req))[1])
    response = manager.submit(request())
    for _ in range(100):
        if manager.get(response["lineup_id"])["status"] == "generating":
            break
        time.sleep(0.01)
    # Simulate a recovered process's store operation without issuing another API.
    recovered = service.JobManager(config, lambda *_: pytest.fail("must not regenerate"))
    recovered.recover()
    assert recovered.get(response["lineup_id"])["error_code"] == "generation_interrupted"
    recovered.close()
    gate.set()
    manager.close()


def test_http_contract_cors_and_private_media(tmp_path):
    config = settings(tmp_path)
    app = service.create_app(config, lambda req, *_: ready_result(req))
    with TestClient(app) as client:
        assert client.get("/").json()["health"] == "/health"
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "study_version": service.STUDY_VERSION}
        healthz = client.get("/healthz")
        assert healthz.status_code == 200
        assert healthz.json() == health.json()
        response = client.post("/v1/lineups", json=request().model_dump(), headers={"Origin": "https://example.qualtrics.com"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://example.qualtrics.com"
        job = response.json()
        assert client.get(job["poll_url"]).status_code == 200
        assert client.get("/v1/lineups/unknown").status_code == 404
        assert client.get("/private/jobs.sqlite3").status_code == 404
        bad = request().model_dump()
        bad["target_lineup"] = "absent"
        assert client.post("/v1/lineups", json=bad).status_code == 422
