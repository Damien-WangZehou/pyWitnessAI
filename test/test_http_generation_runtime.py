"""The synchronous HTTP contract uses the same provider runtime as studies."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from pyWitnessAI.API.http import create_app
from pyWitnessAI.face_generation_backend import ImageGenerationBackend
from pyWitnessAI.generation_runtime import GenerationRuntime, current_runtime


def test_http_runtime_records_private_usage_and_preserves_synchronous_response(tmp_path):
    class Backend(ImageGenerationBackend):
        default_model = "offline"
        def generate(self, request):
            runtime = current_runtime()
            assert runtime is not None
            runtime.call("image", request.model, lambda _: SimpleNamespace(usage={"total_tokens": 123}))
            Image.new("RGB", (8, 8), "blue").save(request.output_path)
    runtime = GenerationRuntime()
    output = tmp_path / "output"
    app = create_app(image_root=tmp_path, output_dir=output, generator_backend=Backend(), generation_runtime=runtime)
    with TestClient(app) as client:
        response = client.post("/v1/lineups", json={"description": "short hair", "size": 2, "rows": 1})
        assert response.status_code == 201
        assert len(response.json()["members"]) == 2
        assert "status" not in response.json()  # No conversion to a queued-job endpoint.
        files = list((output / "private" / "operations").glob("*.json"))
        assert len(files) == 1
        report = json.loads(files[0].read_text())
        assert sum(event["usage"]["total_tokens"] for event in report["calls"]) == 246
        assert client.get(f"/private/operations/{files[0].name}").status_code == 404
    # An injected runtime is caller-owned and still usable after app shutdown.
    assert runtime.call("text", "offline", lambda _: "usable") == "usable"
    runtime.close()
