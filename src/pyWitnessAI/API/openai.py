"""OpenAI Images API adapter for the generic image-generation contract."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from ..face_generation_backend import ImageGenerationBackend, ImageGenerationRequest
from ..generation_runtime import current_runtime


DEFAULT_IMAGE_MODEL = "gpt-image-2.5-sunburst"
OPENAI_IMAGE_MODELS = (
    "gpt-image-2.5-flare",
    "gpt-image-2.5-sunburst",
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
)

__all__ = ["DEFAULT_IMAGE_MODEL", "OPENAI_IMAGE_MODELS", "OpenAIImageBackend"]


class OpenAIImageBackend(ImageGenerationBackend):
    """OpenAI Images API backend."""

    provider = "openai"
    default_model = DEFAULT_IMAGE_MODEL
    known_models = OPENAI_IMAGE_MODELS
    supports_reference_image = True

    def __init__(self, client=None, *, runtime=None) -> None:
        self.client = client
        self.runtime = runtime

    def generate(self, request: ImageGenerationRequest) -> None:
        self.validate_request(request)
        runtime = self.runtime or current_runtime()
        client = self.client
        if client is None:
            client = runtime.client("openai", self._create_client) if runtime else self._create_client()
        arguments = dict(
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            output_format=request.output_format,
        )
        def invoke(timeout=None):
            active = client.with_options(max_retries=0, timeout=timeout) if runtime else client
            if request.reference_image is None:
                return active.images.generate(**arguments)
            # Each eligible retry needs a fresh stream, starting at byte zero.
            with Path(request.reference_image).open("rb") as reference_image:
                return active.images.edit(image=reference_image, **arguments)

        response = runtime.call("image", request.model, invoke) if runtime else invoke()
        images = getattr(response, "data", None) or []
        image_base64 = getattr(images[0], "b64_json", None) if images else None
        if not image_base64:
            raise RuntimeError("OpenAI image response did not include base64 image data.")
        request.output_path.write_bytes(base64.b64decode(image_base64))

    @staticmethod
    def _create_client():
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError("Set OPENAI_API_KEY before calling generate().")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI image generation requires the OpenAI SDK. "
                "Install it with: python -m pip install 'pyWitnessAI[generation]'"
            ) from exc
        return OpenAI()

