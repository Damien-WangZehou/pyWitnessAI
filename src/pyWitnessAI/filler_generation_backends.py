from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


DEFAULT_IMAGE_MODEL = "gpt-image-2"
OPENAI_IMAGE_MODELS = (
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "gpt-image-1-mini",
)
KNOWN_CLIP_MODEL_PREFIXES = ("clip-", "openai/clip", "laion/clip", "siglip")

BackendFactory = Callable[[], "ImageGenerationBackend"]

__all__ = [
    "BackendFactory",
    "DEFAULT_IMAGE_MODEL",
    "ImageGenerationBackend",
    "ImageGenerationRequest",
    "KNOWN_CLIP_MODEL_PREFIXES",
    "OPENAI_IMAGE_MODELS",
    "OpenAIImageBackend",
    "available_image_generation_models",
    "available_image_generation_providers",
    "create_image_generation_backend",
    "register_image_generation_backend",
]


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Single image-generation job passed from FillerGenerator to a backend."""

    prompt: str
    output_path: Path
    model: str
    size: str
    quality: str
    output_format: str
    index: int
    total: int


class ImageGenerationBackend:
    """Base class for image-generation providers.

    Subclasses only need to define provider metadata and implement generate().
    """

    provider: str = "custom"
    default_model: str | None = None
    known_models: tuple[str, ...] = ()

    def generate(self, request: ImageGenerationRequest) -> None:
        raise NotImplementedError


class OpenAIImageBackend(ImageGenerationBackend):
    """OpenAI Images API backend."""

    provider = "openai"
    default_model = DEFAULT_IMAGE_MODEL
    known_models = OPENAI_IMAGE_MODELS

    def __init__(self, client=None) -> None:
        self.client = client

    def generate(self, request: ImageGenerationRequest) -> None:
        client = self.client or self._create_client()
        response = client.images.generate(
            model=request.model,
            prompt=request.prompt,
            size=request.size,
            quality=request.quality,
            output_format=request.output_format,
        )
        image_base64 = getattr(response.data[0], "b64_json", None)
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
            raise ImportError("Install the OpenAI SDK first: python -m pip install --upgrade openai") from exc
        return OpenAI()


_BACKEND_REGISTRY: dict[str, BackendFactory] = {}


def register_image_generation_backend(
    provider: str,
    backend: type[ImageGenerationBackend] | ImageGenerationBackend | BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register an image-generation backend under a provider name.

    The backend can be a backend class, a ready-made backend instance, or a
    zero-argument factory that returns a backend instance.
    """
    provider_name = _normalise_provider(provider)
    if provider_name in _BACKEND_REGISTRY and not overwrite:
        raise ValueError(f"Image generation provider {provider_name!r} is already registered.")

    if isinstance(backend, ImageGenerationBackend):
        _BACKEND_REGISTRY[provider_name] = lambda backend=backend: backend
    elif isinstance(backend, type) and issubclass(backend, ImageGenerationBackend):
        _BACKEND_REGISTRY[provider_name] = backend
    elif callable(backend):
        _BACKEND_REGISTRY[provider_name] = backend
    else:
        raise TypeError("backend must be an ImageGenerationBackend, subclass, or zero-argument factory.")


def create_image_generation_backend(provider: str, backend_kwargs: Mapping[str, object] | None = None) -> ImageGenerationBackend:
    provider_name = _normalise_provider(provider)
    if provider_name not in _BACKEND_REGISTRY:
        known = ", ".join(available_image_generation_providers())
        raise ValueError(f"Unsupported image generation provider: {provider!r}. Registered providers: {known}.")

    factory = _BACKEND_REGISTRY[provider_name]
    kwargs = dict(backend_kwargs or {})
    if kwargs:
        backend = factory(**kwargs)  # type: ignore[misc]
    else:
        backend = factory()
    if not isinstance(backend, ImageGenerationBackend):
        raise TypeError(f"Provider {provider_name!r} did not create an ImageGenerationBackend instance.")
    return backend


def available_image_generation_providers() -> tuple[str, ...]:
    return tuple(sorted(_BACKEND_REGISTRY))


def available_image_generation_models(provider: str = "openai") -> tuple[str, ...]:
    backend = create_image_generation_backend(provider)
    return backend.known_models


def _normalise_provider(provider: str) -> str:
    provider_name = (provider or "").strip().lower()
    if not provider_name:
        raise ValueError("provider must be a non-empty string.")
    return provider_name


register_image_generation_backend("openai", OpenAIImageBackend)
