"""Provider-neutral generation requests, capabilities, and backend registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping


BackendFactory = Callable[..., "ImageGenerationBackend"]

__all__ = [
    "BackendFactory",
    "ImageGenerationBackend",
    "ImageGenerationRequest",
    "available_image_generation_models",
    "available_image_generation_providers",
    "create_image_generation_backend",
    "register_image_generation_backend",
]


@dataclass(frozen=True)
class ImageGenerationRequest:
    """One generation job, optionally guided by an input image.

    Similarity is a prompt preference from 0 (most different) to 100 (most
    similar), not a measured score or an Images API parameter.
    """

    prompt: str
    output_path: Path
    model: str
    size: str
    quality: str
    output_format: str
    index: int
    total: int
    reference_image: Path | None = None
    similarity: float = 75.0


class ImageGenerationBackend:
    """Base class for image-generation providers.

    Subclasses define provider metadata and implement generate(). Generators
    call validate_request() before dispatch, including for custom backends.
    """

    provider: str = "custom"
    default_model: str | None = None
    known_models: tuple[str, ...] = ()
    supports_reference_image: bool = False

    def validate_request(self, request: ImageGenerationRequest) -> None:
        """Reject unsupported inputs before a provider performs any work."""
        try:
            similarity = float(request.similarity)
        except (TypeError, ValueError) as exc:
            raise ValueError("similarity must be a finite number between 0 and 100.") from exc
        if not math.isfinite(similarity) or not 0 <= similarity <= 100:
            raise ValueError("similarity must be a finite number between 0 and 100.")
        if request.reference_image is not None and not self.supports_reference_image:
            raise ValueError(f"Image generation provider {self.provider!r} does not support reference images.")
        if request.reference_image is not None and not Path(request.reference_image).is_file():
            raise FileNotFoundError(f"Reference image file does not exist: {request.reference_image}")

    def generate(self, request: ImageGenerationRequest) -> None:
        self.validate_request(request)
        raise NotImplementedError


_BACKEND_REGISTRY: dict[str, BackendFactory] = {}


def register_image_generation_backend(
    provider: str,
    backend: type[ImageGenerationBackend] | ImageGenerationBackend | BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register an image-generation backend under a provider name.

    The backend can be a backend class, a ready-made backend instance, or a
    factory that returns a backend instance. Classes and factories may accept
    the backend_kwargs supplied to create_image_generation_backend().
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
        raise TypeError("backend must be an ImageGenerationBackend, subclass, or factory.")


def create_image_generation_backend(provider: str, backend_kwargs: Mapping[str, object] | None = None) -> ImageGenerationBackend:
    provider_name = _normalise_provider(provider)
    if provider_name not in _BACKEND_REGISTRY:
        known = ", ".join(available_image_generation_providers())
        raise ValueError(f"Unsupported image generation provider: {provider!r}. Registered providers: {known}.")

    factory = _BACKEND_REGISTRY[provider_name]
    backend = factory(**dict(backend_kwargs or {}))
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


def _create_openai_backend(**kwargs) -> ImageGenerationBackend:
    from .API.openai import OpenAIImageBackend

    return OpenAIImageBackend(**kwargs)


register_image_generation_backend("openai", _create_openai_backend)
