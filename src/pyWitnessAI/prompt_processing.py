"""Optional, provider-neutral preparation of a portrait generation prompt.

Processors return requested generation conditions, never verified image labels.
The summary is a short audit record of decisions, not model reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["PromptProcessingRequest", "PromptProcessingResult", "PromptProcessor"]


@dataclass(frozen=True)
class PromptProcessingRequest:
    """One portrait's source wording, local plan, and optional image guidance.

    ``similarity`` is a requested resemblance from 0 to 100, used for reference
    only generation. ``None`` means there is no numeric resemblance instruction,
    including when an eyewitness description accompanies a reference image.
    ``attributes_authoritative`` distinguishes explicit experimental controls
    from attributes inferred by a fallible local description parser.
    """

    original_description: str
    parsed_description: str
    attributes: dict[str, str]
    base_prompt: str
    portrait_instructions: str
    filler_profile: str = ""
    reference_image: Path | None = None
    similarity: float | None = None
    index: int = 1
    total: int = 1
    attributes_authoritative: bool = False


@dataclass(frozen=True)
class PromptProcessingResult:
    """A complete image prompt with concise, explicit processing metadata."""

    prompt: str
    summary: str
    constraints: tuple[str, ...]
    conflicts: tuple[str, ...]
    model: str
    provider: str = ""

    def __post_init__(self) -> None:
        for name in ("prompt", "summary", "model", "provider"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string.")
        if not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if not self.model.strip():
            raise ValueError("model must be a non-empty string.")
        for name in ("constraints", "conflicts"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
                raise TypeError(f"{name} must be a sequence of strings.")
            object.__setattr__(self, name, tuple(values))


@runtime_checkable
class PromptProcessor(Protocol):
    """Implement ``process`` to prepare a prompt before image generation."""

    def process(self, request: PromptProcessingRequest) -> PromptProcessingResult:
        ...
