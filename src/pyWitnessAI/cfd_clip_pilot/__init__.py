"""Utilities for a CFD-based CLIP retrieval pilot study."""

from .cfd import build_cfd_manifest, load_cfd_metadata
from .description import build_proxy_descriptions, render_cfd_description
from .evaluate import evaluate_retrieval, summarise_retrieval
from .index import ClipIndex
from .lineup import build_filler_sets

__all__ = [
    "ClipIndex",
    "build_cfd_manifest",
    "build_filler_sets",
    "build_proxy_descriptions",
    "evaluate_retrieval",
    "load_cfd_metadata",
    "render_cfd_description",
    "summarise_retrieval",
]
