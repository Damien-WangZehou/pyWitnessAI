"""Utilities for a CFD-based CLIP retrieval pilot study."""

from .cfd import build_cfd_manifest, load_cfd_metadata
from .description import build_proxy_descriptions, render_cfd_description
from .evaluate import evaluate_retrieval, summarise_retrieval
from .index import ClipIndex
from .lineup import build_filler_sets
from .retrieval_probe import (
    ClipRetrievalProbe,
    DescriptionLadder,
    FeaturePromptStep,
    ensure_clip_index,
    filter_expression_images,
    find_manifest_row,
)
from .visualization import search_and_show_lineup, show_clip_lineup

__all__ = [
    "ClipIndex",
    "ClipRetrievalProbe",
    "DescriptionLadder",
    "FeaturePromptStep",
    "build_cfd_manifest",
    "build_filler_sets",
    "build_proxy_descriptions",
    "ensure_clip_index",
    "evaluate_retrieval",
    "filter_expression_images",
    "find_manifest_row",
    "load_cfd_metadata",
    "render_cfd_description",
    "search_and_show_lineup",
    "show_clip_lineup",
    "summarise_retrieval",
]
