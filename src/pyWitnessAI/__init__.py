"""Stable public imports without loading optional model or display libraries.

Named exports are resolved from their owning module on first access. The
star-import surface includes only APIs available with the base dependencies;
vision APIs remain available through explicit imports. To access a module whose
name is also a public class, use ``importlib.import_module("pyWitnessAI.FaceSearch")``.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version
import sys as _sys
from types import ModuleType as _ModuleType


try:
    from ._version import __version__
except ModuleNotFoundError as _version_error:
    if _version_error.name != f"{__name__}._version":
        raise
    try:
        __version__ = _distribution_version("pyWitnessAI")
    except _PackageNotFoundError:
        __version__ = "0.0.1"


# This registry is explicit: importing another module must never accidentally
# publish its dependencies or choose a different owner for a shared name.
_EXPORTS_BY_MODULE = {
    "utils.Constants": (
        "colors", "legend_colors", "line_styles", "get_random_color",
        "get_color_for_analyzer", "get_style_for_analyzer",
    ),
    "Images": ("Images",),
    "ImagesAI": ("ImagesAI",),
    "ImagesCategorizer": ("organize_photos",),
    "Video": ("Video",),
    "VideoAI": (
        "FrameAnalyzerMTCNNIndependent", "FrameAnalyzerOpenCVIndependent",
        "FrameAnalyzerDeepface", "FrameAnalyzerMTCNN", "FrameAnalyzerOpenCV",
        "FrameAnalyzerFastMTCNN", "SimilarityAnalyzer", "LineupLoader",
    ),
    "VideoProcessor": (
        "FrameProcessorCropper", "FrameProcessorMonochrome",
        "FrameProcessorNormalizer", "FrameProcessorRemover",
        "FrameProcessorVideoWriter", "FrameProcessorDisplayer",
        "FrameProcessorHistogramEqualization", "FrameProcessorNoiseReduction",
        "FrameProcessorGammaCorrection", "FrameProcessorSharpening",
    ),
    "LineupBuilder": ("LineupBuilder", "Lineup", "LineupMember"),
    "LineupSession": ("create_lineup", "LineupSession", "LineupRound"),
    "LineupAIDecision": ("LineupAIDecision",),
    "LineupDecider": ("LineupDecisionResult", "LineupDecider", "VerifyStyleDecider"),
    "VideoLineupPipeline": (
        "PipelineConfig", "export_for_pywitness", "VideoLineupPipeline",
    ),
    "FaceAttributeSchema": (
        "FaceAttributeDefinition", "FaceAttributeSchema", "FaceDescriptionSchema",
        "DEFAULT_FACE_ATTRIBUTE_SCHEMA", "LEGACY_FACE_ATTRIBUTE_SCHEMA_V1",
    ),
    "ImageCatalog": (
        "CatalogReport", "ImageCatalog", "MetadataJoinSpec", "MetadataSource",
        "IMAGE_EXTENSIONS", "discover_images", "image_paths", "image_manifest",
    ),
    "Description": (
        "DescriptionParser", "DescriptionAttribute", "ParsedDescription",
        "RawDescriptionParser", "RuleBasedDescriptionParser",
        "LenientDescriptionParser", "StrictDescriptionParser", "SchemaDescriptionParser",
        "create_description_parser", "normalize_description",
    ),
    "Scoring": (
        "ScoringStrategy", "ScoringResult", "CosineScoring", "ContrastScoring",
        "create_scoring_strategy",
    ),
    "FaceSearch": (
        "SelectorQuery", "SelectorBackend", "CallableSelectorBackend",
        "ClipSelectorBackend", "MatchPolicy", "AllRankedPolicy", "TopKPolicy",
        "MinScorePolicy", "TopKAndMinScorePolicy", "GroundTruthSpec",
        "SelectionResult", "FaceSearch", "available_selector_backends",
        "create_selector_backend", "register_selector_backend",
        "normalise_selector_scores",
    ),
    "FillerSelector": ("FillerSelector", "select_by_similarity"),
    "FaceGenerator": ("FaceGenerator", "GeneratedFace"),
    "FaceStandardizer": ("FaceStandardizer", "StandardizedFace"),
    "CodeFormerEnhancer": ("CodeFormerEnhancer",),
    "face_generation_backend": (
        "ImageGenerationBackend", "ImageGenerationRequest",
        "available_image_generation_models", "available_image_generation_providers",
        "create_image_generation_backend", "register_image_generation_backend",
    ),
    "API.openai": ("OpenAIImageBackend",),
    "prompt_processing": ("PromptProcessor", "PromptProcessingRequest", "PromptProcessingResult"),
    "API.openai_prompt_processor": ("OpenAIPromptProcessor",),
    "GeneratedFaceDataset": (
        "GeneratedFaceDataset", "DEFAULT_GENERATED_FACE_DATASET_ROOT",
        "DatasetMatchMode", "SCHEMA_COLUMNS",
    ),
    "FillerSelectionBenchmark": (
        "FillerSelectionBenchmark", "FillerSelectorBenchmark", "BenchmarkStage",
        "MissingContrastPolicy", "UnusedContrastPolicy",
    ),
    "AttributeRetrievalBenchmark": (
        "AttributeRetrievalBenchmark", "AttributeProbeSpec", "AttributeSelector",
        "ScoreMode",
    ),
    "EyewitnessContrastPrompts": (
        "Abstention", "AttributeEvidence", "ContrastPlan", "ContrastPromptPair",
        "ContrastPromptRule", "ContrastPromptSchema", "EyewitnessContrastPromptBuilder",
        "GENERIC_CONTRAST_SCHEMA", "AGE_PROMPT_TEMPLATE", "LENIENT_PARSER_VERSION",
        "PARSER_VERSION", "PROMPT_TEMPLATE", "PROMPT_TEMPLATE_VERSION",
    ),
    "ContrastiveClipScoring": (
        "ContrastMethod", "ContrastiveClipScorer", "ContrastiveScoreResult",
        "DEFAULT_METHODS", "DEFAULT_NEUTRAL_PROMPT", "SCORER_VERSION",
        "score_contrast_plans",
    ),
}
_EXPORTS = {
    name: (module_path, name)
    for module_path, names in _EXPORTS_BY_MODULE.items()
    for name in names
}
# These two historical names denote modules, not classes. CLI access is lazy.
_EXPORTS.update({
    "VideoAI": ("VideoAI", None),
    "VideoProcessor": ("VideoProcessor", None),
    "main": ("API.cli", "main"),
})
_VISION_MODULES = frozenset({
    "ImagesAI", "Video", "VideoAI", "VideoProcessor", "VideoLineupPipeline",
})
__all__ = sorted(
    name for name, (module_path, _) in _EXPORTS.items()
    if module_path not in _VISION_MODULES and name != "main"
)
_EXPORT_CACHE: dict[str, object] = {}

# Catch only a genuinely missing, known optional dependency. Internal import
# errors and installed-but-broken libraries retain their original traceback.
_OPTIONAL_EXTRAS = {
    "cv": "vision",
    "cv2": "vision",
    "deepface": "vision",
    "facenet_pytorch": "vision",
    "torchvision": "vision",
    "tensorflow": "vision",
    "tensorflow_intel": "vision",
    "keras": "vision",
    "tf_keras": "vision",
    "mtcnn": "vision",
    "retinaface": "vision",
    "sentence_transformers": "clip",
    "transformers": "clip",
    "tokenizers": "clip",
    "huggingface_hub": "clip",
    "matplotlib": "plot",
    "IPython": "notebook",
    "openai": "generation",
    "fastapi": "api",
    "starlette": "api",
    "pydantic": "api",
    "uvicorn": "api",
}


def _load_export(name: str):
    if name in _EXPORT_CACHE:
        return _EXPORT_CACHE[name]
    module_path, attribute = _EXPORTS[name]
    try:
        module = _import_module(f"{__name__}.{module_path}")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        extra = _OPTIONAL_EXTRAS.get(missing)
        if missing == "torch":
            extra = "vision" if module_path in _VISION_MODULES else "clip"
        if extra is None:
            raise
        raise ModuleNotFoundError(
            f"pyWitnessAI.{name} requires optional dependency {missing!r}. "
            f'Install it with: pip install "pyWitnessAI[{extra}]"',
            name=missing,
        ) from exc
    value = module if attribute is None else getattr(module, attribute)
    _EXPORT_CACHE[name] = value
    return value


def __getattr__(name: str):
    if name in _EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


class _PublicModule(_ModuleType):
    def __getattribute__(self, name: str):
        # importlib assigns submodules to their parent package. Intercept only
        # declared public names so FaceSearch remains its class after either
        # import order; module objects remain in sys.modules/importlib.
        if name in _EXPORTS:
            return _load_export(name)
        return super().__getattribute__(name)


_sys.modules[__name__].__class__ = _PublicModule
