from __future__ import annotations

from importlib import import_module
import os


_OPTIONAL_DEPENDENCIES = {
    "cv",
    "cv2",
    "deepface",
    "facenet_pytorch",
    "keras",
    "mtcnn",
    "retinaface",
    "tensorflow",
    "torch",
}

_EXPORT_MODULES = (
    "utils.Constants",
    "Images",
    "ImagesAI",
    "Video",
    "VideoAI",
    "VideoProcessor",
    "Lineup",
    "LineupDecider",
    "VideoLineupPipeline",
    "FaceAttributeSchema",
    "ImageCatalog",
    "FaceSearch",
    "FillerGenerator",
    "GeneratedFaceDataset",
    "FillerSelectionBenchmark",
    "AttributeRetrievalBenchmark",
)
_OPTIONAL_EXPORT_MODULES = set(_EXPORT_MODULES) - {"utils.Constants"}
_EAGER_EXPORT_MODULES = (
    "utils.Constants",
    "FaceAttributeSchema",
    "ImageCatalog",
    "FaceSearch",
    "FillerGenerator",
    "GeneratedFaceDataset",
    "FillerSelectionBenchmark",
    "AttributeRetrievalBenchmark",
)
_LAZY_EXPORT_MODULES = {
    module_path.rsplit(".", 1)[-1]: module_path
    for module_path in _EXPORT_MODULES
    if module_path not in _EAGER_EXPORT_MODULES
}
_STAR_EXPORTS = {
    "Images",
    "ImagesAI",
    "Video",
    "VideoAI",
    "VideoProcessor",
    "Lineup",
    "LineupDecider",
    "VideoLineupPipeline",
    "FaceDescriptionSchema",
    "FaceAttributeDefinition",
    "FaceAttributeSchema",
    "DEFAULT_FACE_ATTRIBUTE_SCHEMA",
    "CatalogReport",
    "ImageCatalog",
    "MetadataJoinSpec",
    "discover_images",
    "SelectorQuery",
    "SelectorBackend",
    "CallableSelectorBackend",
    "ClipSelectorBackend",
    "MatchPolicy",
    "AllRankedPolicy",
    "TopKPolicy",
    "MinScorePolicy",
    "TopKAndMinScorePolicy",
    "GroundTruthSpec",
    "SelectionResult",
    "FaceSearch",
    "available_selector_backends",
    "create_selector_backend",
    "register_selector_backend",
    "FillerGenerator",
    "FillerSelectionBenchmark",
    "FillerSelectorBenchmark",
    "GeneratedFiller",
    "GeneratedFaceDataset",
    "DEFAULT_GENERATED_FACE_DATASET_ROOT",
    "DatasetMatchMode",
    "SCHEMA_COLUMNS",
    "ImageGenerationBackend",
    "ImageGenerationRequest",
    "OpenAIImageBackend",
    "available_image_generation_models",
    "available_image_generation_providers",
    "register_image_generation_backend",
    "BenchmarkStage",
    "AttributeProbeSpec",
    "AttributeRetrievalBenchmark",
    "AttributeSelector",
    "ScoreMode",
}

__all__: list[str] = []
_missing_optional_imports: dict[str, str] = {}


def _export_public_names(module_path: str) -> None:
    try:
        module = import_module(f"{__name__}.{module_path}")
    except ModuleNotFoundError as exc:
        if exc.name in _OPTIONAL_DEPENDENCIES:
            _missing_optional_imports[module_path] = exc.name or "unknown"
            return
        raise
    except Exception as exc:
        if module_path in _OPTIONAL_EXPORT_MODULES:
            _missing_optional_imports[module_path] = f"{type(exc).__name__}: {exc}"
            return
        raise

    public_names = getattr(
        module,
        "__all__",
        [name for name in dir(module) if not name.startswith("_")],
    )
    for name in public_names:
        globals()[name] = getattr(module, name)
        if name in _STAR_EXPORTS and name not in __all__:
            __all__.append(name)


_modules_to_export = (
    _EXPORT_MODULES
    if os.environ.get("PYWITNESSAI_EAGER_OPTIONAL_EXPORTS") == "1"
    else _EAGER_EXPORT_MODULES
)

for _module_path in _modules_to_export:
    _export_public_names(_module_path)


def __getattr__(name: str):
    """Load legacy, dependency-heavy exports only when they are requested."""
    module_path = _LAZY_EXPORT_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    _export_public_names(module_path)
    if name in globals():
        return globals()[name]

    reason = _missing_optional_imports.get(module_path, "unknown import error")
    raise AttributeError(
        f"{name!r} is unavailable because {module_path!r} could not be imported: {reason}"
    )


del _module_path
del _modules_to_export
