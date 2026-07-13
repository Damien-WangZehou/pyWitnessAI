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
    "FillerGenerator",
    "GeneratedFaceDataset",
    "FillerSelectionBenchmark",
)
_OPTIONAL_EXPORT_MODULES = set(_EXPORT_MODULES) - {"utils.Constants"}
_EAGER_EXPORT_MODULES = (
    "utils.Constants",
    "FillerGenerator",
    "GeneratedFaceDataset",
    "FillerSelectionBenchmark",
)
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


del _module_path
del _modules_to_export
