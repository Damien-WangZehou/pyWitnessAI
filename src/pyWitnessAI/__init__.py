from __future__ import annotations

from importlib import import_module


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
)
_OPTIONAL_EXPORT_MODULES = set(_EXPORT_MODULES) - {"utils.Constants"}
_STAR_EXPORTS = {
    "Images",
    "ImagesAI",
    "Video",
    "VideoAI",
    "VideoProcessor",
    "Lineup",
    "LineupDecider",
    "VideoLineupPipeline",
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


for _module_path in _EXPORT_MODULES:
    _export_public_names(_module_path)


del _module_path
