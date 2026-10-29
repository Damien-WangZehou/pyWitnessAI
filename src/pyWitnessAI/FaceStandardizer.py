"""Standardize portrait pixels before cataloguing, retrieval, or presentation.

One batch entry point, one single-image operation, and local model adapters.
Geometry uses a similarity transform: individual face proportions are retained.
Source images are never overwritten; uncertain outputs stay out of catalog.ready.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageCms, ImageOps, __version__ as pillow_version

from .ImageCatalog import ImageCatalog
from .API.models import download_model
from .model_files import file_sha256

__all__ = ["FaceStandardizer", "StandardizedFace"]

_VERSION = "portrait-v5"
_FALLBACK_BACKGROUND = (224, 224, 224)
_YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
_YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
_U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
_U2NET_SHA256 = "01eb6a29a5c4d8edb30b56adad9bb3a2a0535338e480724a213e0acfd2d1c73c"


@dataclass(frozen=True)
class StandardizedFace:
    """Portrait, coverage/foreground mask, QC data, and unenhanced counterpart."""

    image: Image.Image
    mask: Image.Image
    metadata: dict[str, object]
    baseline: Image.Image | None = None


class FaceStandardizer:
    """Center-crop portraits; optionally align eyes, remove backgrounds or cap detail.

    ``run(folder_or_manifest, output_dir)`` returns an ImageCatalog and saves a
    manifest, configuration, PNG portraits, and masks. Use a new/empty output
    directory for each run. ``transform(image)`` handles one image in memory.

    Optional callables operate on EXIF-oriented, sRGB PIL images: detector returns
    five (x, y) points (eyes, nose, mouth corners); segmenter returns a same-size
    L mask (0 background, 255 foreground). Backgrounds are retained by default;
    ``remove_background=True`` enables segmentation with local U2Net or the
    supplied segmenter. An explicitly supplied mask also enables masking.
    Basic center crops use no models. Supplying ``eye_distance`` in center mode
    sets the eye-based scale about the image center without rotating the image.
    ``alignment="eyes"`` enables eye alignment
    using local YuNet. ``background="auto"`` estimates a fill color per image.
    An optional enhancer(image, *, landmarks) returns same-size RGB pixels;
    both routes share original measurements and all enhanced outputs need review.
    """

    def __init__(
        self,
        *,
        size: tuple[int, int] = (256, 320),
        alignment: str = "center",
        eye_distance: float | None = None,
        eye_height: float = 0.47,
        background: tuple[int, int, int] | str = "auto",
        remove_background: bool = False,
        detail_eye_distance: float | None = None,
        detector: Callable | None = None,
        segmenter: Callable | None = None,
        enhancer: Callable | None = None,
        model_dir: str | Path | None = None,
    ) -> None:
        if len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 32 for v in size):
            raise ValueError("size must contain two integer pixel dimensions >= 32.")
        if alignment not in ("center", "eyes"):
            raise ValueError('alignment must be "center" or "eyes".')
        if eye_distance is not None and not 0 < eye_distance < 1:
            raise ValueError("eye_distance must be None or a fraction between 0 and 1.")
        if not 0 < eye_height < 1:
            raise ValueError("eye_height must be a fraction between 0 and 1.")
        if isinstance(background, str):
            if background != "auto":
                raise ValueError('background must be "auto" or three integer RGB values in [0, 255].')
        elif (not isinstance(background, (tuple, list)) or len(background) != 3
              or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in background)):
            raise ValueError('background must be "auto" or three integer RGB values in [0, 255].')
        if not isinstance(remove_background, bool):
            raise TypeError("remove_background must be a bool.")
        if detail_eye_distance is not None and (not np.isfinite(detail_eye_distance) or detail_eye_distance < 8):
            raise ValueError("detail_eye_distance must be None or a finite pixel distance >= 8.")
        if any(value is not None and not callable(value) for value in (detector, segmenter, enhancer)):
            raise TypeError("detector, segmenter, and enhancer must be callables or None.")
        self.size, self.eye_height = tuple(size), float(eye_height)
        self.eye_distance = (float(eye_distance) if eye_distance is not None else
                             0.25 if alignment == "eyes" else None)
        self.alignment = alignment
        self.background = background if isinstance(background, str) else tuple(background)
        self.remove_background = remove_background
        self.detail_eye_distance = float(detail_eye_distance) if detail_eye_distance is not None else None
        self.detector, self.segmenter = detector, segmenter
        self.enhancer = enhancer
        self.model_dir = Path(model_dir or Path.home() / ".cache" / "pyWitnessAI" / "portraits")
        self._face_detector = self._segmentation_session = None

    def transform(
        self, image: Image.Image | str | Path, *, landmarks=None, mask: Image.Image | None = None,
    ) -> StandardizedFace:
        """Render one portrait; optional measured landmarks/mask bypass models.

        A supplied mask explicitly enables masking, even when automatic background
        removal is disabled. Otherwise the returned mask records source coverage
        and alpha after alignment, rather than a foreground segmentation.
        Coordinates/mask must refer to the image AFTER EXIF orientation. QC flags
        are review requests, not proof of identity preservation or lineup fairness.
        Spatial detail is a sampling ceiling; blur and compression can still differ.
        """
        rgb, alpha = _read_portrait(image)
        automatic_background = self.background == "auto"
        background = _estimate_background(rgb, alpha) if automatic_background else self.background
        flattened = Image.composite(rgb, Image.new("RGB", rgb.size, background), alpha)
        points = eyes = delta = native_distance = None
        if landmarks is not None or self._needs_landmarks():
            measurements = landmarks if landmarks is not None else self._detect(flattened)
            try:
                points = np.asarray(measurements, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("Cannot obtain five numeric facial landmarks.") from exc
            if points.shape != (5, 2) or not np.isfinite(points).all():
                raise ValueError("Expected five finite (x, y) landmarks: eyes, nose, mouth corners.")
            if (points < 0).any() or (points >= np.array(rgb.size)).any():
                raise ValueError("Landmarks must lie inside the oriented image.")
            eyes = points[:2][np.argsort(points[:2, 0])]
            delta = eyes[1] - eyes[0]
            native_distance = float(np.linalg.norm(delta))
            if native_distance < 4:
                raise ValueError("Eye landmarks are coincident or too close to measure.")
        background_removed = mask is not None or self.remove_background
        if mask is not None:
            foreground, mask_source = mask, "supplied"
        elif self.remove_background:
            foreground = self._segment(flattened)
            mask_source = _callable_name(self.segmenter) if self.segmenter else "u2net_human_seg"
        else:
            foreground, mask_source = alpha, "source_alpha"
        if not isinstance(foreground, Image.Image) or foreground.mode != "L" or foreground.size != rgb.size:
            raise ValueError("Foreground mask must be a same-size PIL L image (0 background, 255 foreground).")
        foreground = Image.fromarray(np.minimum(np.asarray(foreground), np.asarray(alpha)))
        pixels = np.asarray(foreground)
        if background_removed:
            if np.count_nonzero(pixels > 127) < 0.01 * pixels.size:
                raise ValueError("Foreground segmentation is empty or implausibly small.")
            if points is not None and any(pixels[int(y), int(x)] < 128 for x, y in points):
                raise ValueError("Foreground mask excludes core facial landmarks; review the segmentation.")

        if self.alignment == "eyes":
            matrix = _eye_transform(eyes, self.eye_distance * self.size[0],
                                    (self.size[0] / 2, self.eye_height * self.size[1]))
        else:
            requested_scale = self.eye_distance * self.size[0] / native_distance if self.eye_distance is not None else None
            matrix = _center_transform(rgb.size, self.size, scale=requested_scale)
        scale = float(np.linalg.norm(matrix[0, :2]))
        aligned_points = points @ matrix[:, :2].T + matrix[:, 2] if points is not None else None
        target_distance = native_distance * scale if native_distance is not None else None
        sampling_distance = min(target_distance, self.detail_eye_distance or target_distance) if target_distance is not None else None
        detail_ratio = sampling_distance / target_distance if target_distance is not None else 1.0
        aligned_mask = _limit_detail(_warp(foreground, matrix, self.size, 0), detail_ratio)
        flags = _quality_flags(
            pixels, points, aligned_points, np.asarray(aligned_mask), native_distance,
            sampling_distance, matrix, check_foreground=background_removed,
        )
        portrait = self._render(rgb, foreground, matrix, aligned_mask, detail_ratio, background)
        baseline = None
        if self.enhancer is not None:
            # Freeze original measurements for both routes; custom models receive copies.
            enhanced = self.enhancer(flattened.copy(), landmarks=points.copy())
            if not isinstance(enhanced, Image.Image) or enhanced.mode != "RGB" or enhanced.size != rgb.size:
                raise ValueError("Enhancer must return a same-size PIL RGB image.")
            # The model sees visible pixels, while alpha is still applied only once.
            change = np.asarray(enhanced, dtype=np.int16) - np.asarray(flattened, dtype=np.int16)
            enhanced = Image.fromarray(np.clip(np.asarray(rgb, dtype=np.int16) + change, 0, 255).astype(np.uint8))
            baseline = portrait
            portrait = self._render(enhanced, foreground, matrix, aligned_mask, detail_ratio, background)
            flags.append("enhancement_requires_review")
        metadata = {
            "input_width": rgb.width, "input_height": rgb.height,
            "alignment": self.alignment,
            "background_color": json.dumps(list(background)),
            "background_source": "auto" if automatic_background else "supplied",
            "native_eye_distance": native_distance,
            "output_eye_distance": target_distance,
            "detail_eye_distance": sampling_distance,
            "scale": scale,
            "roll_degrees": float(np.degrees(np.arctan2(delta[1], delta[0]))) if delta is not None else None,
            "input_landmarks": json.dumps(points.tolist()) if points is not None else None,
            "landmark_source": ("supplied" if landmarks is not None else
                                (_callable_name(self.detector) if self.detector else "YuNet-2023mar")
                                if points is not None else None),
            "mask_source": mask_source,
            "background_removed": background_removed,
            "transform_matrix": json.dumps(matrix.tolist()),
            "enhancement_applied": self.enhancer is not None,
            "enhancer": _callable_name(self.enhancer) if self.enhancer is not None else None,
            "quality_flags": ";".join(flags), "review_required": bool(flags),
        }
        return StandardizedFace(portrait, aligned_mask, metadata, baseline)

    def _needs_landmarks(self) -> bool:
        return (self.alignment == "eyes" or self.eye_distance is not None
                or self.enhancer is not None or self.detail_eye_distance is not None)

    def _render(self, rgb, foreground, matrix, aligned_mask, detail_ratio, background):
        composite = Image.composite(rgb, Image.new("RGB", rgb.size, background), foreground)
        output = _limit_detail(_warp(composite, matrix, self.size, background), detail_ratio)
        # Exact background away from the soft edge, including after resampling.
        rendered = np.asarray(output).copy()
        rendered[np.asarray(aligned_mask) == 0] = background
        portrait = Image.fromarray(rendered)
        portrait.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        return portrait

    def run(
        self,
        images: ImageCatalog | pd.DataFrame | str | Path | Iterable[str | Path],
        output_dir: str | Path,
    ) -> ImageCatalog:
        """Save a batch to a new/empty directory; retain metadata and failed rows.

        Manifest columns ``landmarks`` (5x2 list/JSON) and ``mask_path`` optionally
        supply reviewed measurements. ``status=review`` images are saved for
        inspection but excluded from catalog.ready. Unknown provenance stays
        unknown: callers supply source_dataset/is_synthetic when known.
        """
        catalog = _catalog(images)
        if "standardization_version" in catalog.manifest and not catalog.manifest["standardization_version"].map(_missing).all():
            raise ValueError("Build each standardization version from original inputs, not an already standardized manifest.")
        destination = Path(output_dir).resolve()
        if isinstance(images, (str, Path)) and Path(images).is_dir() and Path(images).resolve() in destination.parents:
            raise ValueError("Keep output_dir outside the input folder to prevent rediscovering processed images.")
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError("Use a new or empty output_dir; existing outputs are never overwritten.")
        inputs = [Path(p).resolve() for p in catalog.manifest["image_path"]]
        if any(path == destination or destination in path.parents for path in inputs):
            raise ValueError("output_dir must not contain input images.")
        # Global dependency/model errors surface before creating a partial batch.
        ready = catalog.ready
        if not ready.empty:
            if self._needs_landmarks() and ("landmarks" not in ready or ready["landmarks"].map(_missing).any()):
                self._load_detector()
            if self.remove_background and ("mask_path" not in ready or ready["mask_path"].map(_missing).any()):
                self._load_segmenter()
            if callable(getattr(self.enhancer, "prepare", None)):
                self.enhancer.prepare()
        recipe = self._recipe()
        recipe_json = json.dumps(recipe, indent=2, sort_keys=True)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "images").mkdir()
        (destination / "masks").mkdir()
        if self.enhancer is not None:
            (destination / "baseline").mkdir()
        (destination / "standardization.json").write_text(recipe_json + "\n", encoding="utf-8")
        recipe_id = hashlib.sha256(recipe_json.encode()).hexdigest()[:16]
        records = []
        for source in catalog.manifest.to_dict("records"):
            path = Path(source["image_path"])
            token = hashlib.sha256(str(source["image_id"]).encode()).hexdigest()[:24]
            image_path, mask_path = destination / "images" / f"{token}.png", destination / "masks" / f"{token}.png"
            baseline_path = destination / "baseline" / f"{token}.png"
            row = {key: value for key, value in source.items() if key not in {"landmarks", "mask_path"}}
            if not _missing(source.get("mask_path")):
                row["standardization_input_mask"] = source["mask_path"]
            row.update(
                standardization_input_path=str(path), standardization_version=_VERSION,
                standardization_recipe=recipe_id,
                image_path=str(image_path), relative_image_path=image_path.relative_to(destination).as_posix(),
                image_name=image_path.name, file_extension=".png", standardized_mask_path=str(mask_path),
                quality_flags="", review_required=True,
            )
            if self.enhancer is not None:
                row["baseline_image_path"] = str(baseline_path)
            if str(source["status"]) == "ready":
                try:
                    landmarks = source.get("landmarks")
                    if _missing(landmarks):
                        landmarks = None
                    elif isinstance(landmarks, str):
                        landmarks = json.loads(landmarks)
                    mask = None
                    if not _missing(source.get("mask_path")):
                        supplied_mask = Path(source["mask_path"])
                        if not supplied_mask.is_absolute():
                            supplied_mask = (catalog.root or Path.cwd()) / supplied_mask
                        row["standardization_input_mask"] = str(supplied_mask.resolve())
                        row["standardization_input_mask_sha256"] = file_sha256(supplied_mask)
                        with Image.open(supplied_mask) as opened:
                            mask = opened.copy()
                    row["standardization_input_sha256"] = file_sha256(path)
                    result = self.transform(path, landmarks=landmarks, mask=mask)
                    result.image.save(image_path)
                    result.mask.save(mask_path)
                    if result.baseline is not None:
                        result.baseline.save(baseline_path)
                    row.update(result.metadata)
                    row.update(status="review" if result.metadata["review_required"] else "ready", error="")
                except (OSError, ValueError, RuntimeError) as exc:
                    image_path.unlink(missing_ok=True)
                    mask_path.unlink(missing_ok=True)
                    baseline_path.unlink(missing_ok=True)
                    row.update(status="error", error=str(exc), review_required=True)
            records.append(row)
        frame = pd.DataFrame(records) if records else catalog.manifest.copy()
        frame.to_csv(destination / "manifest.csv", index=False)
        return ImageCatalog.from_manifest(frame, base_dir=destination)

    def _load_detector(self) -> None:
        if self.detector is not None or self._face_detector is not None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise ImportError('Install local portrait models with pip install "pyWitnessAI[standardization]".') from exc
        model = download_model(self.model_dir / "face_detection_yunet_2023mar.onnx", _YUNET_URL, _YUNET_SHA256)
        self._face_detector = cv2.FaceDetectorYN.create(str(model), "", (320, 320), 0.8)

    def _detect(self, image: Image.Image) -> np.ndarray:
        if self.detector is not None:
            return self.detector(image)
        self._load_detector()
        import cv2
        ratio = min(1.0, 640 / max(image.size))
        small = image.resize(tuple(round(v * ratio) for v in image.size), Image.Resampling.LANCZOS)
        self._face_detector.setInputSize(small.size)
        _, faces = self._face_detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
        if faces is None or len(faces) != 1:
            raise ValueError("Expected exactly one detected face; review the input or supply measured landmarks.")
        ratios = np.array(small.size) / np.array(image.size)
        return (faces[0, 4:14].reshape(5, 2) + 0.5) / ratios - 0.5

    def _load_segmenter(self) -> None:
        if self.segmenter is not None or self._segmentation_session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError('Install local portrait models with pip install "pyWitnessAI[standardization]".') from exc
        model = download_model(self.model_dir / "u2net_human_seg.onnx", _U2NET_URL, _U2NET_SHA256)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self._segmentation_session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])

    def _segment(self, image: Image.Image) -> Image.Image:
        if self.segmenter is not None:
            return self.segmenter(image)
        self._load_segmenter()
        # U2Net's published input/output convention, cross-checked against
        # rembg 2.0.61 U2netHumanSegSession; no matting or generated pixels.
        pixels = np.asarray(image.resize((320, 320), Image.Resampling.LANCZOS), dtype=float)
        pixels /= max(float(pixels.max()), 1.0)
        pixels = (pixels - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = pixels.transpose(2, 0, 1)[None].astype(np.float32)
        name = self._segmentation_session.get_inputs()[0].name
        prediction = self._segmentation_session.run(None, {name: tensor})[0]
        if prediction.shape != (1, 1, 320, 320) or not np.isfinite(prediction).all():
            raise ValueError("Invalid U2Net segmentation output.")
        prediction = prediction[0, 0]
        low, high = float(prediction.min()), float(prediction.max())
        if high - low < 1e-6:
            raise ValueError("U2Net could not separate a foreground from the background.")
        mask = Image.fromarray(((prediction - low) / (high - low) * 255).astype(np.uint8))
        return mask.resize(image.size, Image.Resampling.LANCZOS)

    def _recipe(self) -> dict[str, object]:
        recipe = dict(
            version=_VERSION, size=self.size, alignment=self.alignment,
            eye_distance=self.eye_distance, eye_height=self.eye_height,
            background=self.background, remove_background=self.remove_background,
            detail_eye_distance=self.detail_eye_distance,
            color="sRGB; untagged RGB assumed sRGB; no exposure or skin-tone matching",
            detector=((_callable_name(self.detector) if self.detector else "YuNet-2023mar")
                      if self._needs_landmarks() else None),
            segmenter=((_callable_name(self.segmenter) if self.segmenter else "u2net_human_seg")
                       if self.remove_background else None),
            python=platform.python_version(), pillow=pillow_version, numpy=np.__version__,
            enhancement=(self.enhancer.recipe() if callable(getattr(self.enhancer, "recipe", None))
                         else {"callable": _callable_name(self.enhancer)} if self.enhancer is not None else None),
        )
        if self._face_detector is not None:
            import cv2
            recipe["opencv"] = cv2.__version__
            recipe["detector_sha256"] = _YUNET_SHA256
        if self._segmentation_session is not None:
            import onnxruntime as ort
            recipe["onnxruntime"], recipe["segmenter_sha256"] = ort.__version__, _U2NET_SHA256
        return recipe


def _catalog(images) -> ImageCatalog:
    if isinstance(images, ImageCatalog):
        return images
    if isinstance(images, pd.DataFrame):
        return ImageCatalog.from_manifest(images)
    if isinstance(images, (str, Path)):
        path = Path(images)
        if path.is_dir():
            return ImageCatalog.from_folder(path)
        images = [path]
    return ImageCatalog.from_manifest(pd.DataFrame({"image_path": [str(Path(p).resolve()) for p in images]}))


def _read_portrait(image) -> tuple[Image.Image, Image.Image]:
    if not isinstance(image, Image.Image):
        with Image.open(image) as opened:
            return _read_portrait(opened)
    oriented = ImageOps.exif_transpose(image)
    alpha = oriented.convert("RGBA").getchannel("A")
    profile = oriented.info.get("icc_profile")
    if profile:
        try:
            source = ImageCms.ImageCmsProfile(BytesIO(profile))
            rgb = ImageCms.profileToProfile(oriented, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except (ImageCms.PyCMSError, OSError) as exc:
            raise ValueError("Cannot convert the embedded color profile to sRGB.") from exc
    else:
        rgb = oriented.convert("RGB")
    return rgb, alpha


def _estimate_background(rgb, alpha):
    """Use the dominant visible border color, ignoring hidden RGB and outliers."""
    band = max(1, min(8, round(min(rgb.size) * 0.02)))
    pixels, opacity = np.asarray(rgb), np.asarray(alpha)
    border = np.concatenate((pixels[:band].reshape(-1, 3), pixels[-band:].reshape(-1, 3),
                             pixels[band:-band, :band].reshape(-1, 3),
                             pixels[band:-band, -band:].reshape(-1, 3)))
    weights = np.concatenate((opacity[:band].ravel(), opacity[-band:].ravel(),
                              opacity[band:-band, :band].ravel(),
                              opacity[band:-band, -band:].ravel()))
    visible = weights > 0
    if not visible.any():
        return _FALLBACK_BACKGROUND
    border, weights = border[visible], weights[visible].astype(float)
    quantized = border.astype(np.int32) // 16
    bins = quantized @ np.array([256, 16, 1])
    dominant = np.argmax(np.bincount(bins, weights=weights, minlength=4096))
    selected = bins == dominant
    border, weights = border[selected], weights[selected]
    color = []
    for channel in border.T:
        order = np.argsort(channel)
        middle = np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)
        color.append(int(channel[order[middle]]))
    return tuple(color)


def _center_transform(source_size, output_size, *, scale=None):
    """Uniform resize about the image center; cover the output if scale is unset."""
    if scale is None:
        scale = max(output_size[0] / source_size[0], output_size[1] / source_size[1])
    translation = ((np.asarray(output_size) - 1) - scale * (np.asarray(source_size) - 1)) / 2
    return np.column_stack((np.eye(2) * scale, translation))


def _eye_transform(eyes, target_distance, target_midpoint):
    """Similarity transform from two ordered eye centers, without face warping."""
    delta = eyes[1] - eyes[0]
    a, b = target_distance * delta / np.dot(delta, delta)
    linear = np.array([[a, b], [-b, a]])
    return np.column_stack((linear, np.asarray(target_midpoint) - linear @ eyes.mean(axis=0)))


def _limit_detail(image, ratio):
    if ratio >= 1:
        return image
    small = tuple(max(1, round(v * ratio)) for v in image.size)
    return image.resize(small, Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.BICUBIC)


def _warp(image, matrix, size, fill):
    """Prefilter large reductions, then apply the same invertible geometry."""
    scale = float(np.linalg.norm(matrix[0, :2]))
    adjusted = matrix.copy()
    # Landmarks use pixel centers; PIL affine transforms use pixel edges.
    adjusted[:, 2] += 0.5 - adjusted[:, :2] @ np.array([0.5, 0.5])
    if scale < 1:
        smaller = tuple(max(1, round(v * scale)) for v in image.size)
        ratios = np.array(smaller) / np.array(image.size)
        image = image.resize(smaller, Image.Resampling.LANCZOS)
        adjusted[:, :2] /= ratios
    inverse = np.linalg.inv(np.vstack((adjusted, [0, 0, 1])))[:2]
    return image.transform(size, Image.Transform.AFFINE, inverse.ravel(), Image.Resampling.BICUBIC, fillcolor=fill)


def _quality_flags(mask, points, aligned, aligned_mask, native_distance, sampling_distance, matrix, *, check_foreground=True):
    flags = []
    # A basic center crop has no face measurements from which to infer face QC.
    if points is None:
        return flags
    if native_distance < sampling_distance - 0.5:
        flags.append("insufficient_native_resolution")
    if check_foreground:
        if np.any(mask[:2] > 127):
            flags.append("source_head_touches_top")
        # Exclude shoulders when checking lateral head truncation.
        head_bottom = min(mask.shape[0], max(1, int(points[2, 1] + native_distance * 0.5)))
        if np.any(mask[:head_bottom, :2] > 127) or np.any(mask[:head_bottom, -2:] > 127):
            flags.append("source_head_touches_side")
        out_head_bottom = min(aligned_mask.shape[0], max(1, int(aligned[2, 1] + np.linalg.norm(aligned[0] - aligned[1]) * 0.5)))
        if np.any(aligned_mask[:2] > 127) or np.any(aligned_mask[:out_head_bottom, :2] > 127) or np.any(aligned_mask[:out_head_bottom, -2:] > 127):
            flags.append("output_head_touches_edge")
    if (aligned < 2).any() or (aligned >= np.array(aligned_mask.shape[::-1]) - 2).any():
        flags.append("landmark_outside_output")
    if check_foreground:
        bottom_x = np.flatnonzero(mask[-1] > 127)
        boundary = np.column_stack((bottom_x, np.full(len(bottom_x), mask.shape[0] - 1)))
        boundary = boundary @ matrix[:, :2].T + matrix[:, 2]
        if np.any(np.all((boundary > 2) & (boundary < np.array(aligned_mask.shape[::-1]) - 2), axis=1)):
            flags.append("source_bottom_in_output")
    eyes = points[:2][np.argsort(points[:2, 0])]
    # A conservative asymmetry cue, not a calibrated yaw estimate.
    if abs(np.dot(points[2] - eyes.mean(axis=0), (eyes[1] - eyes[0]) / native_distance)) > native_distance * 0.25:
        flags.append("possible_nonfrontal_pose")
    return flags


def _missing(value) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip()
    return bool(pd.isna(value)) if np.isscalar(value) else False


def _callable_name(function) -> str:
    return f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', type(function).__name__)}"
