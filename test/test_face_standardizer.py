"""Offline checks of portrait geometry, sampling, and batch provenance."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from pyWitnessAI.FaceStandardizer import FaceStandardizer, StandardizedFace
from pyWitnessAI.ImageCatalog import ImageCatalog


POINTS = np.array([[65, 85], [135, 85], [100, 115], [84, 145], [116, 145]], dtype=float)


def portrait(color=(120, 140, 160)):
    image = Image.new("RGB", (200, 240), color)
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).ellipse((40, 25, 160, 205), fill=255)
    return image, mask


def standardizer(**kwargs):
    kwargs.setdefault("alignment", "eyes")
    kwargs.setdefault("remove_background", True)
    return FaceStandardizer(
        size=(200, 240), eye_distance=0.30, eye_height=0.40,
        detector=lambda image: POINTS.copy(), segmenter=lambda image: portrait()[1],
        **kwargs,
    )


def save_portrait(path, color=(120, 140, 160)):
    path.parent.mkdir(parents=True, exist_ok=True)
    portrait(color)[0].save(path)
    return path


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Portrait unit tests must not download models or use network services.")

    module = importlib.import_module("pyWitnessAI.API.models")
    monkeypatch.setattr(module, "urlopen", fail)


def test_alignment_preserves_shape_and_levels_eyes_without_mutating_inputs():
    points = np.array([[60, 70], [120, 100], [80, 118], [65, 145], [97, 161]], dtype=float)
    image, mask = portrait()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    drawing = ImageDraw.Draw(image)
    for (x, y), color in zip(points, colors):
        drawing.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    original_image, original_mask, original_points = image.tobytes(), mask.tobytes(), points.copy()

    result = standardizer().transform(image, landmarks=points, mask=mask)

    assert isinstance(result, StandardizedFace)
    assert result.image.size == result.mask.size == (200, 240)
    assert result.image.mode == "RGB" and result.mask.mode == "L"
    pixels = np.asarray(result.image).astype(int)
    centers = []
    for color in colors:
        ys, xs = np.where(np.max(np.abs(pixels - color), axis=2) < 30)
        assert len(xs) > 5, "Each facial marker must survive the transform."
        centers.append((xs.mean(), ys.mean()))
    centers = np.array(centers)
    np.testing.assert_allclose(centers[:2, 1], [96, 96], atol=1)
    assert centers[1, 0] - centers[0, 0] == pytest.approx(60, abs=1)
    assert centers[:2, 0].mean() == pytest.approx(100, abs=1)
    source_eye_distance = np.linalg.norm(points[1] - points[0])
    for left, right in [(0, 2), (1, 4), (2, 3), (3, 4)]:
        source_ratio = np.linalg.norm(points[left] - points[right]) / source_eye_distance
        rendered_ratio = np.linalg.norm(centers[left] - centers[right]) / 60
        assert rendered_ratio == pytest.approx(source_ratio, abs=0.025)
    assert image.tobytes() == original_image
    assert mask.tobytes() == original_mask
    np.testing.assert_array_equal(points, original_points)


def test_default_only_crops_without_using_any_models(monkeypatch):
    image, _ = portrait()
    drawing = ImageDraw.Draw(image)
    for bounds, color in [((0, 0, 20, 20), (240, 30, 10)),
                          ((179, 0, 199, 20), (10, 220, 40)),
                          ((0, 219, 20, 239), (20, 50, 230)),
                          ((179, 219, 199, 239), (230, 200, 20))]:
        drawing.rectangle(bounds, fill=color)

    def fail(*args, **kwargs):
        pytest.fail("Default center cropping must not load or invoke any models.")

    processor = FaceStandardizer(size=image.size, segmenter=fail)
    for name in ("_load_detector", "_detect", "_load_segmenter", "_segment"):
        monkeypatch.setattr(processor, name, fail)

    result = processor.transform(image)

    assert result.image.tobytes() == image.tobytes()
    np.testing.assert_array_equal(np.asarray(result.mask), np.full((240, 200), 255))
    assert result.metadata["mask_source"] == "source_alpha"
    assert result.metadata["background_removed"] is False
    assert not result.metadata["review_required"]
    assert processor._recipe()["remove_background"] is False
    assert processor._recipe()["segmenter"] is None
    assert processor._recipe()["detector"] is None
    assert processor._recipe()["eye_distance"] is None
    assert processor._recipe()["alignment"] == "center"
    assert processor._recipe()["background"] == "auto"
    assert result.metadata["alignment"] == "center"
    for field in ("native_eye_distance", "output_eye_distance", "detail_eye_distance",
                  "roll_degrees", "input_landmarks", "landmark_source"):
        assert result.metadata[field] is None


@pytest.mark.parametrize("source_size,output_size", [
    ((321, 199), (128, 160)),
    ((199, 321), (192, 128)),
    ((97, 63), (256, 320)),
    ((63, 97), (256, 320)),
])
def test_center_crop_covers_odd_landscape_and_portrait_inputs(source_size, output_size):
    image = Image.new("RGB", source_size, (30, 80, 140))
    result = FaceStandardizer(size=output_size, background=(255, 0, 255)).transform(image)

    matrix = np.array(json.loads(result.metadata["transform_matrix"]))
    scale = max(output_size[0] / source_size[0], output_size[1] / source_size[1])
    np.testing.assert_allclose(matrix[:, :2], [[scale, 0], [0, scale]])
    source_center = np.r_[(np.array(source_size) - 1) / 2, 1]
    np.testing.assert_allclose(matrix @ source_center, (np.array(output_size) - 1) / 2)
    assert result.image.size == result.mask.size == output_size
    np.testing.assert_array_equal(np.asarray(result.mask), np.full(output_size[::-1], 255))
    np.testing.assert_array_equal(np.asarray(result.image), np.full((*output_size[::-1], 3), [30, 80, 140]))


def test_center_crop_ignores_off_center_tilted_landmarks_for_geometry():
    pixels = np.zeros((240, 200, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(200, dtype=np.uint8)
    pixels[:, :, 1] = np.arange(240, dtype=np.uint8)[:, None]
    image = Image.fromarray(pixels)
    tilted = np.array([[25, 75], [85, 105], [50, 123], [35, 150], [67, 166]], dtype=float)
    processor = FaceStandardizer(size=(120, 160), eye_height=0.80)

    unmeasured = processor.transform(image)
    measured = processor.transform(image, landmarks=tilted)

    assert measured.image.tobytes() == unmeasured.image.tobytes()
    assert measured.mask.tobytes() == unmeasured.mask.tobytes()
    assert measured.metadata["transform_matrix"] == unmeasured.metadata["transform_matrix"]
    assert measured.metadata["landmark_source"] == "supplied"
    assert measured.metadata["native_eye_distance"] == pytest.approx(np.linalg.norm(tilted[1] - tilted[0]))
    assert measured.metadata["output_eye_distance"] == pytest.approx(
        measured.metadata["native_eye_distance"] * (160 / 240))
    assert image.tobytes() == pixels.tobytes()


@pytest.mark.parametrize("eye_distance", [0.20, 0.40])
def test_center_eye_distance_scales_pixels_about_image_center_without_rotation(eye_distance):
    image, _ = portrait((30, 80, 140))
    tilted = np.array([[65, 70], [125, 100], [90, 118], [75, 145], [107, 161]], dtype=float)
    drawing = ImageDraw.Draw(image)
    for (x, y), color in zip(tilted[:2], [(255, 0, 0), (0, 255, 0)]):
        drawing.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    calls = []

    def detect(oriented):
        calls.append(oriented.size)
        return tilted.copy()

    result = FaceStandardizer(alignment="center", size=image.size, eye_distance=eye_distance,
                              detector=detect).transform(image)

    assert calls == [image.size]
    matrix = np.array(json.loads(result.metadata["transform_matrix"]))
    native_distance = np.linalg.norm(tilted[1] - tilted[0])
    scale = eye_distance * image.width / native_distance
    np.testing.assert_allclose(matrix[:, :2], [[scale, 0], [0, scale]])
    center = (np.array(image.size) - 1) / 2
    np.testing.assert_allclose(matrix @ np.r_[center, 1], center)
    transformed = tilted @ matrix[:, :2].T + matrix[:, 2]
    pixels = np.asarray(result.image)
    observed = []
    for channel in (0, 1):
        marker = ((pixels[:, :, channel] > 200)
                  & (pixels[:, :, 1 - channel] < 70) & (pixels[:, :, 2] < 70))
        ys, xs = np.where(marker)
        assert len(xs) > 5
        observed.append([xs.mean(), ys.mean()])
    np.testing.assert_allclose(observed, transformed[:2], atol=0.75)
    assert np.linalg.norm(np.array(observed[1]) - observed[0]) == pytest.approx(eye_distance * image.width, abs=1)
    assert observed[1][1] - observed[0][1] > 10, "The original eye-line tilt must remain visible."
    assert result.metadata["output_eye_distance"] == pytest.approx(eye_distance * image.width)


def test_eye_alignment_keeps_default_eye_distance_when_omitted_or_none():
    image, _ = portrait()
    outputs = [FaceStandardizer(alignment="eyes", **options).transform(image, landmarks=POINTS)
               for options in ({}, {"eye_distance": None}, {"eye_distance": 0.25})]

    assert all(output.metadata["output_eye_distance"] == pytest.approx(64) for output in outputs)
    assert all(output.image.tobytes() == outputs[0].image.tobytes() for output in outputs)
    assert FaceStandardizer(alignment="eyes")._recipe()["eye_distance"] == 0.25


@pytest.mark.parametrize("source_size", [(1024, 1024), (319, 257), (257, 319), (321, 199), (199, 321)])
@pytest.mark.parametrize("output_size", [(256, 320), (257, 319), (319, 257)])
def test_center_fill_canvas_removes_padding_without_stretching_or_rotation(source_size, output_size):
    image = Image.new("RGB", source_size, (30, 80, 140))
    points = np.array([[0.30, 0.40], [0.65, 0.45], [0.50, 0.55],
                       [0.40, 0.65], [0.60, 0.65]]) * source_size
    options = dict(size=output_size, eye_distance=0.20, background=(255, 0, 255))

    padded = FaceStandardizer(**options).transform(image, landmarks=points)
    filled = FaceStandardizer(fill_canvas=True, **options).transform(image, landmarks=points)
    basic = FaceStandardizer(size=output_size).transform(image)

    assert padded.mask.getextrema()[0] == 0
    assert filled.mask.getextrema() == (255, 255)
    assert filled.image.tobytes() == basic.image.tobytes()
    assert filled.metadata["transform_matrix"] == basic.metadata["transform_matrix"]
    assert filled.metadata["eye_distance_clamped"] is True
    assert filled.metadata["output_eye_distance"] > 0.20 * output_size[0]
    matrix = np.array(json.loads(filled.metadata["transform_matrix"]))
    np.testing.assert_allclose(matrix[0, 0], matrix[1, 1])
    np.testing.assert_allclose([matrix[0, 1], matrix[1, 0]], 0)


@pytest.mark.parametrize("native_distance", [120, 163.84])
def test_center_fill_canvas_keeps_eye_target_when_it_already_covers(native_distance):
    image = Image.new("RGB", (1024, 1024), (30, 80, 140))
    points = np.array([[512 - native_distance / 2, 400], [512 + native_distance / 2, 400],
                       [512, 500], [470, 600], [554, 600]])
    options = dict(eye_distance=0.20, background=(255, 0, 255))

    unbounded = FaceStandardizer(**options).transform(image, landmarks=points)
    filled = FaceStandardizer(fill_canvas=True, **options).transform(image, landmarks=points)

    assert filled.image.tobytes() == unbounded.image.tobytes()
    assert filled.mask.tobytes() == unbounded.mask.tobytes()
    assert filled.mask.getextrema() == (255, 255)
    assert filled.metadata["eye_distance_clamped"] is False
    assert filled.metadata["output_eye_distance"] == pytest.approx(51.2)


def test_center_fill_canvas_batch_records_option_and_actual_measurements(tmp_path):
    source = save_portrait(tmp_path / "source.png")
    processor = FaceStandardizer(size=(200, 240), eye_distance=0.20, fill_canvas=True,
                                 detector=lambda image: POINTS.copy())
    destination = tmp_path / "output"

    result = processor.run([source], destination)

    assert len(result.ready) == 1
    row = result.ready.iloc[0]
    assert row["fill_canvas"]
    assert row["eye_distance_clamped"]
    assert row["output_eye_distance"] == pytest.approx(70)
    with Image.open(row["standardized_mask_path"]) as mask:
        assert mask.getextrema() == (255, 255)
    recipe = json.loads((destination / "standardization.json").read_text(encoding="utf-8"))
    assert recipe["fill_canvas"] is True
    assert recipe["eye_distance"] == 0.20


def test_center_fill_canvas_without_eye_distance_still_needs_no_models(monkeypatch):
    image, _ = portrait()
    processor = FaceStandardizer(fill_canvas=True)

    def fail(*args, **kwargs):
        pytest.fail("Filling the canvas alone must not require a model.")

    monkeypatch.setattr(processor, "_detect", fail)
    result = processor.transform(image)

    assert result.image.tobytes() == FaceStandardizer().transform(image).image.tobytes()
    assert result.mask.getextrema() == (255, 255)
    assert result.metadata["eye_distance_clamped"] is False


@pytest.mark.parametrize("fill_canvas", [None, 0, 1, "true"])
def test_invalid_fill_canvas_is_rejected(fill_canvas):
    with pytest.raises(TypeError, match="fill_canvas"):
        FaceStandardizer(fill_canvas=fill_canvas)


def test_fill_canvas_rejects_eye_alignment():
    with pytest.raises(ValueError, match="fill_canvas.*center"):
        FaceStandardizer(alignment="eyes", fill_canvas=True)


def test_center_eye_distance_batch_prepares_detection_and_uses_each_source_background(tmp_path, monkeypatch):
    colors = [(180, 100, 60), (30, 80, 160)]
    sources = [save_portrait(tmp_path / f"source-{index}.png", color)
               for index, color in enumerate(colors)]
    destination = tmp_path / "output"
    calls = []

    def prepare():
        assert not destination.exists()
        calls.append("prepare")

    def detect(image):
        calls.append("detect")
        return POINTS.copy()

    processor = FaceStandardizer(alignment="center", size=(200, 240), eye_distance=0.20, detector=detect)
    monkeypatch.setattr(processor, "_load_detector", prepare)
    result = processor.run(sources, destination)

    assert calls == ["prepare", "detect", "detect"]
    assert len(result.ready) == 2
    for row, color in zip(result.manifest.to_dict("records"), colors):
        with Image.open(row["image_path"]) as image:
            assert image.getpixel((0, 0)) == color
        with Image.open(row["standardized_mask_path"]) as mask:
            assert mask.getpixel((0, 0)) == 0
        assert row["output_eye_distance"] == pytest.approx(40)
        assert row["alignment"] == "center"
    recipe = json.loads((destination / "standardization.json").read_text(encoding="utf-8"))
    assert recipe["eye_distance"] == 0.20
    assert recipe["detector"].endswith(".detect")


@pytest.mark.parametrize("eye_distance", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_eye_distance_is_rejected(eye_distance):
    with pytest.raises(ValueError, match="eye_distance"):
        FaceStandardizer(eye_distance=eye_distance)


def test_default_batch_needs_no_models_or_landmarks(tmp_path, monkeypatch):
    source = save_portrait(tmp_path / "source.png")
    processor = FaceStandardizer()

    def fail(*args, **kwargs):
        pytest.fail("Basic center crop batches must not load or invoke any models.")

    for name in ("_load_detector", "_detect", "_load_segmenter", "_segment"):
        monkeypatch.setattr(processor, name, fail)

    destination = tmp_path / "output"
    result = processor.run([source], destination)

    assert len(result.ready) == 1
    assert result.ready.iloc[0]["alignment"] == "center"
    with Image.open(result.ready.iloc[0]["standardized_mask_path"]) as mask:
        assert mask.getextrema() == (255, 255)
    recipe = json.loads((destination / "standardization.json").read_text(encoding="utf-8"))
    assert recipe["alignment"] == "center"
    assert recipe["background"] == "auto"
    assert recipe["detector"] is None
    assert recipe["segmenter"] is None


def test_center_detail_cap_uses_detection_only_for_measurements():
    calls = []
    image, _ = portrait()

    def detect(oriented):
        calls.append(oriented.size)
        return POINTS

    result = FaceStandardizer(size=image.size, detail_eye_distance=20,
                              detector=detect).transform(image)

    assert calls == [image.size]
    np.testing.assert_allclose(json.loads(result.metadata["transform_matrix"]), [[1, 0, 0], [0, 1, 0]])
    assert result.metadata["native_eye_distance"] == result.metadata["output_eye_distance"] == 70
    assert result.metadata["detail_eye_distance"] == 20


def test_center_enhancer_receives_landmarks_without_changing_crop_geometry():
    image, _ = portrait()
    tilted = np.array([[25, 75], [85, 105], [50, 123], [35, 150], [67, 166]], dtype=float)
    detections, enhancements = [], []

    def detect(oriented):
        detections.append(oriented.size)
        return tilted.copy()

    def enhance(oriented, *, landmarks):
        enhancements.append(landmarks.copy())
        ImageDraw.Draw(oriented).rectangle((188, 228, 192, 232), fill=(230, 60, 10))
        return oriented

    result = FaceStandardizer(size=image.size, detector=detect, enhancer=enhance).transform(image)

    assert detections == [image.size]
    np.testing.assert_array_equal(enhancements, [tilted])
    np.testing.assert_allclose(json.loads(result.metadata["transform_matrix"]), [[1, 0, 0], [0, 1, 0]])
    assert result.image.getpixel((190, 230)) == (230, 60, 10)
    assert result.baseline.tobytes() == image.tobytes()
    assert "enhancement_requires_review" in result.metadata["quality_flags"].split(";")


def test_eye_alignment_auto_background_matches_dominant_visible_border():
    color = (180, 100, 60)
    image, _ = portrait(color)
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((50, 30, 150, 215), fill=(30, 80, 160))
    # Small border artifacts must not choose the padding color.
    drawing.rectangle((0, 0, 2, 2), fill=(0, 255, 0))
    processor = FaceStandardizer(alignment="eyes", size=image.size,
                                 eye_distance=0.35, eye_height=0.50)

    result = processor.transform(image, landmarks=POINTS)

    assert result.image.getpixel((100, 10)) == color
    assert result.mask.getpixel((100, 10)) == 0
    assert json.loads(result.metadata["background_color"]) == list(color)
    assert result.metadata["background_source"] == "auto"
    assert result.metadata["alignment"] == "eyes"


@pytest.mark.parametrize("hidden_alpha", [0, 1])
def test_auto_background_ignores_hidden_rgb_and_weights_visible_alpha(hidden_alpha):
    image = Image.new("RGBA", (200, 240), (255, 0, 255, hidden_alpha))
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, 4, 239), fill=(180, 100, 60, 255))
    drawing.rectangle((30, 30, 170, 220), fill=(30, 80, 160, 255))
    processor = FaceStandardizer(alignment="eyes", size=image.size,
                                 eye_distance=0.35, eye_height=0.50)

    result = processor.transform(image, landmarks=POINTS)

    assert result.image.getpixel((100, 10)) == (180, 100, 60)
    assert json.loads(result.metadata["background_color"]) == [180, 100, 60]


def test_auto_background_falls_back_when_boundary_is_fully_transparent():
    image = Image.new("RGBA", (200, 240), (255, 0, 255, 0))
    ImageDraw.Draw(image).rectangle((30, 30, 170, 220), fill=(30, 80, 160, 255))
    result = FaceStandardizer(alignment="eyes", size=image.size,
                              eye_distance=0.35, eye_height=0.50).transform(image, landmarks=POINTS)

    assert result.image.getpixel((100, 10)) == (224, 224, 224)
    assert json.loads(result.metadata["background_color"]) == [224, 224, 224]
    assert result.metadata["background_source"] == "auto"


def test_eye_alignment_auto_background_is_resolved_per_batch_image(tmp_path):
    colors = [(180, 100, 60), (30, 80, 160)]
    sources = [save_portrait(tmp_path / f"source-{index}.png", color)
               for index, color in enumerate(colors)]
    frame = pd.DataFrame({"image_path": sources, "landmarks": [POINTS.tolist()] * 2})
    processor = FaceStandardizer(alignment="eyes", size=(200, 240),
                                 eye_distance=0.35, eye_height=0.50)

    result = processor.run(frame, tmp_path / "output")

    assert result.manifest["status"].tolist() == ["ready", "ready"]
    for row, color in zip(result.manifest.to_dict("records"), colors):
        with Image.open(row["image_path"]) as image:
            assert image.getpixel((100, 10)) == color
        assert json.loads(row["background_color"]) == list(color)
        assert row["background_source"] == "auto"
    assert processor.background == "auto"


def test_explicit_background_overrides_auto_color_in_eye_alignment():
    image, _ = portrait((180, 100, 60))
    result = FaceStandardizer(alignment="eyes", size=image.size, eye_distance=0.35,
                              eye_height=0.50, background=(11, 22, 33)).transform(image, landmarks=POINTS)

    assert result.image.getpixel((100, 10)) == (11, 22, 33)
    assert json.loads(result.metadata["background_color"]) == [11, 22, 33]
    assert result.metadata["background_source"] == "supplied"


@pytest.mark.parametrize("alignment", [None, "", "eye", "CENTER", 0, ["center"]])
def test_invalid_alignment_is_rejected(alignment):
    with pytest.raises((TypeError, ValueError), match="alignment"):
        FaceStandardizer(alignment=alignment)


@pytest.mark.parametrize("background", [None, "white", "AUTO", (0, 0), (0, 0, 256),
                                        (0, 0, -1), (0, 0, True), (0, 0, 1.5)])
def test_invalid_background_is_rejected(background):
    with pytest.raises((TypeError, ValueError), match="background"):
        FaceStandardizer(background=background)


def test_eye_alignment_keeps_background_during_rotation_scaling_and_crop():
    image, _ = portrait((15, 30, 60))
    ImageDraw.Draw(image).rectangle((10, 50, 25, 65), fill=(230, 60, 10))
    tilted = np.array([[60, 70], [120, 100], [80, 118], [65, 145], [97, 161]], dtype=float)
    processor = FaceStandardizer(alignment="eyes", size=image.size, eye_distance=0.30, eye_height=0.40)

    result = processor.transform(image, landmarks=tilted)

    matrix = np.array(json.loads(result.metadata["transform_matrix"]))
    aligned = tilted @ matrix[:, :2].T + matrix[:, 2]
    np.testing.assert_allclose(aligned[:2], [[70, 96], [130, 96]])
    location = tuple(np.rint(matrix @ [17, 57, 1]).astype(int))
    assert result.image.getpixel(location) == (230, 60, 10)
    assert result.mask.getpixel(location) == 255
    assert result.image.size == (200, 240)
    assert result.metadata["background_removed"] is False
    assert not set(result.metadata["quality_flags"].split(";")) & {
        "source_head_touches_top", "source_head_touches_side",
        "output_head_touches_edge", "source_bottom_in_output",
    }
    assert not result.metadata["review_required"]


def test_eye_alignment_preserves_source_alpha_and_uses_background_for_transformed_padding():
    image, _ = portrait((180, 100, 60))
    image = image.convert("RGBA")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((80, 160, 120, 185), fill=(180, 100, 60, 128))
    drawing.rectangle((95, 170, 105, 180), fill=(255, 0, 0, 0))
    processor = FaceStandardizer(alignment="eyes", size=image.size, eye_distance=0.35, eye_height=0.50,
                                 background=(11, 22, 33))

    result = processor.transform(image, landmarks=POINTS)

    # Eye placement translates the image down by 35 pixels, leaving top padding.
    assert result.mask.getpixel((100, 10)) == 0
    assert result.image.getpixel((100, 10)) == (11, 22, 33)
    assert result.mask.getpixel((0, 35)) == 255
    assert result.image.getpixel((0, 35)) == (180, 100, 60)
    assert result.mask.getpixel((100, 200)) == 128
    np.testing.assert_allclose(result.image.getpixel((100, 200)), (96, 61, 47), atol=1)
    assert result.mask.getpixel((100, 210)) == 0
    assert result.image.getpixel((100, 210)) == (11, 22, 33)
    assert result.metadata["mask_source"] == "source_alpha"
    assert result.metadata["background_removed"] is False
    assert not result.metadata["review_required"]


def test_default_accepts_translucent_source_without_segmentation_validation():
    image = Image.new("RGBA", (200, 240), (180, 100, 60, 100))
    processor = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240,
                                 background=(11, 22, 33))

    result = processor.transform(image, landmarks=POINTS)

    np.testing.assert_array_equal(np.asarray(result.mask), np.full((240, 200), 100))
    np.testing.assert_array_equal(np.asarray(result.image), np.full((240, 200, 3), [77, 53, 44]))
    assert result.metadata["background_removed"] is False
    assert not result.metadata["review_required"]


@pytest.mark.parametrize("options,points,flag", [
    ({"size": (400, 480), "eye_distance": 0.35}, POINTS, "insufficient_native_resolution"),
    ({"eye_height": 0.80}, POINTS, "landmark_outside_output"),
    ({}, np.array([[65, 85], [135, 85], [125, 115], [84, 145], [116, 145]]), "possible_nonfrontal_pose"),
])
def test_eye_alignment_retains_resolution_landmark_and_pose_review(options, points, flag):
    configuration = {"alignment": "eyes", "size": (200, 240), "eye_distance": 0.35, "eye_height": 85 / 240}
    configuration.update(options)

    result = FaceStandardizer(**configuration).transform(portrait()[0], landmarks=points)

    assert flag in result.metadata["quality_flags"].split(";")
    assert result.metadata["review_required"]


def test_background_removal_is_enabled_explicitly_for_custom_segmenter():
    image, mask = portrait((180, 100, 60))
    calls = []

    def segment(oriented):
        calls.append(oriented.size)
        return mask

    processor = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240,
                                 background=(11, 22, 33), segmenter=segment,
                                 remove_background=True)

    result = processor.transform(image, landmarks=POINTS)

    assert calls == [image.size]
    assert result.image.getpixel((0, 0)) == (11, 22, 33)
    assert result.mask.getpixel((0, 0)) == 0
    assert result.metadata["background_removed"] is True
    assert result.metadata["mask_source"].endswith(".segment")
    assert processor._recipe()["remove_background"] is True
    assert processor._recipe()["segmenter"].endswith(".segment")


def test_background_replacement_respects_foreground_and_original_alpha():
    image, mask = portrait((180, 100, 60))
    image = image.convert("RGBA")
    # An existing transparent hole cannot be filled by an opaque supplied mask.
    ImageDraw.Draw(image).rectangle((95, 170, 105, 180), fill=(255, 0, 0, 0))
    processor = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240,
                                 background=(11, 22, 33))
    result = processor.transform(image, landmarks=POINTS, mask=mask)
    pixels, foreground = np.asarray(result.image), np.asarray(result.mask)

    np.testing.assert_array_equal(pixels[foreground == 0], np.tile([11, 22, 33], ((foreground == 0).sum(), 1)))
    assert result.image.getpixel((80, 115)) == (180, 100, 60)
    assert result.image.getpixel((100, 175)) == (11, 22, 33)
    assert result.mask.getpixel((100, 175)) == 0
    assert result.metadata["background_removed"] is True
    assert result.metadata["mask_source"] == "supplied"


def test_existing_soft_alpha_is_composited_only_once():
    image, mask = portrait((180, 100, 60))
    image = image.convert("RGBA")
    ImageDraw.Draw(image).rectangle((80, 160, 120, 185), fill=(180, 100, 60, 128))
    processor = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240,
                                 background=(11, 22, 33))

    result = processor.transform(image, landmarks=POINTS, mask=mask)

    assert result.mask.getpixel((100, 175)) == 128
    np.testing.assert_allclose(result.image.getpixel((100, 175)), (96, 61, 47), atol=1)


def test_detail_ceiling_removes_fine_detail_without_changing_canvas_or_eye_scale():
    _, mask = portrait()
    checkerboard = ((np.indices((240, 200)).sum(axis=0) % 2) * 255).astype(np.uint8)
    image = Image.fromarray(checkerboard).convert("RGB")
    full = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240)
    limited = FaceStandardizer(size=image.size, eye_distance=0.35, eye_height=85 / 240,
                               detail_eye_distance=20)

    original = full.transform(image, landmarks=POINTS, mask=mask)
    result = limited.transform(image, landmarks=POINTS, mask=mask)

    assert result.image.size == original.image.size
    assert result.metadata["output_eye_distance"] == original.metadata["output_eye_distance"] == 70
    assert result.metadata["detail_eye_distance"] == 20
    assert np.asarray(result.image)[100:140, 75:125].std() < np.asarray(original.image)[100:140, 75:125].std() / 10
    assert not result.metadata["review_required"]


def test_low_native_resolution_is_flagged_instead_of_claiming_restored_detail():
    image, mask = portrait()
    result = FaceStandardizer(alignment="eyes", size=(400, 480), eye_distance=0.35).transform(
        image, landmarks=POINTS, mask=mask,
    )

    assert result.metadata["native_eye_distance"] == 70
    assert result.metadata["output_eye_distance"] == 140
    assert result.metadata["review_required"]
    assert "insufficient_native_resolution" in result.metadata["quality_flags"]


def test_exif_orientation_is_applied_before_detector_and_segmenter(tmp_path):
    image, mask = portrait()
    ImageDraw.Draw(image).rectangle((50, 50, 70, 70), fill=(255, 0, 0))
    raw = image.transpose(Image.Transpose.ROTATE_90)
    exif = raw.getexif()
    exif[274] = 6  # Stored pixels need a clockwise turn to become upright.
    path = tmp_path / "oriented.png"
    raw.save(path, exif=exif)
    seen = []

    def detect(oriented):
        seen.append((oriented.size, oriented.getpixel((60, 60))))
        return POINTS

    processor = FaceStandardizer(alignment="eyes", size=image.size, detector=detect, segmenter=lambda oriented: mask,
                                 remove_background=True)
    expected = processor.transform(image)
    actual = processor.transform(path)

    assert seen == [((200, 240), (255, 0, 0))] * 2
    assert actual.image.tobytes() == expected.image.tobytes()
    assert actual.mask.tobytes() == expected.mask.tobytes()
    assert (actual.metadata["input_width"], actual.metadata["input_height"]) == image.size
    with Image.open(path) as saved:
        assert saved.getexif()[274] == 6 and saved.size == (240, 200)


def test_manual_measurements_bypass_models_for_transform_and_batch(tmp_path, monkeypatch):
    source = save_portrait(tmp_path / "source.png")
    image, mask = portrait()
    mask_path = tmp_path / "mask.png"
    mask.save(mask_path)
    processor = FaceStandardizer(size=(200, 240), eye_distance=0.30)

    def fail(*args, **kwargs):
        pytest.fail("Supplying both measurements must not load or invoke any models.")

    for name in ("_load_detector", "_load_segmenter", "_detect", "_segment"):
        monkeypatch.setattr(processor, name, fail)
    assert processor.transform(image, landmarks=POINTS, mask=mask).image.size == (200, 240)
    frame = pd.DataFrame({"image_path": [source], "landmarks": [POINTS.tolist()], "mask_path": [mask_path]})
    result = processor.run(frame, tmp_path / "output")
    assert len(result.ready) == 1


def test_default_batch_skips_segmentation_and_applies_only_explicit_manifest_masks(tmp_path, monkeypatch):
    plain = save_portrait(tmp_path / "plain.png", (180, 100, 60))
    masked = save_portrait(tmp_path / "masked.png", (180, 100, 60))
    mask_path = tmp_path / "mask.png"
    portrait()[1].save(mask_path)

    def fail(*args, **kwargs):
        pytest.fail("Default batches must not load or invoke segmentation.")

    processor = FaceStandardizer(size=(200, 240), eye_distance=0.35, eye_height=85 / 240,
                                 segmenter=fail, background=(11, 22, 33))
    monkeypatch.setattr(processor, "_load_segmenter", fail)
    monkeypatch.setattr(processor, "_segment", fail)
    frame = pd.DataFrame({"image_path": [plain, masked], "landmarks": [POINTS.tolist()] * 2,
                          "mask_path": [None, mask_path]})
    destination = tmp_path / "output"

    result = processor.run(frame, destination)

    assert result.manifest["status"].tolist() == ["ready", "ready"]
    assert result.manifest["background_removed"].tolist() == [False, True]
    assert result.manifest["mask_source"].tolist() == ["source_alpha", "supplied"]
    for row, expected_color, expected_mask in zip(result.manifest.to_dict("records"),
                                                 [(180, 100, 60), (11, 22, 33)], [255, 0]):
        with Image.open(row["image_path"]) as image:
            assert image.getpixel((0, 0)) == expected_color
        with Image.open(row["standardized_mask_path"]) as saved_mask:
            assert saved_mask.getpixel((0, 0)) == expected_mask
    recipe = json.loads((destination / "standardization.json").read_text(encoding="utf-8"))
    assert recipe["remove_background"] is False
    assert recipe["segmenter"] is None


def test_batch_preserves_metadata_updates_paths_and_separates_review_and_failure(tmp_path):
    good = save_portrait(tmp_path / "source" / "good.png")
    review = save_portrait(tmp_path / "source" / "review.png")
    broken = tmp_path / "source" / "broken.jpg"
    broken.write_bytes(b"not an image")
    good_mask = tmp_path / "source" / "mask.png"
    review_mask = tmp_path / "source" / "review_mask.png"
    _, mask = portrait()
    mask.save(good_mask)
    ImageDraw.Draw(mask).rectangle((90, 0, 110, 30), fill=255)
    mask.save(review_mask)
    frame = pd.DataFrame({
        "image_id": ["real-good", "synthetic-review", "real-broken"],
        "target_id": ["person-a", "person-b", "person-c"],
        "image_path": [good, review, broken],
        "landmarks": [json.dumps(POINTS.tolist())] * 3,
        "mask_path": [good_mask.name, review_mask.name, good_mask.name],
        "source_dataset": ["camera-a", "generator", "camera-b"],
        "is_synthetic": [False, True, False],
        "custom_attribute": ["value-a", "value-b", "value-c"],
    })
    catalog = ImageCatalog.from_manifest(frame, base_dir=good.parent)
    original_manifest = catalog.manifest.copy(deep=True)
    input_bytes = good.read_bytes()
    destination = tmp_path / "output"

    result = standardizer().run(catalog, destination)

    pd.testing.assert_frame_equal(catalog.manifest, original_manifest)
    assert result.manifest["status"].tolist() == ["ready", "review", "error"]
    assert result.ready["image_id"].tolist() == ["real-good"]
    assert result.manifest.loc[2, "error"]
    for field in ("image_id", "target_id", "source_dataset", "is_synthetic", "custom_attribute"):
        assert result.manifest[field].tolist() == frame[field].tolist()
    for row in result.manifest.to_dict("records"):
        output = Path(row["image_path"])
        assert output.parent == destination / "images"
        assert row["image_name"] == output.name
        assert row["relative_image_path"] == output.relative_to(destination).as_posix()
        assert row["file_extension"] == ".png"
        assert Path(row["standardization_input_path"]).parent == good.parent
        assert output.exists() == (row["status"] != "error")
        assert Path(row["standardized_mask_path"]).exists() == (row["status"] != "error")
    assert result.manifest.loc[0, "standardization_input_sha256"] == hashlib.sha256(input_bytes).hexdigest()
    assert good.read_bytes() == input_bytes
    reloaded = ImageCatalog.from_manifest(pd.read_csv(destination / "manifest.csv"), base_dir=destination)
    assert reloaded.ready["image_id"].tolist() == ["real-good"]
    recipe = json.loads((destination / "standardization.json").read_text(encoding="utf-8"))
    assert recipe["size"] == [200, 240]


def test_existing_review_status_does_not_become_ready(tmp_path):
    source = save_portrait(tmp_path / "source.png")
    frame = pd.DataFrame({"image_path": [source], "status": ["review"], "error": ["manual review pending"]})
    result = standardizer().run(frame, tmp_path / "output")
    assert result.ready.empty
    assert result.manifest.loc[0, "status"] == "review"
    assert result.manifest.loc[0, "error"] == "manual review pending"
    assert not Path(result.manifest.loc[0, "image_path"]).exists()


@pytest.mark.parametrize("input_kind", ["folder", "paths"])
def test_same_filename_from_different_datasets_stays_distinct(tmp_path, input_kind):
    root = tmp_path / "sources"
    first = save_portrait(root / "dataset-a" / "face.png", color=(200, 40, 50))
    second = save_portrait(root / "dataset-b" / "face.png", color=(30, 180, 70))
    inputs = root if input_kind == "folder" else [first, second]

    result = standardizer().run(inputs, tmp_path / "output")

    assert len(result.ready) == 2
    assert result.ready["image_id"].is_unique and result.ready["image_path"].is_unique
    colors = []
    for output in result.ready["image_path"]:
        with Image.open(output) as image:
            colors.append(image.getpixel((100, 125)))
    assert colors == [(200, 40, 50), (30, 180, 70)]
    assert result.manifest["standardization_input_path"].tolist() == [str(first), str(second)]


def test_output_is_never_overwritten(tmp_path):
    source = save_portrait(tmp_path / "source.png")
    destination = tmp_path / "output"
    processor = standardizer()
    result = processor.run([source], destination)
    before = {path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()}

    with pytest.raises(FileExistsError, match="never overwritten"):
        processor.run([source], destination)

    assert {path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before
    assert len(result.ready) == 1


@pytest.mark.parametrize("loader, failure", [
    ("_load_detector", ImportError("Optional detector dependency is unavailable.")),
    ("_load_segmenter", OSError("The global segmentation model cannot be loaded.")),
])
def test_global_model_failure_surfaces_before_creating_output(tmp_path, monkeypatch, loader, failure):
    source = save_portrait(tmp_path / "source.png")
    destination = tmp_path / "output"
    processor = standardizer()

    def fail():
        raise failure

    monkeypatch.setattr(processor, loader, fail)
    with pytest.raises(type(failure)) as raised:
        processor.run([source], destination)

    assert raised.value is failure
    assert not destination.exists()
    assert source.is_file()


def test_rotated_source_bottom_cut_inside_output_requires_review(tmp_path):
    image, mask = portrait()
    tilted = np.array([[60, 70], [120, 100], [80, 118], [65, 145], [97, 161]], dtype=float)
    processor = standardizer()
    complete = processor.transform(image, landmarks=tilted, mask=mask)
    assert "source_bottom_in_output" not in complete.metadata["quality_flags"]
    # A neck truncated by the source's lower border leaves a diagonal cut after alignment.
    ImageDraw.Draw(mask).rectangle((60, 170, 120, 239), fill=255)
    result = processor.transform(image, landmarks=tilted, mask=mask)
    matrix = np.array(json.loads(result.metadata["transform_matrix"]))
    bottom_midpoint = matrix @ np.array([90, 239, 1])
    assert np.all(bottom_midpoint > 2) and np.all(bottom_midpoint < np.array(result.image.size) - 2)
    assert "source_bottom_in_output" in result.metadata["quality_flags"]
    assert result.metadata["review_required"]

    source, mask_path = tmp_path / "portrait.png", tmp_path / "mask.png"
    image.save(source)
    mask.save(mask_path)
    frame = pd.DataFrame({"image_path": [source], "landmarks": [tilted.tolist()], "mask_path": [mask_path]})
    catalog = processor.run(frame, tmp_path / "output")
    assert catalog.manifest.loc[0, "status"] == "review"
    assert catalog.ready.empty


@pytest.mark.parametrize("landmarks", [np.zeros((4, 2)), np.full((5, 2), np.nan), np.zeros((5, 2))])
def test_invalid_landmarks_fail_without_returning_a_portrait(landmarks):
    image, mask = portrait()
    with pytest.raises(ValueError):
        standardizer().transform(image, landmarks=landmarks, mask=mask)


@pytest.mark.parametrize("mask", [Image.new("L", (200, 240)), Image.new("RGB", (200, 240)), Image.new("L", (100, 120), 255)])
def test_invalid_segmentation_fails_without_returning_a_portrait(mask):
    with pytest.raises(ValueError, match="mask|segmentation"):
        standardizer().transform(portrait()[0], landmarks=POINTS, mask=mask)
