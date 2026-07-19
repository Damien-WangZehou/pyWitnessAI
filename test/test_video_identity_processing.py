import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from pyWitnessAI.VideoAnalysis import FaceDetection, FaceIdentityTracker
from pyWitnessAI.VideoAI import (
    FrameAnalyzerDeepFace,
    FrameAnalyzerFaceIdentity,
    FrameAnalyzerMTCNN,
)
from pyWitnessAI.Video import Video
from pyWitnessAI.VideoProcessor import FrameProcessorFaceMasker, FrameProcessorFaceRemover


def _mean_embedding(face_img):
    return face_img.mean(axis=(0, 1)).astype("float32")


def test_face_identity_tracker_assigns_consistent_labels_across_frames():
    frame_a = np.zeros((20, 40, 3), dtype=np.uint8)
    frame_a[:, 20:40] = 255
    frame_b = np.zeros((20, 40, 3), dtype=np.uint8)
    frame_b[:, 0:20] = 255

    tracker = FaceIdentityTracker(
        max_distance=0.25,
        embedding_function=_mean_embedding,
    )

    labels_a = tracker.update_frame(
        frame_index=0,
        frame=frame_a,
        detections=[
            FaceDetection.from_box([0, 0, 20, 20], confidence=1.0),
            FaceDetection.from_box([20, 0, 20, 20], confidence=1.0),
        ],
        analyzer="identity",
    )
    labels_b = tracker.update_frame(
        frame_index=1,
        frame=frame_b,
        detections=[
            FaceDetection.from_box([0, 0, 20, 20], confidence=1.0),
            FaceDetection.from_box([20, 0, 20, 20], confidence=1.0),
        ],
        analyzer="identity",
    )

    assert labels_a == ["face1", "face2"]
    assert labels_b == ["face2", "face1"]


def test_frame_processor_face_remover_keeps_selected_label_from_context():
    frame = np.zeros((12, 24, 3), dtype=np.uint8)
    result = {
        "face_count": 2,
        "face_area": 128,
        "confidence": [1.0, 1.0],
        "coordinates": [[0, 0, 8, 8], [12, 0, 8, 8]],
        "labels": ["face1", "face2"],
    }

    processor = FrameProcessorFaceRemover(keep=["face1"])
    processed = processor.process_frame(
        frame.copy(),
        context={"frame": 0, "frame_index": 0, "analysis": {"identity": result}},
    )

    assert processed[2, 2].tolist() == [0, 0, 0]
    assert processed[2, 14].tolist() == [128, 128, 128]


def test_deepface_analyzer_wraps_backend_and_preserves_legacy_output(monkeypatch):
    calls = []

    def fake_extract_faces(img_path, detector_backend, enforce_detection):
        calls.append((detector_backend, enforce_detection))
        return [
            {
                "facial_area": {"x": 1, "y": 2, "w": 4, "h": 5},
                "confidence": 0.9,
            }
        ]

    monkeypatch.setattr("pyWitnessAI.VideoAI.DeepFace.extract_faces", fake_extract_faces)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    analyzer = FrameAnalyzerDeepFace(detector_backend="retinaface", include_average_confidence=True)
    output = analyzer.analyze_frame(frame)

    assert calls == [("retinaface", False)]
    assert output["face_count"] == 1
    assert output["coordinates"] == [[1, 2, 4, 5]]
    assert output["confidence"] == [0.9]
    assert output["average_confidence"] == 0.9
    assert analyzer.detected_faces[0]["coordinates"] == [[1, 2, 4, 5]]


def test_mtcnn_wrapper_keeps_original_defaults(monkeypatch):
    def fake_extract_faces(img_path, detector_backend, enforce_detection):
        assert detector_backend == "mtcnn"
        return []

    monkeypatch.setattr("pyWitnessAI.VideoAI.DeepFace.extract_faces", fake_extract_faces)

    analyzer = FrameAnalyzerMTCNN()
    output = analyzer.analyze_frame(np.zeros((12, 12, 3), dtype=np.uint8))

    assert analyzer.name == "mtcnn"
    assert analyzer.detect_backend == "mtcnn"
    assert output == {
        "face_count": 0,
        "face_area": 0,
        "confidence": [],
        "coordinates": [],
    }


def test_tracker_mugbook_keeps_highest_quality_samples():
    tracker = FaceIdentityTracker(
        max_distance=0.1,
        max_samples_per_label=1,
        embedding_function=_mean_embedding,
    )
    frame = np.full((20, 20, 3), 64, dtype=np.uint8)

    tracker.update_frame(
        4,
        frame,
        [FaceDetection.from_box([0, 0, 4, 4], confidence=0.99)],
    )
    tracker.update_frame(
        8,
        frame,
        [FaceDetection.from_box([0, 0, 10, 10], confidence=0.90)],
    )

    entry = tracker.gallery["face1"]
    assert entry["count"] == 2
    assert entry["first_seen_frame"] == 4
    assert entry["last_seen_frame"] == 8
    assert entry["best_samples"][0]["frame"] == 8
    assert entry["best_samples"][0]["quality_score"] == 0.225


def test_face_masker_supports_mosaic_and_preserves_kept_identity():
    frame = np.zeros((12, 24, 3), dtype=np.uint8)
    gradient = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    frame[0:8, 12:20] = gradient[:, :, None]
    result = {
        "coordinates": [[0, 0, 8, 8], [12, 0, 8, 8]],
        "confidence": [1.0, 1.0],
        "labels": ["face1", "face2"],
    }
    processor = FrameProcessorFaceMasker(
        keep=["face1"], mask_style="mosaic", mosaic_blocks=4
    )

    processed = processor.process_frame(
        frame.copy(),
        context={"frame_index": 0, "analysis": {"identity": result}},
    )

    assert np.array_equal(processed[:, :8], frame[:, :8])
    assert len(np.unique(processed[0:8, 12:20, 0])) < len(np.unique(gradient))
    assert not np.all(processed[0:8, 12:20] == 128)


def test_top_probe_frames_are_ranked_per_identity_with_real_frame_numbers():
    video = Video("does-not-exist.mp4")
    video.frame_count = [10, 20, 30]
    video.frame_area = 1000
    video._gallery_built_from = "identity"
    video.frame_analyzer_output["identity"] = [
        {
            "coordinates": [[0, 0, 10, 10], [20, 0, 10, 10]],
            "confidence": [0.95, 0.80],
            "labels": ["face1", "face2"],
        },
        {
            "coordinates": [[0, 0, 20, 10]],
            "confidence": [0.80],
            "labels": ["face1"],
        },
        {
            "coordinates": [[0, 0, 12, 10]],
            "confidence": [0.90],
            "labels": ["face2"],
        },
    ]

    probes = video.find_top_probe_frames(top_k=1)

    assert isinstance(probes, pd.DataFrame)
    assert probes.set_index("label").loc["face1", "frame"] == 20
    assert probes.set_index("label").loc["face1", "appearance_frame_count"] == 2
    assert probes.set_index("label").loc["face2", "frame"] == 30

    figure, axes = video.plot_video_quality(detector="identity")
    assert len(axes) == 4
    assert axes[1].get_title() == "Identity Presence / Detection Continuity"
    plt.close(figure)


def test_second_pass_video_preserves_fps_and_masks_other_identity(tmp_path):
    source_path = str(tmp_path / "source.mp4")
    output_path = str(tmp_path / "face1_only.mp4")
    writer = cv.VideoWriter(
        source_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        7.0,
        (64, 32),
    )
    assert writer.isOpened()
    for _ in range(3):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, :32] = (20, 40, 220)
        frame[:, 32:] = (220, 40, 20)
        writer.write(frame)
    writer.release()

    video = Video(source_path)
    video.frame_count = [0, 1, 2]
    video._gallery_built_from = "identity"
    video.face_records = [
        {
            "frame": frame,
            "face_index": face_index,
            "label": label,
            "bbox_x": x,
            "bbox_y": 0,
            "bbox_w": 32,
            "bbox_h": 32,
            "confidence": 1.0,
            "area": 1024,
            "embedding_distance": 0.0,
            "analyzer": "identity",
        }
        for frame in range(3)
        for face_index, (label, x) in enumerate((("face1", 0), ("face2", 32)))
    ]

    saved = video.save_face_filtered_video(
        output_path,
        keep=["face1"],
        mask_style="gray",
    )

    capture = cv.VideoCapture(saved)
    assert capture.isOpened()
    assert abs(capture.get(cv.CAP_PROP_FPS) - 7.0) < 0.1
    assert int(capture.get(cv.CAP_PROP_FRAME_COUNT)) == 3
    ok, rendered = capture.read()
    capture.release()
    assert ok
    assert np.abs(rendered[:, 40:].astype(float).mean(axis=(0, 1)) - 128).max() < 8
    assert rendered[:, :24, 2].mean() > 150


def test_video_identity_analyzer_builds_mugbook_in_one_pass(tmp_path, monkeypatch):
    source_path = str(tmp_path / "identities.mp4")
    writer = cv.VideoWriter(
        source_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (40, 20),
    )
    assert writer.isOpened()
    first = np.zeros((20, 40, 3), dtype=np.uint8)
    first[:, 20:] = 255
    second = np.zeros((20, 40, 3), dtype=np.uint8)
    second[:, :20] = 255
    writer.write(first)
    writer.write(second)
    writer.release()

    monkeypatch.setattr(
        "pyWitnessAI.VideoAI.DeepFace.extract_faces",
        lambda **kwargs: [
            {"facial_area": {"x": 0, "y": 0, "w": 20, "h": 20}, "confidence": 0.9},
            {"facial_area": {"x": 20, "y": 0, "w": 20, "h": 20}, "confidence": 0.9},
        ],
    )
    analyzer = FrameAnalyzerFaceIdentity(
        embedding_function=lambda face: np.array(
            [face.mean() / 255.0, 1.0 - face.mean() / 255.0], dtype=np.float32
        ),
        max_distance=0.25,
    )
    video = Video(source_path)
    video.add_analyzer(analyzer)

    video.run(frame_end=None)

    assert video.frame_analyzer_output["identity"][0]["labels"] == ["face1", "face2"]
    assert video.frame_analyzer_output["identity"][1]["labels"] == ["face2", "face1"]
    assert video.list_face_labels() == ["face1", "face2"]
    mugbook = video.get_mugbook().set_index("label")
    assert mugbook.loc["face1", "appearance_frame_count"] == 2
    assert mugbook.loc["face2", "appearance_frame_count"] == 2


def test_save_data_keeps_identity_fields_that_appear_after_first_frame(tmp_path):
    video = Video("does-not-exist.mp4")
    video.frame_count = [0, 1]
    video.average_pixel_values = [10, 11]
    video.frame_analyzer_output["identity"] = [
        {"face_count": 0, "face_area": 0, "confidence": [], "coordinates": []},
        {
            "face_count": 1,
            "face_area": 100,
            "confidence": [0.9],
            "coordinates": [[0, 0, 10, 10]],
            "labels": ["face1"],
            "embedding_distance": [0.2],
        },
    ]

    video.save_data(directory=str(tmp_path), prefix="identity")
    saved = pd.read_csv(tmp_path / "identity_data.csv")

    assert "identity_labels" in saved
    assert "identity_embedding_distance" in saved
    assert pd.isna(saved.loc[0, "identity_labels"])
