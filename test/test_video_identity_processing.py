import numpy as np

from pyWitnessAI.VideoAnalysis import FaceDetection, FaceIdentityTracker
from pyWitnessAI.VideoAI import FrameAnalyzerDeepFace, FrameAnalyzerMTCNN
from pyWitnessAI.VideoProcessor import FrameProcessorFaceRemover


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
