from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import cv2 as cv
import numpy as np


BoxXYWH = tuple[int, int, int, int]
EmbeddingFunction = Callable[[np.ndarray], np.ndarray]


def normalize_box_xywh(box: Iterable[Any], frame_shape: tuple[int, ...] | None = None) -> BoxXYWH:
    # Normalize a bounding box to the frame dimensions
    values = list(box)
    if len(values) != 4:
        raise ValueError(f"Expected a 4-value bounding box, got {box!r}.")

    x, y, w, h = (int(round(float(v))) for v in values)
    if frame_shape is None:
        return max(0, x), max(0, y), max(0, w), max(0, h)

    frame_h, frame_w = frame_shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def crop_box_xywh(frame: np.ndarray, box: Iterable[Any], padding: int = 0) -> np.ndarray:
    x, y, w, h = normalize_box_xywh(box, frame.shape)
    if padding:
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame_w, x + w + padding)
        y2 = min(frame_h, y + h + padding)
        return frame[y1:y2, x1:x2]
    return frame[y:y + h, x:x + w]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@dataclass
class FaceDetection:
    bbox: BoxXYWH
    confidence: float = 0.0
    label: str | None = None
    embedding_distance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_box(
        cls,
        box: Iterable[Any],
        confidence: float = 0.0,
        label: str | None = None,
        embedding_distance: float | None = None,
        frame_shape: tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "FaceDetection":
        return cls(
            bbox=normalize_box_xywh(box, frame_shape),
            confidence=float(confidence or 0.0),
            label=label,
            embedding_distance=embedding_distance,
            metadata=dict(metadata or {}),
        )

    @property
    def area(self) -> int:
        return int(self.bbox[2]) * int(self.bbox[3])

    def quality_score(self, frame_area: int | None = None) -> float:
        """Return a resolution-independent confidence/size quality score."""
        area = self.area / frame_area if frame_area else self.area
        return max(0.0, float(self.confidence)) * float(area)

    def to_record(self, frame_index: int | None, face_index: int, analyzer: str | None = None) -> dict[str, Any]:
        x, y, w, h = self.bbox
        record = {
            "frame": frame_index,
            "face_index": face_index,
            "label": self.label,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "confidence": self.confidence,
            "area": self.area,
            "embedding_distance": self.embedding_distance,
        }
        if analyzer is not None:
            record["analyzer"] = analyzer
        return record


@dataclass
class FrameAnalysisResult:
    frame_index: int | None = None
    detections: list[FaceDetection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.detections)

    @property
    def face_area(self) -> int:
        return int(sum(face.area for face in self.detections))

    def to_dict(self, include_average_confidence: bool = False) -> dict[str, Any]:
        confidences = [face.confidence for face in self.detections]
        data: dict[str, Any] = {
            "face_count": self.face_count,
            "face_area": self.face_area,
            "confidence": confidences,
            "coordinates": [list(face.bbox) for face in self.detections],
        }

        labels = [face.label for face in self.detections]
        if any(label is not None for label in labels):
            data["labels"] = labels

        distances = [face.embedding_distance for face in self.detections]
        if any(distance is not None for distance in distances):
            data["embedding_distance"] = distances

        if include_average_confidence:
            data["average_confidence"] = float(np.mean(confidences)) if confidences else 0.0

        data.update(self.metadata)
        return data

    def to_records(self, analyzer: str | None = None) -> list[dict[str, Any]]:
        return [
            detection.to_record(self.frame_index, face_index, analyzer=analyzer)
            for face_index, detection in enumerate(self.detections)
        ]

    @classmethod
    def from_dict(cls, data: dict[str, Any], frame_index: int | None = None) -> "FrameAnalysisResult":
        coordinates = data.get("coordinates", []) or []
        confidences = data.get("confidence", []) or []
        labels = data.get("labels", data.get("face_labels", [])) or []
        distances = data.get("embedding_distance", []) or []

        detections: list[FaceDetection] = []
        for index, box in enumerate(coordinates):
            confidence = confidences[index] if index < len(confidences) else 0.0
            label = labels[index] if index < len(labels) else None
            distance = distances[index] if index < len(distances) else None
            detections.append(
                FaceDetection.from_box(
                    box=box,
                    confidence=confidence,
                    label=label,
                    embedding_distance=distance,
                )
            )
        return cls(frame_index=frame_index, detections=detections)


@dataclass(frozen=True)
class ProbeFrame:
    """A ranked face crop candidate from one video frame."""

    frame_index: int
    face_index: int
    bbox: BoxXYWH
    confidence: float
    area: int
    area_ratio: float
    quality_score: float
    label: str | None = None
    analyzer: str | None = None
    rank: int | None = None
    appearance_frame_count: int | None = None

    def to_record(self) -> dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "label": self.label,
            "rank": self.rank,
            "frame": self.frame_index,
            "face_index": self.face_index,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "confidence": self.confidence,
            "area": self.area,
            "area_ratio": self.area_ratio,
            "quality_score": self.quality_score,
            "appearance_frame_count": self.appearance_frame_count,
            "analyzer": self.analyzer,
        }


def detections_from_deepface(frame: np.ndarray, faces: Iterable[dict[str, Any]]) -> list[FaceDetection]:
    detections: list[FaceDetection] = []
    frame_h, frame_w = frame.shape[:2]
    for face in faces:
        area = face.get("facial_area") or face.get("region") or {}
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)
        confidence = float(face.get("confidence", 0.0) or 0.0)

        if int(w) <= 0 or int(h) <= 0:
            continue
        if int(w) == frame_w and int(h) == frame_h:
            continue
        if confidence == 0:
            continue

        detections.append(
            FaceDetection.from_box(
                [x, y, w, h],
                confidence=confidence,
                frame_shape=frame.shape,
                metadata={"source": "deepface"},
            )
        )
    return detections


class FaceIdentityTracker:
    """
    Tracks face identities in a video.
    """
    def __init__(
        self,
        model_name: str = "Facenet512",
        max_distance: float = 0.90,
        label_prefix: str = "face",
        max_samples_per_label: int = 4,
        embedding_function: EmbeddingFunction | None = None,
        crop_padding: int = 0,
    ):
        self.model_name = model_name
        self.max_distance = float(max_distance)
        self.label_prefix = label_prefix
        self.max_samples_per_label = int(max_samples_per_label)
        self.embedding_function = embedding_function
        self.crop_padding = int(crop_padding)
        self.gallery: dict[str, dict[str, Any]] = {}
        self.records: list[dict[str, Any]] = []
        self._next_label_id = 1

    def reset(self) -> None:
        self.gallery.clear()
        self.records.clear()
        self._next_label_id = 1

    def update_frame(
        self,
        frame_index: int,
        frame: np.ndarray,
        detections: list[FaceDetection],
        analyzer: str | None = None,
    ) -> list[str | None]:
        labels: list[str | None] = [None] * len(detections)
        face_images: list[np.ndarray | None] = [None] * len(detections)
        embeddings: list[np.ndarray | None] = [None] * len(detections)

        # Embed the complete frame first.  Matching all faces together avoids
        # assigning one gallery identity twice and is independent of detector order.
        for face_index, detection in enumerate(detections):
            face_img = crop_box_xywh(frame, detection.bbox, padding=self.crop_padding)
            face_images[face_index] = face_img
            try:
                embeddings[face_index] = self._embedding(face_img)
            except Exception as exc:
                detection.label = None
                detection.metadata["embedding_error"] = str(exc)

        assignments: dict[int, tuple[str, float]] = {}
        candidate_matches: list[tuple[float, int, str]] = []
        for face_index, embedding in enumerate(embeddings):
            if embedding is None:
                continue
            for label, info in self.gallery.items():
                distance = float(np.linalg.norm(embedding - info["rep"]))
                if distance <= self.max_distance:
                    candidate_matches.append((distance, face_index, label))

        used_faces: set[int] = set()
        used_labels: set[str] = set()
        for distance, face_index, label in sorted(candidate_matches):
            if face_index in used_faces or label in used_labels:
                continue
            assignments[face_index] = (label, distance)
            used_faces.add(face_index)
            used_labels.add(label)

        for face_index, (detection, embedding, face_img) in enumerate(
            zip(detections, embeddings, face_images)
        ):
            if embedding is None or face_img is None:
                continue
            if face_index in assignments:
                label, distance = assignments[face_index]
            else:
                label = self._create_identity(embedding)
                distance = None
            detection.label = label
            detection.embedding_distance = distance
            self._store_sample(label, embedding, frame_index, face_index, detection, face_img, frame.shape)
            labels[face_index] = label

        result = FrameAnalysisResult(frame_index=frame_index, detections=detections)
        self.records.extend(result.to_records(analyzer=analyzer))
        return labels

    def _embedding(self, face_img_bgr: np.ndarray) -> np.ndarray:
        if self.embedding_function is not None:
            return l2_normalize(self.embedding_function(face_img_bgr))

        from deepface import DeepFace

        face_rgb = cv.cvtColor(face_img_bgr, cv.COLOR_BGR2RGB)
        reps = DeepFace.represent(
            img_path=face_rgb,
            model_name=self.model_name,
            enforce_detection=False,
        )
        return l2_normalize(np.asarray(reps[0]["embedding"], dtype=np.float32))

    def _match_or_create(self, embedding: np.ndarray, used_labels: set[str]) -> tuple[str, float | None]:
        best_label: str | None = None
        best_distance = float("inf")

        for label, info in self.gallery.items():
            if label in used_labels:
                continue
            distance = float(np.linalg.norm(embedding - info["rep"]))
            if distance < best_distance:
                best_distance = distance
                best_label = label

        if best_label is not None and best_distance <= self.max_distance:
            return best_label, best_distance

        return self._create_identity(embedding), None

    def _create_identity(self, embedding: np.ndarray) -> str:
        label = f"{self.label_prefix}{self._next_label_id}"
        self._next_label_id += 1
        self.gallery[label] = {
            "rep": embedding,
            "samples": [],
            "best_samples": [],
            "thumb": [],
            "count": 0,
            "first_seen_frame": None,
            "last_seen_frame": None,
        }
        return label

    def _store_sample(
        self,
        label: str,
        embedding: np.ndarray,
        frame_index: int,
        face_index: int,
        detection: FaceDetection,
        face_img: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> None:
        info = self.gallery[label]
        count = int(info.get("count", 0))
        info["rep"] = l2_normalize((info["rep"] * count + embedding) / (count + 1))
        info["count"] = count + 1
        info["first_seen_frame"] = (
            frame_index if info.get("first_seen_frame") is None
            else min(int(info["first_seen_frame"]), frame_index)
        )
        info["last_seen_frame"] = (
            frame_index if info.get("last_seen_frame") is None
            else max(int(info["last_seen_frame"]), frame_index)
        )
        info["samples"].append((frame_index, tuple(map(int, detection.bbox)), face_index))

        if not face_img.size or self.max_samples_per_label <= 0:
            return

        frame_area = int(frame_shape[0]) * int(frame_shape[1])
        sample = {
            "frame": int(frame_index),
            "face_index": int(face_index),
            "bbox": tuple(map(int, detection.bbox)),
            "confidence": float(detection.confidence),
            "area": int(detection.area),
            "area_ratio": float(detection.area / frame_area) if frame_area else 0.0,
            "quality_score": detection.quality_score(frame_area),
            "image": face_img.copy(),
        }
        best_samples = info.setdefault("best_samples", [])
        best_samples.append(sample)
        best_samples.sort(
            key=lambda item: (
                item["quality_score"], item["confidence"], item["area"], -item["frame"]
            ),
            reverse=True,
        )
        del best_samples[self.max_samples_per_label:]
        info["thumb"] = [cv.resize(item["image"], (112, 112)) for item in best_samples]

