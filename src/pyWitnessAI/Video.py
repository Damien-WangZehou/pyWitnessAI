import inspect
import os
import time
import heapq
from dataclasses import replace

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyWitnessAI.utils.Constants import legend_colors, get_color_for_analyzer, get_style_for_analyzer
from pyWitnessAI.utils.DataFlattener import flatten_data, flatten_keys
from .ImagesCategorizer import *
from .VideoAnalysis import (
    FaceIdentityTracker,
    FrameAnalysisResult,
    ProbeFrame,
    crop_box_xywh,
)
# You should also load the path of cascade, similarity_model, lineup_images before using the analyzer

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Video:
    def __init__(self, video_path, save_directory='Video analysis results'):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.frame_count = []
        self.frame_width = None
        self.frame_height = None
        self.frame_area = None
        self.fps = 0.0
        self.average_pixel_values = []
        self.average_value = 0  # This is for the mean of the average pixel values of the whole video

        self.frame_processor = {}
        self.frame_analyzer = {}
        self.frame_analyzer_output = {}
        self.frame_analyzed = 0
        self.frame_total = 0
        self.face_records = []

        self.save_directory = save_directory
        self.top_frames = None  # An attribute to get the best quality frame
        self.top_probe_frames = []
        self.find_probe_frames_detector = None
        self.find_probe_frames_method = None

        self.face_gallery = {}  # {label: {"rep": np.ndarray, "samples": [(frame_idx, (x,y,w,h))], "thumb": [np.ndarray,...]}}
        self.face_labels_by_frame = []  # Corresponding to frame_count, list of lists of labels per frame, in order
        self._gallery_built_from = None  # Model used to build the gallery (e.g., 'mtcnn')
        self._gallery_model_name = None  # Model name used for embeddings (e.g., 'Facenet512')

    def add_analyzer(self, analyzer):
        #  Add an external frame analyzer
        self.frame_analyzer[analyzer.name] = analyzer
        self.frame_analyzer_output[analyzer.name] = []

    def add_processor(self, processor):
        self.frame_processor[processor.name] = processor

    def release_resources(self):
        self.cap.release()
        for processor in self.frame_processor.values():
            if hasattr(processor, 'release'):
                processor.release()

    def get_frame_info(self):
        #  Retrieve frame information
        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_area = self.frame_width * self.frame_height
        self.fps = float(self.cap.get(cv.CAP_PROP_FPS) or 0.0)

    def _reset_run_state(self):
        # Reset the run state so that the video can be processed again
        self.frame_count = []
        self.average_pixel_values = []
        self.average_value = 0
        self.frame_analyzed = 0
        self.face_records = []
        self.top_frames = None
        self.top_probe_frames = []
        self.face_gallery = {}
        self.face_labels_by_frame = []
        self._gallery_built_from = None
        self._gallery_model_name = None
        for analyzer_name in self.frame_analyzer:
            self.frame_analyzer_output[analyzer_name] = []
        for analyzer in self.frame_analyzer.values():
            if hasattr(analyzer, "reset"):
                analyzer.reset()
        for processor in self.frame_processor.values():
            if hasattr(processor, "reset"):
                processor.reset()

    @staticmethod
    def _accepts_context(callable_obj, parameter_name):
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return (
            parameter_name in signature.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        )

    def _process_frame_with_context(self, processor, frame, context):
        method = processor.process_frame
        if self._accepts_context(method, "context"):
            return method(frame, context=context)
        return method(frame)

    def _analyze_frame_with_context(self, analyzer, frame, frame_index):
        method = analyzer.analyze_frame
        if self._accepts_context(method, "frame_index"):
            return method(frame, frame_index=frame_index)
        return method(frame)

    def _processors_for_stage(self, stage):
        return [
            processor
            for processor in self.frame_processor.values()
            if getattr(processor, "stage", "pre") == stage
        ]

    def _append_face_records(self, frame_index, analyzer_name, result):
        if not isinstance(result, dict) or "coordinates" not in result:
            return
        frame_result = FrameAnalysisResult.from_dict(result, frame_index=frame_index)
        self.face_records.extend(frame_result.to_records(analyzer=analyzer_name))

    def rebuild_face_records(self):
        self.face_records = []
        for analyzer_name, results in self.frame_analyzer_output.items():
            for index, result in enumerate(results):
                if index < len(self.frame_count):
                    self._append_face_records(self.frame_count[index], analyzer_name, result)
        return self.face_records

    def _sync_identity_gallery(self):
        """Expose a streaming identity analyzer's gallery through ``Video``."""
        for analyzer_name, analyzer in self.frame_analyzer.items():
            tracker = getattr(analyzer, "tracker", None)
            if tracker is None:
                continue
            self.face_gallery = tracker.gallery
            self.face_labels_by_frame = [
                list((result or {}).get("labels", []))
                for result in self.frame_analyzer_output.get(analyzer_name, [])
            ]
            self._gallery_built_from = analyzer_name
            self._gallery_model_name = tracker.model_name
            return

    def process_video(self, frame_start=0, frame_end=1000000000):
        #  Process the video frame between frame_start and frame_end
        self._reset_run_state()
        self.cap.release()
        self.cap = cv.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.frame_total = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)) or 0
        self.get_frame_info()

        frame_start = max(0, int(frame_start))
        if frame_end is None:
            frame_end = self.frame_total - 1 if self.frame_total else 1000000000
        frame_end = max(-1, int(frame_end))
        if self.frame_total:
            frame_end = min(frame_end, self.frame_total - 1)
        if frame_start > frame_end >= 0:
            self.release_resources()
            return

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_start)
        frame_analyzed = 0

        # Initialize timing dictionary
        analyzer_timings = {analyzer: 0.0 for analyzer in self.frame_analyzer}

        try:
            for frame_count in range(frame_start, frame_end + 1):
                ret, frame = self.cap.read()

                if not ret:
                    break

                self.frame_count.append(frame_count)
                average_pixel_value = int(frame.mean())
                self.average_pixel_values.append(average_pixel_value)

                context = {
                    "frame": frame_count,
                    "frame_index": frame_count,
                    "video": self,
                    "fps": self.fps,
                    "analysis": {},
                }

                for processor in self._processors_for_stage("pre"):
                    frame = self._process_frame_with_context(processor, frame, context)

                for analyzer_name, analyzer in self.frame_analyzer.items():
                    start_time = time.time()
                    result = self._analyze_frame_with_context(analyzer, frame, frame_count)
                    self.frame_analyzer_output[analyzer_name].append(result)
                    context["analysis"][analyzer_name] = result
                    self._append_face_records(frame_count, analyzer_name, result)
                    analyzer_timings[analyzer_name] += time.time() - start_time

                for processor in self._processors_for_stage("post"):
                    frame = self._process_frame_with_context(processor, frame, context)

                frame_analyzed += 1
        finally:
            self.average_value = float(np.mean(self.average_pixel_values)) if self.average_pixel_values else 0
            self.frame_analyzed = frame_analyzed
            self._sync_identity_gallery()
            self.release_resources()
            try:
                cv.destroyAllWindows()
            except cv.error:
                pass

        # Print timing results
        for analyzer_name, total_time in analyzer_timings.items():
            print(f"Total time for {analyzer_name}: {total_time:.2f} seconds")

    def get_analysis_info(self):
        #  Get the number of analyzed frame and total frames
        return {
            'frame_analyzed': self.frame_analyzed,
            'frame_total': self.frame_total
        }

    def run(self, frame_start=0, frame_end=100000):
        self.process_video(frame_start, frame_end)

    def build_face_gallery(self, detector='mtcnn', model_name='Facenet512',
                           max_distance=0.90, save_dir=None, max_samples_per_label=4,
                           embedding_function=None, crop_padding=0):
        """
        Identify and cluster faces across all frames in the video. Label them as face1, face2, etc.

        Parameters
        ----------
        detector: Choose which detector's output to use for cropping faces
        model_name: DeepFace embedding
        max_distance: The maximum distance threshold to consider two faces as the same person
        save_dir: optional path to save the face gallery
        max_samples_per_label: Each label will store up to this many face thumbnails
        """
        analyzer = self.frame_analyzer.get(detector)
        if analyzer is not None and getattr(analyzer, "tracker", None) is not None:
            self._sync_identity_gallery()
            if save_dir:
                self.save_mugbook(save_dir)
            return self.face_gallery

        if detector not in self.frame_analyzer_output:
            raise ValueError(f"Detector '{detector}' analyzer is not added or has no output.")

        outputs = self.frame_analyzer_output[detector]
        if len(outputs) != len(self.frame_count):
            raise RuntimeError("Analyzer output length mismatch with frame_count. Did you run process_video()?")

        tracker = FaceIdentityTracker(
            model_name=model_name,
            max_distance=max_distance,
            max_samples_per_label=max_samples_per_label,
            embedding_function=embedding_function,
            crop_padding=crop_padding,
        )
        self.face_labels_by_frame = [[] for _ in self.frame_count]
        self._gallery_built_from = detector
        self._gallery_model_name = model_name

        # Reopen video to read frames
        cap2 = cv.VideoCapture(self.video_path)
        if not cap2.isOpened():
            raise RuntimeError(f"Cannot reopen video: {self.video_path}")

        next_frame = None
        try:
            for output_index, frame_number in enumerate(self.frame_count):
                frame_number = int(frame_number)
                if next_frame != frame_number:
                    cap2.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap2.read()
                if not ret:
                    break
                next_frame = frame_number + 1

                frame_result = FrameAnalysisResult.from_dict(
                    outputs[output_index],
                    frame_index=frame_number,
                )
                if not frame_result.detections:
                    self.face_labels_by_frame[output_index] = []
                    continue

                labels = tracker.update_frame(
                    frame_index=frame_number,
                    frame=frame,
                    detections=frame_result.detections,
                    analyzer=f"{detector}_identity",
                )
                self.face_labels_by_frame[output_index] = labels
                updated = frame_result.to_dict(
                    include_average_confidence='average_confidence' in outputs[output_index]
                )
                outputs[output_index].update(updated)
        finally:
            cap2.release()
        self.face_gallery = tracker.gallery
        self.rebuild_face_records()

        # Optionally save the face gallery
        if save_dir:
            self.save_mugbook(save_dir)

        return self.face_gallery

    def list_face_labels(self):
        # List all identified face labels in the gallery
        return list(self.face_gallery.keys())

    def get_mugbook(self):
        """Return one summary row per numbered identity in the video."""
        rows = []
        for label, info in self.face_gallery.items():
            samples = info.get("samples", [])
            frames = sorted({int(sample[0]) for sample in samples})
            best = (info.get("best_samples") or [None])[0]
            rows.append({
                "label": label,
                "appearance_frame_count": len(frames),
                "detection_count": int(info.get("count", len(samples))),
                "first_seen_frame": info.get("first_seen_frame", frames[0] if frames else None),
                "last_seen_frame": info.get("last_seen_frame", frames[-1] if frames else None),
                "best_frame": best.get("frame") if best else (frames[0] if frames else None),
                "best_face_index": best.get("face_index") if best else None,
                "best_confidence": best.get("confidence") if best else None,
                "best_area": best.get("area") if best else None,
                "best_quality_score": best.get("quality_score") if best else None,
            })
        if not rows:
            return pd.DataFrame(columns=[
                "label", "appearance_frame_count", "detection_count",
                "first_seen_frame", "last_seen_frame", "best_frame",
                "best_face_index", "best_confidence", "best_area",
                "best_quality_score",
            ])
        return pd.DataFrame(rows).sort_values(
            by=["first_seen_frame", "label"], na_position="last"
        ).reset_index(drop=True)

    def save_face_gallery(self, save_dir, top_k=None):
        """
        Save the face gallery to the specified directory (label, samples_count, first_seen_frame).
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save the highest-quality crops, rather than whichever detections came first.
        for lbl, info in self.face_gallery.items():
            best_samples = info.get("best_samples", [])
            if top_k is not None:
                best_samples = best_samples[:max(0, int(top_k))]
            if best_samples:
                for i, sample in enumerate(best_samples):
                    path = os.path.join(
                        save_dir,
                        f"{lbl}_rank{i + 1}_frame{sample['frame']}.jpg",
                    )
                    cv.imwrite(path, sample["image"])
            else:
                thumbs = info.get("thumb", [])
                if top_k is not None:
                    thumbs = thumbs[:max(0, int(top_k))]
                for i, thumb in enumerate(thumbs):
                    cv.imwrite(os.path.join(save_dir, f"{lbl}_rank{i + 1}.jpg"), thumb)

        # Save index CSV
        df = self.get_mugbook()
        df.to_csv(os.path.join(save_dir, "gallery_index.csv"), index=False)

        # Record metadata
        meta = {
            "built_from_detector": self._gallery_built_from,
            "embedding_model": self._gallery_model_name,
            "total_labels": len(self.face_gallery)
        }
        pd.DataFrame([meta]).to_csv(os.path.join(save_dir, "gallery_meta.csv"), index=False)
        print(f"Face gallery saved to: {save_dir}")
        return df

    def save_mugbook(self, save_dir, top_k=None, contact_sheet=True):
        """Persist the Mugbook index, its best crops, and an optional contact sheet."""
        if not self.face_gallery:
            raise RuntimeError("Mugbook is empty. Run an identity analysis first.")
        index = self.save_face_gallery(save_dir, top_k=top_k)
        if contact_sheet:
            self.show_gallery_contact_sheet(
                save_path=os.path.join(save_dir, "mugbook_contact_sheet.jpg")
            )
        return index

    def filter_faces(self, detector=None, keep=None, remove=None):
        """
        Filter faces based on their labels in the face_gallery from the analyzer output (in-place modification).

        Parameters
        ----------
        detector: Which detector's output to filter (default is the one used to build the gallery)
        keep: Only keep these labels (list/tuple/set)
        remove: Remove these labels (list/tuple/set)
        """
        if detector is None:
            detector = self._gallery_built_from
        if detector not in self.frame_analyzer_output:
            raise ValueError(f"Detector '{detector}' analyzer is not added or has no output.")
        if not self.face_labels_by_frame:
            raise RuntimeError("No face_labels_by_frame. Did you run build_face_gallery()?")

        if keep is not None and remove is not None:
            raise ValueError("keep and remove cannot be used together.")

        keep_set = set(keep) if keep is not None else None
        remove_set = set(remove) if remove is not None else None

        outputs = self.frame_analyzer_output[detector]
        for i in range(len(outputs)):
            data = outputs[i]
            coords = data.get('coordinates', [])
            confs = data.get('confidence', [])
            distances = data.get('embedding_distance', [])

            labels_this_frame = self.face_labels_by_frame[i] if i < len(self.face_labels_by_frame) else []
            if not coords or not labels_this_frame:
                empty_data = {
                    'face_count': 0,
                    'face_area': 0,
                    'confidence': [],
                    'coordinates': [],
                    'labels': [],
                    'embedding_distance': [],
                }
                if 'average_confidence' in data:
                    empty_data['average_confidence'] = 0
                outputs[i] = empty_data
                continue

            # The boolean mask to decide which faces to keep
            keep_mask = []
            for lbl in labels_this_frame:
                if lbl is None:
                    keep_mask.append(False)
                elif keep_set is not None:
                    keep_mask.append(lbl in keep_set)
                elif remove_set is not None:
                    keep_mask.append(lbl not in remove_set)
                else:
                    keep_mask.append(True)  # If neither keep nor remove is specified, keep all

            # Filter coordinates and confidences
            new_coords = [c for c, k in zip(coords, keep_mask) if k]
            new_confs = [c for c, k in zip(confs, keep_mask) if k]
            new_labels = [c for c, k in zip(labels_this_frame, keep_mask) if k]
            new_distances = [c for c, k in zip(distances, keep_mask) if k]

            # Calculate new face area and average confidence
            new_face_area = sum(int(b[2]) * int(b[3]) for b in new_coords) if new_coords else 0
            if 'average_confidence' in data:
                avg_conf = float(np.mean(new_confs)) if len(new_confs) > 0 else 0
                new_data = {
                    'face_count': len(new_coords),
                    'face_area': new_face_area,
                    'coordinates': new_coords,
                    'confidence': new_confs,
                    'labels': new_labels,
                    'embedding_distance': new_distances,
                    'average_confidence': avg_conf
                }
            else:
                new_data = {
                    'face_count': len(new_coords),
                    'face_area': new_face_area,
                    'coordinates': new_coords,
                    'confidence': new_confs,
                    'labels': new_labels,
                    'embedding_distance': new_distances
                }
            outputs[i] = new_data

        self.rebuild_face_records()
        print(f"Filtering done on detector '{detector}'.")

    def get_face_table(self, labeled_only=False):
        """
        Return one row per detected face across analyzed frames.
        """
        if not self.face_records:
            self.rebuild_face_records()
        df = pd.DataFrame(self.face_records)
        if labeled_only and not df.empty and 'label' in df.columns:
            df = df[df['label'].notna()].reset_index(drop=True)
        return df

    def save_face_table(self, directory='results', prefix='faces', labeled_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
        df = self.get_face_table(labeled_only=labeled_only)
        path = os.path.join(directory, f'{prefix}_data.csv')
        df.to_csv(path, index=False)
        return path

    def save_face_filtered_video(self, output_path, keep=None, remove=None,
                                 mask_style='gray', detector=None, padding=0,
                                 frame_start=None, frame_end=None, fps=None):
        """Render a second-pass video while masking selected Mugbook identities."""
        if keep is not None and remove is not None:
            raise ValueError("keep and remove cannot be used together.")
        detector = detector or self._gallery_built_from
        if detector is None:
            raise RuntimeError("No identity analyzer is available. Build the Mugbook first.")

        face_table = self.get_face_table(labeled_only=False)
        if face_table.empty or "label" not in face_table:
            raise RuntimeError("No labeled face records are available. Build the Mugbook first.")
        if "analyzer" in face_table:
            face_table = face_table[face_table["analyzer"] == detector].copy()
        if face_table.empty:
            raise RuntimeError(f"No face records are available for analyzer '{detector}'.")

        known_labels = set(face_table["label"].dropna())
        requested = set(keep or remove or [])
        unknown = requested - known_labels
        if unknown:
            raise ValueError(f"Unknown Mugbook labels: {sorted(unknown)}")

        from .VideoProcessor import FrameProcessorFaceMasker, FrameProcessorVideoWriter

        render = Video(self.video_path, save_directory=self.save_directory)
        render.add_processor(FrameProcessorFaceMasker(
            keep=keep,
            remove=remove,
            face_table=face_table,
            analyzer_name=detector,
            mask_style=mask_style,
            padding=padding,
        ))
        render.add_processor(FrameProcessorVideoWriter(output_path, fps=fps))
        if frame_start is None:
            frame_start = min(self.frame_count) if self.frame_count else 0
        if frame_end is None:
            frame_end = max(self.frame_count) if self.frame_count else None
        render.process_video(frame_start=frame_start, frame_end=frame_end)
        return os.path.abspath(output_path)

    def show_gallery_contact_sheet(self, save_path=None, cols=8, thumb_size=(112, 112), show_window=False, window_name='Face Gallery'):
        """
        Build contact sheet of the face gallery.

        Parameters
        ----------
        save_path: If provided, save the contact sheet image to this path
        cols: Number of columns in the contact sheet
        thumb_size: Size of each thumbnail (width, height).
        show_window: If True, show the contact sheet using OpenCV window
        """
        if not self.face_gallery:
            raise RuntimeError("face_gallery is empty. Please run build_face_gallery() first.")

        thumbs = []
        labels = []
        for lbl, info in self.face_gallery.items():
            # Take the first thumbnail available
            if info.get("thumb"):
                thumbs.append(info["thumb"][0])
                labels.append(lbl)

        if not thumbs:
            print("No thumbnails available in the gallery.")
            return None

        # Uniformly resize thumbnails and add labels
        th, tw = thumb_size[1], thumb_size[0]
        norm_thumbs = []
        for img in thumbs:
            t = cv.resize(img, (tw, th))
            # Label bar
            bar_h = 18
            canvas = np.full((th + bar_h, tw, 3), 245, dtype=np.uint8)
            canvas[:th, :, :] = t
            cv.putText(canvas, labels[len(norm_thumbs)], (4, th + 14), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv.LINE_AA)
            norm_thumbs.append(canvas)

        rows = int(np.ceil(len(norm_thumbs) / float(cols)))
        cell_h, cell_w = norm_thumbs[0].shape[:2]
        sheet = np.full((rows * cell_h, cols * cell_w, 3), 255, dtype=np.uint8)

        for idx, img in enumerate(norm_thumbs):
            r = idx // cols
            c = idx % cols
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            sheet[y1:y2, x1:x2, :] = img

        if save_path:
            cv.imwrite(save_path, sheet)
            print(f"Contact sheet saved to: {save_path}")

        if show_window:
            cv.imshow(window_name, sheet)
            cv.waitKey(0)
            cv.destroyWindow(window_name)

        return sheet

    def _probe_analyzer_name(self, detector=None):
        if detector is not None:
            if detector not in self.frame_analyzer_output:
                raise ValueError(f"Analyzer '{detector}' is not available.")
            return detector
        if self._gallery_built_from in self.frame_analyzer_output:
            return self._gallery_built_from
        for name, outputs in self.frame_analyzer_output.items():
            if any(isinstance(item, dict) and item.get("labels") for item in outputs):
                return name
        for name, outputs in self.frame_analyzer_output.items():
            if any(isinstance(item, dict) and "coordinates" in item for item in outputs):
                return name
        raise RuntimeError("No face analyzer output is available. Run process_video() first.")

    def find_top_probe_frames(self, top_k=3, detector=None, labels=None,
                              per_identity=True, min_frame_gap=0):
        """
        Rank clear face crops using detection confidence multiplied by face area.

        By default, ``top_k`` candidates are returned for every Mugbook identity.
        Set ``per_identity=False`` to request the best candidates in the whole video.
        """
        top_k = int(top_k)
        min_frame_gap = max(0, int(min_frame_gap))
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        analyzer_name = self._probe_analyzer_name(detector)
        outputs = self.frame_analyzer_output[analyzer_name]
        if len(outputs) != len(self.frame_count):
            raise RuntimeError("Analyzer output length does not match the analyzed frames.")

        wanted = set(labels) if labels is not None else None
        candidates = []
        frames_by_label = {}
        frame_area = int(self.frame_area or 0)
        for output_index, data in enumerate(outputs):
            if not isinstance(data, dict):
                continue
            frame_number = int(self.frame_count[output_index])
            result = FrameAnalysisResult.from_dict(data, frame_index=frame_number)
            for face_index, detection in enumerate(result.detections):
                label = detection.label
                if wanted is not None and label not in wanted:
                    continue
                group = label if per_identity else "__all__"
                frames_by_label.setdefault(group, set()).add(frame_number)
                area_ratio = float(detection.area / frame_area) if frame_area else 0.0
                candidates.append(ProbeFrame(
                    frame_index=frame_number,
                    face_index=face_index,
                    bbox=detection.bbox,
                    confidence=float(detection.confidence),
                    area=int(detection.area),
                    area_ratio=area_ratio,
                    quality_score=detection.quality_score(frame_area or None),
                    label=label,
                    analyzer=analyzer_name,
                ))

        if wanted is not None:
            available = {candidate.label for candidate in candidates}
            missing = wanted - available
            if missing:
                raise ValueError(f"Unknown or undetected Mugbook labels: {sorted(missing)}")

        grouped = {}
        for candidate in candidates:
            group = candidate.label if per_identity else "__all__"
            grouped.setdefault(group, []).append(candidate)

        selected = []
        for group, group_candidates in grouped.items():
            ordered = sorted(
                group_candidates,
                key=lambda item: (
                    item.quality_score, item.confidence, item.area, -item.frame_index
                ),
                reverse=True,
            )
            group_selected = []
            for candidate in ordered:
                if min_frame_gap and any(
                    abs(candidate.frame_index - existing.frame_index) < min_frame_gap
                    for existing in group_selected
                ):
                    continue
                group_selected.append(candidate)
                if len(group_selected) == top_k:
                    break
            appearance_count = len(frames_by_label.get(group, set()))
            selected.extend(
                replace(
                    candidate,
                    rank=rank,
                    appearance_frame_count=appearance_count,
                )
                for rank, candidate in enumerate(group_selected, start=1)
            )

        self.top_probe_frames = sorted(
            selected,
            key=lambda item: (
                "" if item.label is None else str(item.label),
                item.rank or 0,
            ),
        )
        records = [candidate.to_record() for candidate in self.top_probe_frames]
        return pd.DataFrame(records, columns=[
            "label", "rank", "frame", "face_index",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "confidence", "area", "area_ratio", "quality_score",
            "appearance_frame_count", "analyzer",
        ])

    def save_top_probe_faces(self, probes=None, save_dir=None):
        """Save the cropped faces represented by ``find_top_probe_frames``."""
        if probes is None:
            if not self.top_probe_frames:
                raise RuntimeError("Run find_top_probe_frames() first.")
            probes = pd.DataFrame([probe.to_record() for probe in self.top_probe_frames])
        elif not isinstance(probes, pd.DataFrame):
            probes = pd.DataFrame(probes)
        probes = probes.copy()
        if probes.empty:
            return probes

        save_dir = save_dir or os.path.join(self.save_directory, "probe_faces")
        os.makedirs(save_dir, exist_ok=True)
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot reopen video: {self.video_path}")

        image_paths = {}
        try:
            for row_index, row in probes.sort_values("frame").iterrows():
                frame_number = int(row["frame"])
                cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
                ok, frame = cap.read()
                if not ok:
                    continue
                bbox = [row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"]]
                crop = crop_box_xywh(frame, bbox)
                label = row.get("label")
                label = "face" if pd.isna(label) else str(label)
                safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
                path = os.path.abspath(os.path.join(
                    save_dir,
                    f"{safe_label}_rank{int(row['rank'])}_frame{frame_number}.jpg",
                ))
                if crop.size and cv.imwrite(path, crop):
                    image_paths[row_index] = path
        finally:
            cap.release()

        probes["image_path"] = [image_paths.get(index) for index in probes.index]
        probes.to_csv(os.path.join(save_dir, "probe_faces.csv"), index=False)
        return probes

    def find_probe_frames(self, top_n=1, log_file='probe_frames_log.txt', detector='mtcnn', method='confidence'):
        save_directory = f'{self.save_directory}/probe_frames/{detector}_{method}_probe_frames'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.find_probe_frames_detector = detector
        self.find_probe_frames_method = method
        frames_metric = []
        frames_confidence = []
        log_file = f'{save_directory}/{detector}_{method}_{log_file}'

        if detector in self.frame_analyzer_output:
            analyzer_output = self.frame_analyzer_output[detector]

            for i, frame_data in enumerate(analyzer_output):
                confidences = frame_data.get('confidence', []) or []
                coords = frame_data.get('coordinates', []) or []
                for face_index, (x, y, w, h) in enumerate(coords):
                    confidence = float(confidences[face_index]) if face_index < len(confidences) else 0.0
                    face_area = int(w) * int(h)
                    center_x = x + w / 2
                    aspect_ratio = w / h if h > 0 else 0
                    frontal_score = 0
                    if 0.75 < aspect_ratio < 1.33 and abs(center_x - self.frame_width / 2) < 0.25 * self.frame_width:
                        frontal_score = 1
                    metric = face_area * confidence * (1 + 0.5 * frontal_score)
                    frames_metric.append((metric, self.frame_count[i], face_area, confidence, frontal_score))
                    frames_confidence.append((confidence, self.frame_count[i]))

        else:
            print(f"{detector} analyzer is not added.")
            return []

        # Select top frames based on the specified method
        if method == 'confidence':
            top_frames = heapq.nlargest(top_n, frames_confidence, key=lambda x: x[0])
        elif method == 'metrics':
            top_frames = heapq.nlargest(top_n, frames_metric, key=lambda x: x[0])
        else:
            print(f"Unknown method: {method}. Please use 'confidence' or 'metrics'.")
            return []

        self.top_frames = top_frames

        with open(log_file, 'w') as f:
            for frame in top_frames:
                if method == 'confidence':
                    fst_conf, frame_num = frame
                    log_message = (f"Probe frame at frame number: {frame_num} with confidence score: {fst_conf} "
                                   f"by {detector}\n")
                elif method == 'metrics':
                    metric, frame_num, face_area, fst_conf, frontal_score = frame
                    log_message = (f"Probe frame at frame number: {frame_num} with metric: {metric:.2f} "
                                   f"(face_area: {face_area}, confidence score: {fst_conf:.2f}, frontal_score: {frontal_score}) "
                                   f"by detector {detector}\n")

                print(log_message.strip())
                f.write(log_message)

        return top_frames

    def print_probe_frames(self, top_frames):
        if self.top_frames is None:
            print("Please run find_probe_frames first.")
            return
        for i, frame in enumerate(top_frames):
            frame_number = self._probe_frame_number(frame)
            self.print_frame(frame_number, f"Probe Frame {i+1}")

    def save_probe_frames(self, top_frames):
        save_directory = (f'{self.save_directory}/probe_frames/{self.find_probe_frames_detector}_'
                          f'{self.find_probe_frames_method}_probe_frames')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        for i, frame in enumerate(top_frames):
            frame_number = self._probe_frame_number(frame)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                save_path = os.path.join(save_directory, f'probe_frame_{i+1}.jpg')
                cv.imwrite(save_path, frame)
                print(f"Probe frame {i+1} saved at {save_path}")
            else:
                print(f"Failed to retrieve frame at frame number: {frame_number}")

        self.release_resources()

    @staticmethod
    def _probe_frame_number(probe):
        if isinstance(probe, ProbeFrame):
            return int(probe.frame_index)
        if isinstance(probe, dict):
            return int(probe.get("frame", probe.get("frame_index")))
        if hasattr(probe, "get") and "frame" in probe:
            return int(probe["frame"])
        if len(probe) >= 2:
            return int(probe[1])
        raise ValueError(f"Unsupported probe frame value: {probe!r}")

    def print_frame(self, frame_number, window_name="Frame"):
        # Reinitialize the video capture to ensure frames can be accessed correctly
        self.cap = cv.VideoCapture(self.video_path)

        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            cv.imshow(window_name, frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print(f"Failed to retrieve frame at frame number: {frame_number}")

    def plot_face_counts(self, detector=None, ax=None):
        #  Plots the number of faces against frame numbers
        ax = ax or plt.gca()
        upper_limit = 1
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity" or (detector is not None and k != detector):
                continue
            face_counts = [data.get('face_count', 0) if isinstance(data, dict) else 0 for data in output]
            if face_counts:
                upper_limit = max(upper_limit, max(face_counts))
            ax.step(self.frame_count[:len(face_counts)], face_counts, where='mid', label=k,
                     linestyle=get_style_for_analyzer(k),
                     color=get_color_for_analyzer(k), alpha=0.75)

        ax.set_xlabel('Frame')
        ax.set_ylim(-0.05, upper_limit + 0.5)
        ax.set_ylabel('Number of Faces')
        ax.set_title('Detected Faces per Frame')
        if ax.lines:
            ax.legend()
        ax.grid(True)
        return ax

    def plot_face_areas(self, detector=None, ax=None):
        #  Plots the face area recognized by the classifiers against frame numbers
        ax = ax or plt.gca()
        upper_limit = 0.05
        for k, output in self.frame_analyzer_output.items():
            if k == "similarity" or (detector is not None and k != detector):
                continue
            face_areas = []
            for data in output:
                if isinstance(data, dict) and 'face_area' in data and self.frame_area:
                    face_areas.append(data['face_area'] / self.frame_area)
                else:
                    face_areas.append(0)
            ax.plot(self.frame_count[:len(face_areas)], face_areas, label=k, linestyle=get_style_for_analyzer(k),
                     color=get_color_for_analyzer(k), alpha=0.75)
            if face_areas:
                upper_limit = max(upper_limit, max(face_areas) + 0.05)

        ax.set_xlabel('Frame')
        ax.set_ylim(0, upper_limit)
        ax.set_ylabel('Face Area Ratio')
        ax.set_title('Detected Face Area per Frame')
        if ax.lines:
            ax.legend()
        ax.grid(True)
        return ax

    def plot_average_pixel_values(self, ax=None):
        #  Plot the average pixel values of the video
        if not self.average_pixel_values:
            raise RuntimeError("No video frames have been analyzed.")
        ax = ax or plt.gca()
        ax.plot(self.frame_count, self.average_pixel_values, color=legend_colors['general'])
        ax.axhline(y=self.average_value, color=legend_colors['mean'], linestyle='--', label='Average value')
        ax.set_xlabel('Frame')
        ax.set_ylim(min(self.average_pixel_values)-5, max(self.average_pixel_values)+5)
        ax.set_ylabel('Average pixel value')
        ax.set_title('Pixel Intensity Trend across the Video')
        ax.legend()
        ax.grid(True)
        return ax

    def plot_confidence_vs_frame(self, detector=None, label=None, ax=None, show=False):
        #  Plot the confidence as a function of frame number
        ax = ax or plt.gca()
        for analyzer_name, output in self.frame_analyzer_output.items():
            if detector is not None and analyzer_name != detector:
                continue
            series = {}
            for output_index, data in enumerate(output):
                if not isinstance(data, dict) or output_index >= len(self.frame_count):
                    continue
                result = FrameAnalysisResult.from_dict(data, frame_index=self.frame_count[output_index])
                for detection in result.detections:
                    if label is not None and detection.label != label:
                        continue
                    series_label = detection.label or analyzer_name
                    points = series.setdefault(series_label, ([], []))
                    points[0].append(self.frame_count[output_index])
                    points[1].append(detection.confidence)
            for series_label, (frames, confidences) in series.items():
                ax.plot(frames, confidences, 'o-', label=str(series_label), alpha=0.75, markersize=3)

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Detection Confidence')
        ax.set_title('Face Confidence over Time')
        if ax.lines:
            ax.legend()
        ax.grid(True)
        if show:
            plt.show()
        return ax

    def plot_face_presence(self, detector=None, ax=None):
        """Plot the frames in which every Mugbook identity was detected."""
        analyzer_name = self._probe_analyzer_name(detector)
        ax = ax or plt.gca()
        frames_by_label = {}
        for output_index, data in enumerate(self.frame_analyzer_output[analyzer_name]):
            if not isinstance(data, dict) or output_index >= len(self.frame_count):
                continue
            result = FrameAnalysisResult.from_dict(data, frame_index=self.frame_count[output_index])
            for detection in result.detections:
                label = detection.label or "unlabeled"
                frames_by_label.setdefault(label, set()).add(self.frame_count[output_index])

        labels = sorted(frames_by_label, key=str)
        for row, label in enumerate(labels):
            frames = sorted(frames_by_label[label])
            ax.scatter(frames, [row] * len(frames), marker='s', s=18, label=str(label))
        ax.set_yticks(range(len(labels)), [str(label) for label in labels])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mugbook identity')
        ax.set_title('Identity Presence / Detection Continuity')
        ax.grid(True, axis='x', alpha=0.3)
        return ax

    def plot_video_quality(self, detector=None, label=None, figsize=(12, 10), show=False):
        """Build the face-count, identity-continuity, confidence, and area dashboard."""
        detector = self._probe_analyzer_name(detector)
        figure, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        self.plot_face_counts(detector=detector, ax=axes[0])
        self.plot_face_presence(detector=detector, ax=axes[1])
        self.plot_confidence_vs_frame(detector=detector, label=label, ax=axes[2])
        self.plot_face_areas(detector=detector, ax=axes[3])
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes

    def plot_confidence(self, start_frame=0, end_frame=None):
        if end_frame is None:
            end_frame = int(self.frame_total)  # Ensure frame_total is an integer

        # Filter data for frames within the specified range
        frame_range = range(int(start_frame), min(int(end_frame), len(self.frame_count)))

        # Initialize a plot
        plt.figure(figsize=(14, 7))

        for analyzer_name, output in self.frame_analyzer_output.items():
            confidences = []
            frames = []

            for i in frame_range:
                if i < len(output):
                    frame_data = output[i]
                    if 'confidence' in frame_data and frame_data['confidence']:
                        confidences.append(frame_data['confidence'][0])  # Take the first face's confidence
                        frames.append(self.frame_count[i])
                    else:
                        confidences.append(None)  # Add None for missing confidence data
                        frames.append(self.frame_count[i])

            if confidences:
                plt.plot(frames, confidences, 'o-', label=f'{analyzer_name}_confidence_0')

        plt.xlabel('Frame Number')
        plt.ylabel('Confidence')
        plt.ylim(0.5, 1)
        plt.title('Confidence of Different Analyzers for the Face in Each Frame')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confidence_histogram(self, transparency=0.5):
        """
        Plot the confidence histogram for all analyzers with a specified transparency.
        """
        for analyzer_name, output in self.frame_analyzer_output.items():
            confidences = []
            for data in output:
                if 'confidence' in data:
                    confidences.append(data['confidence'])

            flattened_confidences = []
            for sublist in confidences:
                for item in sublist:
                    flattened_confidences.append(item)

            if flattened_confidences:
                color = get_color_for_analyzer(analyzer_name)
                plt.hist(flattened_confidences, bins=30, alpha=transparency, label=analyzer_name, color=color, edgecolor='k')

        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Histogram for All Analyzers')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_data(self, directory='results', prefix='analyzed'):
        #  Save the analyzed data results to .csv files.
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {
            'frame': self.frame_count,
            'avg_pixel_value': self.average_pixel_values
        }

        for analyzer_name, results_list in self.frame_analyzer_output.items():
            if not results_list:
                continue
            # if all entries are dictionaries
            if all(isinstance(entry, dict) for entry in results_list):
                keys = set().union(*(entry.keys() for entry in results_list))
                for key in sorted(keys):
                    data[f'{analyzer_name}_{key}'] = [entry.get(key) for entry in results_list]
            else:
                # if the results list contains non-dictionary entries (like lists)
                # If the entry is a list, save its length (For simplicity).
                data[f'{analyzer_name}_length'] = [len(entry) if isinstance(entry, list) else None for entry in
                                                   results_list]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)

    def save_data_flattened(self, directory='', prefix='analyzed_flattened'):
        directory = f'{self.save_directory}/{directory}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        #  Initialize data structures for flattened data
        flattened_data = []
        #  Iterate through each frame's analyzed output
        for i, frame in enumerate(self.frame_count):
            #  Initialize a dictionary for each frame
            frame_data = {'frame': frame}

            #  Flatten and merge all analyzer data into frame_data
            for analyzer_name, results_list in self.frame_analyzer_output.items():
                result = results_list[i]  # Get the result for the current frame
                flat_result = flatten_data(result)  # Flatten the result
                flat_keys = flatten_keys(result)  # Flatten the keys

                # Pair flattened keys and values, then merge into frame_data
                for key, value in zip(flat_keys, flat_result):
                    frame_data[f"{analyzer_name}_{key}"] = value

            #  Add the average pixel value for the frame
            frame_data['avg_pixel_value'] = self.average_pixel_values[i]

            #  Append the frame data to the flattened data list
            flattened_data.append(frame_data)

        #  Create a DataFrame from the flattened data
        df = pd.DataFrame(flattened_data)

        #  Define a preferred column order
        preferred_order = ['frame', 'avg_pixel_value']
        analyzer_keys = set()

        #  Collect keys for each analyzer type, assuming they start with the analyzer's name
        for column in df.columns:
            if column.startswith(tuple(self.frame_analyzer.keys())):
                analyzer_keys.add(column.split('_')[0])  # Get the analyzer name prefix

        #  Add analyzer data to preferred order, grouped by analyzer
        for analyzer in analyzer_keys:
            preferred_order.extend([col for col in df.columns if col.startswith(analyzer)])

        #  Ensure all columns are included by adding any remaining columns at the end
        remaining_columns = [col for col in df.columns if col not in preferred_order]
        preferred_order.extend(remaining_columns)
        # preferred_order[2:] = preferred_order[2:][::-1]  # Reverse the results of analyzers

        #  Reorder the DataFrame according to the preferred order
        df = df[preferred_order]

        df.to_csv(os.path.join(directory, f'{prefix}_data.csv'), index=False)

    # ---------- Face Gallery: toolkit ----------
    @staticmethod
    def _crop_face_from_frame(frame, box):
        x, y, w, h = map(int, box)
        h_img, w_img = frame.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        return frame[y:y+h, x:x+w]

    @staticmethod
    def _l2_normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    @staticmethod
    def _euclidean_distance(a, b):
        return np.linalg.norm(a - b)

    def _represent_face(self, face_img_bgr, model_name='Facenet'):
        # DeepFace expects RGB input.
        from deepface import DeepFace

        face_rgb = cv.cvtColor(face_img_bgr, cv.COLOR_BGR2RGB)
        reps = DeepFace.represent(face_rgb, model_name=model_name, enforce_detection=False)
        emb = np.array(reps[0]['embedding'], dtype=np.float32)
        return self._l2_normalize(emb)

    def _update_centroid(self, old, new, count_old):
        # Update centroid with new embedding
        return self._l2_normalize((old * count_old + new) / (count_old + 1))


    













