# Video face detection and gallery review

Install the vision dependencies before using the video API:

```bash
python -m pip install "pyWitnessAI[vision]"
```

The video modules use OpenCV, DeepFace and the face-detection stack. Detection
and embedding models may download weights on first use. Import the required
classes explicitly; the base installation supports text and catalog workflows.

## Detect faces

`FrameAnalyzerDeepface` runs the selected DeepFace detector and records face
counts, confidence values and `(x, y, width, height)` coordinates per frame.

```python
from pyWitnessAI import Video, FrameAnalyzerDeepface

video = Video("input.mp4", save_directory="results")
video.add_analyzer(FrameAnalyzerDeepface(detector_backend="mtcnn"))
video.run(frame_start=0, frame_end=999)

print(video.get_analysis_info())
video.save_data(directory="results")
video.plot_face_counts()
video.plot_confidence_vs_frame()
```

Use an integer `frame_end`; processing also stops at the end of the file.
For gallery construction, analyze from frame zero so the stored detections
remain aligned with the gallery's second pass through the video.

## Build and review a face gallery

After detection, `build_face_gallery` reopens the video, extracts embeddings
from the recorded boxes and groups similar faces into labels such as `face1`.
The distance threshold controls this grouping and should be checked on the
target footage.

```python
gallery = video.build_face_gallery(
    detector="mtcnn",
    model_name="Facenet512",
    max_distance=0.90,
)
print(video.list_face_labels())
if gallery:
    video.save_face_gallery("results/gallery")
    video.show_gallery_contact_sheet(save_path="results/gallery.jpg")
```

The saved gallery contains sample thumbnails, `gallery_index.csv` with sample
counts and first-seen frames, and `gallery_meta.csv` with detector/model details.
Samples are taken in observation order, without quality ranking.

`video.filter_faces(detector="mtcnn", keep=["face1"])` filters the stored
analysis records in place after gallery construction. It does not write a
masked video. The current API does not provide `FrameAnalyzerFaceIdentity`,
`get_mugbook`, `save_face_filtered_video`, or an identity-level quality dashboard.

## Review probe frames

```python
top_frames = video.find_probe_frames(
    top_n=3,
    detector="mtcnn",
    method="confidence",  # Or "metrics": face area, confidence and a frontal heuristic
)
video.save_probe_frames(top_frames)
```

This selects and saves whole frames. Confidence ranking uses the first detected
face in each frame; it does not return per-identity face crops. For distances
against a specified lineup, use [VideoLineupPipeline](VideoLineupPipeline.md).
