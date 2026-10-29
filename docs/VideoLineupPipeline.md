# VideoLineupPipeline

`VideoLineupPipeline` compares probes from a video with a `Lineup`, computes
face-embedding distances and attaches `LineupAIDecision` summaries.

## Install and run

```bash
python -m pip install "pyWitnessAI[vision]"
```

The vision extra includes face recognition, detection and plotting dependencies.
The current `ImagesAI` constructor initializes its FaceNet stack even for the
DeepFace workflow, so constructing a pipeline can download pretrained weights.

```python
from pyWitnessAI import LineupBuilder, PipelineConfig, VideoLineupPipeline

lineup = LineupBuilder(
    suspect="/path/to/perp.jpg",
    filler_pool="/path/to/fillers",
    size=6,
    seed=42,
).build()
config = PipelineConfig(
    analyzer_method="process",
    model="Facenet",
    backend="mtcnn",
    distance_metric="cosine",
    use_full_frame=False,
    enforce_detection=False,
    show_progress=True,
)
pipe = VideoLineupPipeline(
    video_path="/path/to/video.mp4",
    lineup=lineup,
    cfg=config,
)
result = pipe.run(frame_start=0, frame_end=99, frame_stride=1)
pipe.save(result, output_csv="results.csv")
pipe.savepywitness(result, "results_pyWitness.csv")
```

Build the lineup first with `LineupBuilder`. To run a target-absent condition,
supply an innocent suspect and `target="absent"`, or pass `suspect=None` to build
an all-filler lineup from `size` pool members. Construct target-present and
target-absent lineups separately.
The pipeline consumes the built members, images and roles without sampling or
reordering. A lineup returned by `lineup.standardize(FaceStandardizer())` uses the
same interface. See [LineupBuilder](LineupBuilder.md) for preparation workflows.

The constructor accepts `lineup=` and `cfg=`; `cfg` may also be a dictionary with
the fields below. The historical `VideoAI.LineupLoader` has a different interface.

## Configuration

| `PipelineConfig` field | Default | Meaning |
| --- | --- | --- |
| `analyzer_method` | `"process"` | `process`, `process_verify`, or `process_georgia_pipeline` |
| `distance_metric` | `"euclidean_l2"` | `euclidean`, `euclidean_l2`, or `cosine` |
| `backend` | `"opencv"` | DeepFace detection backend |
| `model` | `"VGG-Face"` | DeepFace embedding model |
| `align` | `False` | DeepFace alignment |
| `normalization` | `"base"` | DeepFace normalization |
| `enforce_detection` | `False` | DeepFace detection enforcement |
| `show_progress` | `False` | Show progress bars |
| `use_full_frame` | `True` | Use the full frame as the probe instead of extracting faces |
| `no_face_fill` | `50.0` | Distance recorded when there is no usable face |

`process` and `process_verify` use DeepFace. `process_georgia_pipeline` uses
facenet-pytorch MTCNN and its pretrained FaceNet embeddings; the DeepFace
`model` and `backend` settings do not select that embedding model.

With `use_full_frame=False`, face extraction follows the selected workflow.
The pipeline emits a fallback distance and filler-style summary when no usable
face is found. Choose `no_face_fill` with the selected distance scale in mind.

The pipeline constructs its own `LineupAIDecision` and reads the roles recorded
in the lineup. `detect_faces`, `identifier_threshold`, `identifier_targetLineup` and
`identifier_target` are not `PipelineConfig` fields. Its constructor does not
accept a separate identifier.

## Results and exports

`run()` accepts inclusive frame limits; `frame_end=None` processes to the end
of the video. It returns a wide DataFrame with one row per frame/probe,
per-member distance columns, role-minimum columns and TP/TA summary fields.
Distances are lower for more similar faces.

`pipe.save()` writes that DataFrame to CSV. `pipe.savepywitness()` writes
`responseType`, `confidence`, `targetLineup` and `lineupSize`. By default it uses
the built lineup's target condition and actual member count, so a six-member
lineup exports `lineupSize=6`. An explicit `targetLineup` must match the lineup;
run separate lineups for target-present and target-absent exports. A lineup with
`suspect=None` uses target-absent exports and has only `fillerId` or `rejectId`
responses, since no member has a suspect role. The standalone
`export_for_pywitness` helper accepts a DataFrame and an output path. The existing
export convention uses distances as confidence values; these are not calibrated
probabilities.

Optional `pipe.plot_role_histograms(result)` plots the role-distance
distributions. Processing is currently performed per frame.
