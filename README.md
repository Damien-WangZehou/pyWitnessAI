# pyWitnessAI

## Install

Use Python 3.10 or 3.11. The base installation includes NumPy, pandas and Pillow;
description parsing and custom scoring do not require a model installation.

```bash
python -m pip install pyWitnessAI
pyWitnessAI --help
python -m pyWitnessAI --version
pyWitnessAI parse "Short hair; maybe blue eyes" --parser lenient
```

Install the optional features needed by your experiment:

| Feature | Install |
| --- | --- |
| CLIP selection and result plots | `python -m pip install "pyWitnessAI[clip,plot]"` |
| Face recognition and video analysis | `python -m pip install "pyWitnessAI[vision]"` |
| Image generation with OpenAI | `python -m pip install "pyWitnessAI[generation]"` |
| Local portrait standardization | `python -m pip install "pyWitnessAI[standardization]"` |
| Local AI portrait enhancement and standardization | `python -m pip install "pyWitnessAI[enhancement]"` |
| Excel metadata | `python -m pip install "pyWitnessAI[excel]"` |
| General lineup and witness-session HTTP API | `python -m pip install "pyWitnessAI[api]"` |

For an editable checkout, replace `pyWitnessAI[clip,plot]` with `.[clip,plot]`
and add `-e`. Import the names your script uses explicitly; public interfaces
load their owning modules on first access.
Models load when their feature is used and may download weights on first use.

## Local data

`data/` is an ignored local workspace for datasets and generated results. Its
images, CFD norming workbook, generated prompt CSVs and benchmark outputs are
not included in a clone, source distribution or wheel. Supply the image and
metadata paths required by your study through its configuration. Retained
generation scripts and study configurations live in `studies`.

For the controlled attribute probe, prepare the local prompt CSV and per-image
prompt files without calling an image service:

```bash
python studies/generate_controlled_attribute_probe_prompts.py --data-root ./data/controlled_clip_attribute_probe
```

The generated-face benchmark continues to use `data/generated_filler_benchmark`
by default; generate or import its images and manifest before reusing a dataset.
The eight small regression PNGs in
`test/fixtures/georgia_state_video1` are kept with the tests in the repository
and source distribution; they are excluded from the wheel.

## Description-based filler selection

Install `pyWitnessAI[clip,plot]` for this example:

```python
from pyWitnessAI import FillerSelector

selector = FillerSelector("./face_pool")
result = selector.select("Short hair, maybe blue eyes")  # Top 5, raw CLIP cosine
result.show()
```

Choose `parser="raw"`, `"lenient"`, or `"strict"` independently of the model
and `scoring="cosine"` or `"contrast"`. Contrast retains unpaired features as
positive scores and keeps uncertainty markers. See the
[progressive experiment guide](docs/FillerSelection.md) for custom models,
scoring strategies, generation, and migration examples.



## Face generation

For random, description-guided, or reference-image face generation, see the
[FaceGenerator guide](docs/FaceGenerator.md). `FaceGenerator` shares its parser
with face selection and can append requested attributes to `manifest.csv` for
dataset queries and benchmarks.

```python
from pyWitnessAI import FaceGenerator

faces = FaceGenerator(description="short hair and blue eyes", n=5).generate()
```

Use `description=` for text and `generator_backend=` for a provider name or
backend instance. Generic generation requests and the provider registry live in
`pyWitnessAI.face_generation_backend`; the OpenAI adapter lives in
`pyWitnessAI.API.openai`. `from pyWitnessAI import OpenAIImageBackend` remains
available for configuring a client. Output records use `manifest.csv`.

## Portrait standardization

To mix photographs from different datasets with generated portraits, use
[FaceStandardizer](docs/FaceStandardizer.md) before retrieval or presentation:

```python
from pyWitnessAI import FaceStandardizer

catalog = FaceStandardizer().run("./face_pool", "./standardized_v1")
print(catalog.manifest[["image_id", "status", "quality_flags"]])
```

The basic call crops portraits around the image center to the output aspect
ratio, then resizes uniformly while preserving their backgrounds. It does not
rotate images or load face detection or segmentation. Use
`FaceStandardizer(alignment="eyes")` for eye alignment; padding defaults to a
solid color estimated from each source image's visible border. Supply
`background=(224, 224, 224)` to choose a fixed RGB color instead. The same call accepts
`generator.to_manifest()`. It preserves originals and metadata, saves standardized
portraits and coverage masks, and separates outputs requiring review from
`catalog.ready`. Pass `remove_background=True` to segment the foreground and
replace the background. A supplied reviewed `mask` or manifest `mask_path` also
applies background replacement to that image. The optional
`detail_eye_distance=48` setting is an example sampling ceiling; choose it from
the native resolution and display needs of your pool.

For optional local AI restoration, pass
`enhancer=CodeFormerEnhancer(fidelity=1.0, strength=0.5)` to `FaceStandardizer`.
Enhanced runs save paired unenhanced portraits in `baseline/` and mark every AI
output for review. See the [enhancement guide](docs/FaceStandardizer.md#optional-ai-enhancement)
for setup, provenance, and identity-preservation limits.

## Build and present a lineup

```python
from pyWitnessAI import LineupBuilder

lineup = LineupBuilder(suspect="./123.jpg", filler_pool="./fillers", seed=42).build()
lineup.show()
```

With a suspect, the defaults are six members, two rows, random presentation and a
target-present lineup. The builder samples five fillers from a larger pool without replacement.
Pass `suspect=None` to build six fillers with no designated suspect; `target`
automatically becomes `"absent"`, and `lineup.suspect_position` is `None`.
It also accepts image-path lists, an `ImageCatalog`, or a DataFrame with
`image_path`, including `FillerSelector(...).select(...).matches` and
`generator.to_manifest()`. Generation and selection remain separate reusable steps.
Use `select_by_similarity(suspect, pool, n=5, order="most")` or `"least"` to
choose fillers by face distance before building.

`lineup.standardize(FaceStandardizer())` returns a new lineup with the same
members, roles and order, while preserving quality records. `lineup.save(...)`
writes neutral member images and a presentation manifest to a new or empty
directory. See the [LineupBuilder guide](docs/LineupBuilder.md) for combined
workflows, target-absent lineups, layout, exports and migration.

## Lineup workflows and witness sessions

`create_lineup()` combines the existing generator and builder. Pass `suspect`
for the member to include, and `description`, `reference_image`, or both for
generation. A supplied `filler_pool` is used directly instead of generation.

```python
from pyWitnessAI import LineupSession, create_lineup

lineup = create_lineup(suspect="./suspect.png", description="short hair", target="present")

session = LineupSession("short hair and blue eyes", output_dir="./sessions/example_01")
first = session.start()
first.lineup.show()
second = session.respond(choice=3, confirmed=False, round_number=first.number)
second.lineup.show()
```

Each subsequent round keeps the selected displayed face and generates new
candidates from that reference and the original description. Sessions record
feedback, support retries and restoration, and stop on confirmation, explicit
stop, or a round limit. Their `unknown` / `candidate` labels keep witness
confirmation separate from TP/TA ground truth.

The optional `pyWitnessAI.API.http` module serves these workflows through HTTP.
Embed it with `from pyWitnessAI.API.http import create_app`:

```python
from pyWitnessAI.API.http import create_app

app = create_app(image_root="./images", output_dir="./lineup_service")
```

The same service can be started from the command line:

```bash
pyWitnessAI serve --image-root ./images --output-dir ./lineup_service
```

See the [workflow and session guide](docs/LineupSession.md) for TP/TA examples,
feedback, persistence, HTTP requests and deployment boundaries.

External adapters live in `pyWitnessAI.API`: `http.py` serves HTTP requests,
`cli.py` handles command-line arguments, `openai.py` calls the image service,
and `models.py` downloads and verifies model files. Importing the `API` package
itself does not load FastAPI or the OpenAI SDK. Local inference, scoring,
generation contracts and file hashing remain in their core modules.

## Video and lineup identification

Install `pyWitnessAI[vision]` for face recognition and video processing. This
includes the DeepFace and FaceNet stacks and plotting dependencies. The current
`ImagesAI` constructor initializes FaceNet as well as the chosen recognition
workflow, so constructing a pipeline can download weights on first use.

```python
from pyWitnessAI import LineupBuilder, VideoLineupPipeline

lineup = LineupBuilder(
    suspect="/path/to/perp.jpg",
    filler_pool="/path/to/fillers",
    seed=42,
).build()
pipe = VideoLineupPipeline(
    video_path="/path/to/crime_video.mp4",
    lineup=lineup,
    cfg={
        "analyzer_method": "process",
        "model": "Facenet",
        "backend": "mtcnn",
        "use_full_frame": False,
        "show_progress": True,
    },
)

result = pipe.run(frame_start=0, frame_end=99)
pipe.save(result, output_csv="results.csv")
pipe.savepywitness(result, "results_pyWitness.csv")
```

`LineupBuilder` assigns the supplied suspect its role from `target="present"`
or `"absent"`. For a target-absent lineup, either supply an innocent suspect with
`target="absent"`, or pass `suspect=None` for an all-filler lineup.
The resulting `Lineup` fixes the members and presentation order; roles are never
inferred from filenames.
`ImagesAI` computes distances between image sets, and `LineupAIDecision` turns
those distances into decisions. Lower distances indicate greater similarity.
`VideoLineupPipeline` combines these steps and returns a DataFrame with one row
per frame/probe, per-member distance columns and decision summaries.

Use `PipelineConfig` or a dictionary to select the recognition workflow:

```python
from pyWitnessAI import PipelineConfig

config = PipelineConfig(
    analyzer_method="process",       # Also process_verify / process_georgia_pipeline
    model="Facenet",
    backend="mtcnn",
    distance_metric="cosine",
    use_full_frame=False,
    no_face_fill=50.0,
)
pipe = VideoLineupPipeline("/path/to/crime_video.mp4", lineup=lineup, cfg=config)
```

`use_full_frame=True` uses the complete frame as a probe; `False` extracts faces.
The pipeline currently creates its own `LineupAIDecision`; its constructor does
not accept a separate identifier or `identifier_*` configuration fields.
See [VideoLineupPipeline](docs/VideoLineupPipeline.md) for supported parameters
and exports. The older `VideoAI.LineupLoader` is a separate API used by legacy
replication scripts. The old folder-loading `pyWitnessAI.Lineup` module has been
removed; use `LineupBuilder(...).build()` for current workflows.

For frame-level detection, gallery construction and probe-frame review, see
[Video analysis](docs/VideoAnalysis.md).

---

## Face search

`FaceSearch` recursively scans an image folder, scores every valid image with a
selected backend, applies an explicit match rule, and returns the matching faces
as a DataFrame. Optional metadata can be joined to the catalog and evaluated
with an explicit ground-truth rule to report hit counts.

See [docs/FaceSearch.md](docs/FaceSearch.md) for examples, metadata join rules,
custom selector backends, caching, and result display.

---

## Benchmarks and reusable CLIP tools

Both benchmarks are general package features. They accept a shared
`SelectorBackend` or a custom scoring function and use the same image discovery,
score validation, ranking and result display as face search.

| Benchmark | Inputs and purpose | Evaluation |
| --- | --- | --- |
| `AttributeRetrievalBenchmark` | A manifest or two labelled folders for one binary attribute | Precision/recall at K, average precision, ROC-AUC and threshold diagnostics |
| `FillerSelectionBenchmark` | A description, explicit negative labels and a generated-face dataset; evaluates one feature or cumulative stages | Per-stage top-N positive count, precision/recall and perfect-top-N checks |

The second benchmark can generate missing controlled examples. Choosing negative
sample labels defines its dataset; it does not automatically change positive
CLIP scoring into a positive-minus-negative prompt score. See the
[attribute benchmark guide](docs/AttributeRetrievalBenchmark.md) and
[filler benchmark guide](docs/FillerSelection.md#7-保留的两个通用-benchmark).

For direct embedding and index work, use:

```python
from pyWitnessAI.clip import ClipIndex, SentenceTransformerClipEncoder
```

The package CLI provides `parse` and `serve`. Core description and contrast
scoring defaults use general face prompts without a fixed CFD candidate pool.

## Research integrations

Study projects own their dataset adapters, candidate pools, questionnaire assets,
condition assignment, validation protocols and deployment configuration. Their
scripts and services call the shared Python interfaces described above and
record the data, model, prompt and code versions used for each experiment.

`pyWitnessAI serve` starts the general lineup/session service. A study-specific
service defines its own entry point, request contract and tests.

The former CFD/CLIP Qualtrics study has been removed from the current source
tree. Reproducing that study requires a historical Git revision containing its
study implementation, configuration and data requirements.

Deployment infrastructure belongs in a separate project. See the
[module map](docs/CoreModuleMap.zh-CN.md) for the current ownership boundaries.

---

## Development and release builds

See the [development guide](docs/Development.md) for core tests, study-test
boundaries, and release builds from a clean source distribution. Historical
files under `build/` are local artifacts, not supported entry points.

## FAQ

**Q: My similarity matrix is empty?**

A: Call an analyzer method first (`process*`) before `dataframe()`. 

**Q: How are decisions made in the video pipeline?**

A: The pipeline uses `LineupAIDecision` and the roles assigned by `LineupBuilder`.
See the pipeline guide for the exported TP/TA summaries.

**Q: What happens when MTCNN finds no face in a frame?**

A: The pipeline writes rows for every lineup member using `no_face_fill` (50 as default)
as the distance and records a filler-style summary.

---

