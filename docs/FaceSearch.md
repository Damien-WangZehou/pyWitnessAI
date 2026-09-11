# FaceSearch

`FaceSearch` recursively discovers face images, optionally joins metadata, scores
every valid image with a selector backend, applies an explicit match policy, and
returns a manifest-backed result that can be displayed or saved.

For a default top-five workflow, use `FillerSelector` from the
[progressive experiment guide](FillerSelection.md). Both interfaces support
independent `parser` and `scoring` options, with raw normalized text and unscaled
CLIP cosine as the defaults. Selection results retain `parsed_description`,
per-attribute `components`, and `scoring_metadata` when applicable.

## Basic search

Install the CLIP and plotting extras for this example:

```bash
python -m pip install "pyWitnessAI[clip,plot]"
```

The base installation supports catalogs, description parsing and custom backends.
`clip` adds the model encoder; `plot` adds result display. The first CLIP scoring
call may download model weights. Import the API names you use explicitly so
unrelated optional features remain unloaded.

```python
from pyWitnessAI import FaceSearch

search = FaceSearch.from_folder(
    "./face_pool",
    recursive=True,
    selector="clip",
    selector_kwargs={
        "model_name": "clip-ViT-B-32",
        "cache_dir": "./build/face_search_cache",
    },
)

result = search.select(
    "A middle-aged male person with short dark hair.",
    min_score=0.27,
)

print(result.matches)
result.print_summary()
result.show()
```

CLIP produces a similarity ranking rather than a natural yes/no decision.
Consequently, `select()` requires one of:

- `top_k=20`
- `min_score=0.27`
- both `top_k=20, min_score=0.27`
- an explicit `policy=AllRankedPolicy()`

Thresholds should be calibrated with labelled data for the selected model and
dataset.

## Contrastive prompt scoring

For a binary or ordinal attribute, an experiment can compare a positive-minus-negative
prompt margin with single positive similarity. `SelectorQuery.contrastive(...)`
constructs this query explicitly:

```python
from pyWitnessAI import FaceSearch, SelectorQuery

search = FaceSearch.from_folder(
    "./face_pool",
    selector="clip",
    selector_kwargs={
        "model_name": "clip-ViT-B-32",
        "cache_dir": "./build/face_search_cache",
    },
)

query = SelectorQuery.contrastive(
    positive_texts=[
        "A frontal face photograph of a person with thick eyebrows.",
        "A face with dense, prominent eyebrows.",
    ],
    negative_texts=[
        "A frontal face photograph of a person with thin eyebrows.",
        "A face with sparse, faint eyebrows.",
    ],
)

ranked = search.score(query)
print(ranked[[
    "image_id",
    "positive_similarity",
    "negative_similarity",
    "selector_score",
]])
```

The CLIP backend computes:

```text
selector_score = mean(positive prompt similarities)
               - mean(negative prompt similarities)
```

Larger values indicate stronger evidence for the positive pole. A score of zero
is a natural prompt midpoint, not a calibrated operational threshold; choose any
classification threshold on separate validation data.

## Metadata and hit counts

Metadata can be a DataFrame, CSV, TSV, or Excel file. For nested folders, a
relative path or stable image ID is safer than a filename because different
folders may contain files with the same name. Joins are `one_to_one` by default;
use `many_to_one` only when several images intentionally share one metadata row.
Excel files require `python -m pip install "pyWitnessAI[excel]"`; CSV and TSV
use the base installation.

Use `ImageCatalog` directly to inspect or reuse a pool before scoring:

```python
from pyWitnessAI import ImageCatalog

catalog = ImageCatalog.from_folder("./face_pool", metadata="./face_metadata.csv")
print(catalog.report.to_dict())
all_images = catalog.manifest  # Includes missing/invalid rows with status and error.
valid_images = catalog.ready

# An existing table needs only image_path; supplied IDs and metadata are retained.
catalog = ImageCatalog.from_manifest(image_table, base_dir="./face_pool")
```

Both factories prepare the same image fields, including `file_extension`.
Folder scanning validates image contents by default. `from_manifest` checks
file existence when status fields are absent; pass `validate_images=True` to
recheck image contents and replace supplied status/error fields.

Automatic metadata joins choose the greatest overlap among valid one-to-one
keys. Repeated filenames are skipped when a unique relative-path key is
available. Use an explicit `MetadataJoinSpec` for intentional many-to-one joins.
Joining preserves image order and count. Conflicting metadata columns receive
an unused `metadata__` prefix, repeated if necessary, so source values are kept
without overwriting catalog fields. No metadata column names are reserved for
internal temporary join values.

```python
from pyWitnessAI import (
    FaceSearch,
    GroundTruthSpec,
    MetadataJoinSpec,
)

search = FaceSearch.from_folder(
    "./face_pool",
    recursive=True,
    metadata="./face_metadata.csv",
    metadata_join=MetadataJoinSpec(
        image_key="relative_image_path",
        metadata_key="relative_image_path",
        validate="one_to_one",
    ),
    selector="clip",
)

result = search.select(
    "A male person with short dark hair.",
    top_k=50,
    min_score=0.25,
    ground_truth=GroundTruthSpec(
        conditions={
            "gender": "male",
            "hair_length": "short",
        }
    ),
)
```

`result.summary` reports:

- total, valid, skipped, scored, and selected image counts;
- metadata match and missing counts;
- labelled and unlabelled selected counts;
- `hit_image_count` and `hit_target_count`;
- precision and recall when ground truth is available.

Metadata alone does not define a hit. Without an explicit `GroundTruthSpec`,
hit counts are `None`, not zero.

## Custom selector backend

A backend scores every candidate and returns one row per `image_id`.

```python
import pandas as pd

from pyWitnessAI import SelectorBackend, SelectorQuery


class MySelector(SelectorBackend):
    name = "my-selector"
    model_name = "my-model-v1"

    def score(
        self,
        manifest: pd.DataFrame,
        query: SelectorQuery,
    ) -> pd.DataFrame:
        scores = my_model_score(
            manifest["image_path"].tolist(),
            query.description,
        )
        return pd.DataFrame(
            {
                "image_id": manifest["image_id"],
                "selector_score": scores,
            }
        )
```

The shared result normaliser rejects duplicate, unknown, missing, non-numeric,
NaN, and infinite scores. The current catalog remains the source of truth for
all metadata, so cached embeddings cannot return stale labels. Extra backend
diagnostics are retained under a `backend__` prefix unless they use a documented
score column or an explicit `selector__` namespace.

## Image handling

The general service treats one image as one candidate face and uses
`preprocess="whole_image"`. Invalid or unreadable images remain in the catalog
with `status` and `error`, but are not sent to the selector.

The historical `CLIPFillerSelector` detects and crops the largest face before
encoding. It remains available for compatibility; its preprocessing semantics
are intentionally not applied silently to existing benchmarks.
