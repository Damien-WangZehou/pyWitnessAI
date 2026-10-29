# Attribute retrieval benchmark

`AttributeRetrievalBenchmark` evaluates a binary visual attribute without requiring
that attribute to be part of the generated-face dataset layout. It accepts either
a CSV/DataFrame manifest or two labelled folders, and uses CLIP by default.

Install `pyWitnessAI[clip,plot]` for CLIP scoring and plots. A custom selector can
run without CLIP; pass `display=False` to score without loading plotting tools.

`FillerSelectionBenchmark` serves a different workflow: it builds single-feature
or cumulative stages from a description and matches or generates controlled
positive/negative examples. This benchmark evaluates an existing labelled pool
and supports explicit positive-minus-negative prompt scoring. Both remain
general core modules and share the selector and display functions; neither
requires the former CFD pilot package. See the
[benchmark comparison](FillerSelection.md#7-保留的两个通用-benchmark).

## Two-folder probe

```python
from pyWitnessAI import AttributeRetrievalBenchmark

bushy_probe = AttributeRetrievalBenchmark.from_folders(
    positive_dir="./data/eyebrows/brown_bushy",
    negative_dir="./data/eyebrows/brown_thin",
    attribute="eyebrow_bushiness",
    positive_label="brown bushy eyebrows",
    negative_label="brown thin eyebrows",
    positive_prompts=[
        "A frontal face photograph of a person with brown bushy eyebrows.",
        "A face with brown, very bushy eyebrows.",
    ],
    negative_prompts=[
        "A frontal face photograph of a person with brown thin eyebrows.",
        "A face with thin, sparse brown eyebrows.",
    ],
    selector_model="clip-ViT-B-32",
)

bushy_probe.print()
bushy_probe.run()
bushy_probe.statistics()
bushy_probe.plot()
```

With negative prompts present, the default score is contrastive:

```text
mean similarity(image, positive prompts)
- mean similarity(image, negative prompts)
```

`score_mode="auto"` is the default and resolves to `"contrastive"` whenever
negative prompts are present. Pass `score_mode="contrastive"` explicitly when
you want the experiment configuration to state that choice rather than infer it.
The resolved backend query is available as `benchmark.selector_query`.

The summary includes `Precision@K`, `Recall@K`, average precision, ROC-AUC,
threshold accuracy, balanced accuracy, and positive-rank diagnostics. By default,
`K` equals the number of positive images.

Metrics count labelled images, even when several images share one identity.
Equal scores preserve the backend's supplied rank, then catalog order; ROC-AUC
credits a positive/negative score tie as half a win. Threshold metrics use
`selector_score >= decision_threshold` (default `0.0`). With the built-in
`selector="clip"` and no negative prompts, threshold metrics are left unset.

## Manifest probe

The manifest needs an image path and one binary label per image:

```csv
image_path,label
images/001.png,hawk nose
images/002.png,straight nose
```

```python
from pyWitnessAI import AttributeProbeSpec, AttributeRetrievalBenchmark

probe = AttributeProbeSpec.from_schema(
    "nose",
    "hawk nose",
    negative_label="straight nose",
)
benchmark = AttributeRetrievalBenchmark("./data/nose/manifest.csv", probe)
benchmark.run()
```

The general face schema is parsing-only, so omitting `negative_label` fails
closed instead of inventing an opposite. The output metadata records whether
the contrast was passed explicitly or came from an explicitly selected custom
schema, together with that schema's name and version.

Relative image paths are resolved relative to the manifest CSV.

The same schema-derived probe can be used with folders:

```python
benchmark = AttributeRetrievalBenchmark.from_folders(
    "./data/nose/hawk",
    "./data/nose/straight",
    probe=probe,
)
```

## Reusing the same CLIP backend

The benchmark accepts a shared `ClipSelectorBackend`. This lets several probes
use the same validated image-embedding cache while each probe supplies its own
positive and negative prompts:

```python
from pyWitnessAI import (
    AttributeProbeSpec,
    AttributeRetrievalBenchmark,
    ClipSelectorBackend,
)

backend = ClipSelectorBackend(
    model_name="clip-ViT-B-32",
    cache_dir="./build/attribute_clip_cache",
)

eyebrow_probe = AttributeProbeSpec(
    attribute="eyebrows",
    positive_label="thick eyebrows",
    negative_label="thin eyebrows",
    positive_prompts=(
        "A frontal face photograph of a person with thick eyebrows.",
        "A face with dense, prominent eyebrows.",
    ),
    negative_prompts=(
        "A frontal face photograph of a person with thin eyebrows.",
        "A face with sparse, faint eyebrows.",
    ),
)

benchmark = AttributeRetrievalBenchmark(
    "./data/eyebrows/manifest.csv",
    eyebrow_probe,
    selector=backend,
    score_mode="contrastive",
)

print(benchmark.selector_query)
results = benchmark.run(display=False)
summary = benchmark.statistics(display=False)
```

The returned ranking retains `positive_similarity`, `negative_similarity`, and
their difference in `selector_score`. Reuse is safe only when the gallery has
stable image IDs; the cache fingerprint also checks model name, file path, file
size, and modification time.

## Extending the face schema

An extension is immutable: it returns a new schema and does not modify the project
default for other experiments. The project default is deliberately parsing-only:
it never invents a negative label. A custom `contrasts` mapping is therefore an
explicit, experiment-specific decision and should be preregistered.

```python
from pyWitnessAI import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    FaceAttributeDefinition,
    FillerSelectionBenchmark,
)

extended_schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
    FaceAttributeDefinition(
        name="skin_marking",
        order=65,
        patterns=((r"\bprominent cheek freckles\b", "prominent cheek freckles"),),
        contrasts={"prominent cheek freckles": "an even cheek complexion"},
    )
)

benchmark = FillerSelectionBenchmark(
    "a person with prominent cheek freckles",
    mode="ladder",
    attribute_schema=extended_schema,
)
```

Custom schema attributes are stored in generated-face manifests as
`attribute__<name>` columns, so they remain selectable without changing the
dataset manager.

## Experimental separation

To distinguish eyebrow colour from bushiness, use separate controlled probes:

- Bushiness: brown bushy versus brown thin/sparse eyebrows.
- Colour: brown bushy versus non-brown bushy eyebrows.
- Joint concept: brown bushy eyebrows versus matched controls.

A joint probe alone cannot establish which part of a compound description CLIP
used for retrieval.
