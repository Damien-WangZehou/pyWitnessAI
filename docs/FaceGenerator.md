# FaceGenerator

`FaceGenerator` prepares portrait prompts, calls an image backend, and records each successful image. Construction and `generation_prompts()` are local operations. Call `generate()` or the configured instance to create images.

For common framing, backgrounds, and sampling across generated and real photos,
pass `generator.to_manifest()` to [FaceStandardizer](FaceStandardizer.md).
Prompt instructions alone do not enforce those pixel-level properties.

```python
from pyWitnessAI import FaceGenerator

generator = FaceGenerator()  # OpenAI backend, ./facesGenerated, one image
print(generator.generation_prompts())
faces = generator.generate()
```

Defaults include `parser="schema"`, `write_manifest=True`, and `similarity=75`. Install `pyWitnessAI[generation]` and set `OPENAI_API_KEY` to use OpenAI. `generator_backend` accepts a registered provider name or a configured `ImageGenerationBackend` instance.

## Input modes

| Inputs | Generation behavior | Recorded attributes |
| --- | --- | --- |
| No description or reference; `write_manifest=False` | Ask the backend for a random portrait | No manifest is written |
| No description or reference; `write_manifest=True` | Sample explicit schema values and use the shared portrait template | Sampled, requested attributes |
| `description` | Parse the description and generate a portrait | Canonical attributes supplied by the parser |
| `reference_image` | Generate using the supplied image | Undescribed attributes remain empty |
| Both | Combine the description with the reference | Requested attributes and reference information |

Disabling manifest writing retains any description or reference conditions.

```python
# Random portraits without attribute records.
FaceGenerator(output_dir="./random_faces", n=10, write_manifest=False).generate()

# Sample reproducible description conditions and record them.
FaceGenerator(output_dir="./labelled_faces", n=10, seed=42).generate()

# Description-guided portraits.
FaceGenerator(description="a man with short hair and blue eyes", n=5).generate()

# Reference-guided portraits.
FaceGenerator(reference_image="./reference.png", n=5).generate()

# Combined inputs, requesting less resemblance to the reference.
FaceGenerator(
    description="short hair and brown eyes",
    reference_image="./reference.png",
    similarity=20,
    output_dir="./different_faces",
    n=5,
).generate()
```

`similarity` ranges from 0, meaning distinctly dissimilar, to 100, meaning the greatest resemblance. It is prompt guidance, not a measured face-recognition score or a guaranteed output threshold. Explicit description attributes take priority when they conflict with the reference.

A backend must declare `supports_reference_image=True` to accept a reference. Unsupported requests raise an error. OpenAI uploads the reference file to `images.edit`; without a reference it calls `images.generate`. Similarity instructions stay in the prompt, and no `similarity` API parameter is sent. See the [OpenAI image editing reference](https://developers.openai.com/api/reference/python/resources/images/methods/edit).

## Share a parser with selection

```python
from pyWitnessAI import FaceGenerator, FaceSearch, SchemaDescriptionParser

parser = SchemaDescriptionParser()
generator = FaceGenerator(description="a man with blue eyes", parser=parser)
prompts = generator.generation_prompts()  # No files or provider calls
search = FaceSearch.from_folder("./existing_faces", parser=parser)
```

`SchemaDescriptionParser` implements `parse(text) -> ParsedDescription`. Its English rules retain explicitly recognized attributes and do not invent definite labels from unknown, uncertain, or negated text. If no usable description remains, generation raises an error. A custom parser can support another language or vocabulary.

You can also choose `"raw"`, `"lenient"`, or `"strict"`. The generator uses that parser's text inside shared portrait instructions. It does not parse the text again to populate the manifest. Canonical attributes use `family == dimension == schema field name`; other attribute columns remain empty.

An existing `ParsedDescription` can be passed directly and is not parsed again. This also preserves explicit benchmark contrast values:

```python
from pyWitnessAI import ParsedDescription

parsed = ParsedDescription.from_attributes(
    {"gender": "male", "eyes": "blue eyes"},
    original_text="a man with blue eyes",
)
generator = FaceGenerator(description=parsed, n=3)
```

Attribute definitions and random sampling values belong in `FaceAttributeSchema`:

```python
from pyWitnessAI import DEFAULT_FACE_ATTRIBUTE_SCHEMA, FaceAttributeDefinition

schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
    FaceAttributeDefinition(
        "skin_marking", 200,
        patterns=((r"\bfreckled skin\b", "freckled skin"),),
        sample_values=("freckled skin",),
    )
)
generator = FaceGenerator(attribute_schema=schema, n=10, seed=42)
```

Random sampling uses explicit `sample_values`, not regular expressions or benchmark contrast definitions. `seed` reproduces sampled descriptions, not backend pixels. Default sampling keeps neutral expressions and consistent clothing; bald heads are not assigned hair color or texture. A custom schema without sample values cannot supply random attribute records.

## Inspect results and query the manifest

```python
generator = FaceGenerator(description="short hair and blue eyes", n=3)
faces = generator.generate()
records = generator.to_manifest()
paths = generator.image_paths()

# Optional Notebook display, separate from generation.
from IPython.display import Image, display

for face in faces:
    display(Image(filename=face.path, width=240))
```

With recording enabled, successful images are appended to `manifest.csv` in `output_dir`. Each new generator owns a new batch ID. Repeating `generate()` on the same instance reuses that batch's files. Failed outputs are not recorded; earlier successful records remain. Create a new instance when changing generation conditions.

Records include image IDs and paths, batch information, prompts, model details, schema attributes, `generation_mode`, `reference_image_path`, and `requested_similarity`. Standard attributes use columns such as `eyes`; extensions use names such as `attribute__skin_marking`.

`attribute_label_source="requested"` describes generation conditions. `attributes_verified=False` means those attributes have not been checked in the output pixels. Reference-only generation does not infer attributes from the reference. These records support condition-based queries and experiment design; they are not automatically human-verified visual labels.

```python
from pyWitnessAI import GeneratedFaceDataset, SchemaDescriptionParser

dataset = GeneratedFaceDataset("./labelled_faces")
matches = dataset.find("blue eyes", parser=SchemaDescriptionParser(), n=10)
```

`find()` parses text and matches manifest columns without loading an AI model. Its default `match="contains"` allows additional recorded attributes; `match="exact"` requires the attribute set to match. `select(FaceDescriptionSchema(...))` remains available for structured dataset queries. CLIP search through `FaceSearch` is separate.

## Reuse generated data in benchmarks

`FillerSelectionBenchmark(dataset_root=output_dir, generate_missing=False, dataset_match="contains", ...)` can reuse this dataset. Each stage still needs enough positive and negative images and an explicit contrast definition.

For a single attribute, filter the manifest to the labels being compared:

```python
import pandas as pd
from pyWitnessAI import AttributeProbeSpec, AttributeRetrievalBenchmark

manifest = pd.read_csv("./labelled_faces/manifest.csv")
pool = manifest[manifest["eyes"].isin(["blue eyes", "brown eyes"])]
probe = AttributeProbeSpec(
    attribute="eyes", positive_label="blue eyes", negative_label="brown eyes",
    positive_prompts=("blue eyes",), negative_prompts=("brown eyes",),
)
benchmark = AttributeRetrievalBenchmark(pool, probe, label_column="eyes")
```

Both labels must be present. Summaries propagate `evaluation_label_source`, so metrics computed against requested attributes retain that provenance. Dataset generation helpers use `GeneratedFaceDataset.generate_faces(schema, verbal_description=..., n=..., generator_backend=...)`.

## Backend configuration

The OpenAI adapter lives in `pyWitnessAI.API.openai`. It accepts an optional
configured client; model names and OpenAI-specific request handling belong to
this adapter:

```python
from pyWitnessAI import FaceGenerator
from pyWitnessAI.API.openai import OpenAIImageBackend

backend = OpenAIImageBackend()  # Optionally pass an already configured client
generator = FaceGenerator(generator_backend=backend, description="short hair", n=2)
```

The shorter `from pyWitnessAI import OpenAIImageBackend` is also supported.
Importing `pyWitnessAI.API` does not load the OpenAI SDK; client construction is
deferred until the backend performs a request.

Generic contracts remain in `pyWitnessAI.face_generation_backend`:

```python
from pyWitnessAI.face_generation_backend import (
    ImageGenerationBackend,
    ImageGenerationRequest,
    register_image_generation_backend,
)
```

Custom backends subclass `ImageGenerationBackend`, implement `generate(request)`, and write one image to `request.output_path`. Declare reference support only when the backend actually consumes `request.reference_image`. The generator validates requests before dispatch. `register_image_generation_backend()` adds a provider name for later use. The generic request, validation and registry do not import a provider SDK.

## Module responsibilities

| Module | Responsibility | Main dependencies |
| --- | --- | --- |
| `FaceGenerator.py` | Prompt planning, dispatch, results and manifest records | Description, FaceAttributeSchema, backend, GeneratedFaceDataset |
| `Description.py` | Reusable parsing strategies | FaceAttributeSchema and optional contrast rules |
| `FaceAttributeSchema.py` | Canonical attributes, sampling values and portrait templates | Standard library |
| `face_generation_backend.py` | Provider-independent request, base class, validation and registry | Standard library; loads the chosen adapter when requested |
| `API/openai.py` | OpenAI model defaults, client calls, reference-file uploads and response decoding | Generic backend interface; OpenAI SDK when used |
| `GeneratedFaceDataset.py` | Manifest storage, requested-attribute matching and dataset generation | FaceAttributeSchema, backend; FaceGenerator when generating |
