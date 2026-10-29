# FaceGenerator

`FaceGenerator` prepares portrait prompts, optionally processes them with a
conversation model, calls an image backend, and records each successful image.
The default keeps the direct image-generation path. Construction and
`generation_prompts()` are local operations; `generation_prompts(process=True)`
explicitly contacts the optional prompt processor. Call `generate()` or the
configured instance to create images.

`n` is the total number of individual portraits. The generator makes `n`
independent image requests, each asking for one portrait and writing one image file.
The optional conversation layer adds one processing request per portrait.
For a six-person lineup, generate five fillers for TP and add the suspect, or
generate six fillers for an all-filler TA lineup. `create_lineup(size=6, ...)`
calculates these counts from whether `suspect` is supplied.

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

# Reference-guided portraits, requesting less resemblance to the reference.
FaceGenerator(reference_image="./reference.png", similarity=20, n=5).generate()

# Combined inputs, using the simple description-and-reference template.
FaceGenerator(
    description="short hair and brown eyes",
    reference_image="./reference.png",
    output_dir="./reference_fillers",
    n=5,
).generate()
```

For reference-only generation, `similarity` ranges from 0, meaning distinctly
dissimilar, to 100, meaning the greatest resemblance. It supplies numeric prompt
guidance, rather than a measured face-recognition score or guaranteed output
threshold. When both a description and reference are supplied, the simple
template below omits numeric resemblance instructions. In that mode,
`similarity` is retained in the request and manifest for compatibility and does
not affect the prompt.

A backend must declare `supports_reference_image=True` to accept a reference. Unsupported requests raise an error. OpenAI uploads the reference file to `images.edit`; without a reference it calls `images.generate`. No `similarity` API parameter is sent. See the [OpenAI image editing reference](https://developers.openai.com/api/reference/python/resources/images/methods/edit).

Each `FaceGenerator` batch uses one `reference_image`. To use a different
reference for every filler, first sample five reference paths for TP or six for
TA, then make a separate `FaceGenerator(n=1, reference_image=path, ...)` call for
each path and pass the generated portraits to `LineupBuilder`. Reference-folder
sampling and ethnicity matching are experiment setup steps.

## Portrait prompts

Description-only generation asks for one under average-looking man matching
the description. Each request uses a singular portrait prompt. Description and
reference generation uses this template:

```text
Make a photo of a man based on this image, with the following features: {description} Make it an image portrait, only from shoulders above. Make the background gray. The people should be looking at the camera with frontal face. Do not emphasize or depict body actions, hand placement, or clothing interactions.
```

Both templates use the parser's `ParsedDescription.model_text` as the description
and request the same shoulders-above crop, gray background, frontal face, and
restriction on depicting body actions. Explicitly accepted attribute labels
missing from `model_text` are appended to the same description block. The
templates use male framing for this experiment; explicit female or non-binary
schema labels adjust the subject accordingly. Reference-only generation uses
the neutral subject "person".

## Optional conversation layer

Set `conversation_model` to enable an OpenAI Responses processing step before
each image request. The processor interprets the request, extracts constraints,
resolves conflicts according to the supplied priorities, and rewrites the image
prompt. The image backend and its `model` parameter remain separately configured.
Omitting both `conversation_model` and `prompt_processor` preserves direct mode.

```python
from pyWitnessAI import FaceGenerator

# Direct mode: local prompt construction followed by the image request.
direct = FaceGenerator(description="a man with short hair and blue eyes")

# Optional processing: one conversation request, then one image request per face.
generator = FaceGenerator(
    description="一名短发男子，鼻子很突出；眼睛可能是蓝色，也可能是灰色。",
    conversation_model="gpt-5-mini",
    n=2,
)
base_prompts = generator.generation_prompts()  # Local, unprocessed preview
final_prompts = generator.generation_prompts(process=True)  # Network and cost
processing_results = generator.prompt_processing_results
faces = generator.generate()  # Reuses the cached processed prompts
```

Processing receives the original description, parser attribute hints, any filler
profile, and any reference image. This preserves details or languages that the
default English schema parser may omit. The request's `attributes` dictionary
contains parser hints unless `attributes_authoritative=True` identifies explicit
experimental controls. Conversation-model interpretations do not become
canonical manifest attribute labels automatically.

Fixed portrait composition has the highest priority: one portrait from the
shoulders above, a gray background, and a frontal face looking at the camera,
without depicting body actions or clothing interactions. The generator ensures
those instructions are included in the final prompt. Among appearance sources, the original
witness description takes priority over the filler profile, followed by the
reference image. Explicit attributes in a supplied `ParsedDescription` and
randomly sampled conditions are authoritative experimental controls and take
priority over conflicting raw wording. The full original description remains
available to the processor. The processor is instructed to omit uncertain or mutually
conflicting witness traits and record the conflict rather than invent a
definite value. A model refusal, incomplete response, malformed result, or
processing error stops the request before image generation; there is no silent
fallback to direct mode.

`generation_prompts(process=True)` processes and caches each portrait's result
without generating images. Later previews return the cached final prompt and
later `generate()` calls reuse it. `generate(dry_run=True)` never contacts either
provider or writes files: it prints base prompts unless processing results
have already been cached. Create a new instance when changing descriptions,
profiles, reference images, or processing settings.

`prompt_processing_results` returns available results in portrait order. Each
`PromptProcessingResult` has `prompt`, `summary`, `constraints`, `conflicts`,
`model`, and `provider`; constraints and conflicts are tuples of strings. These
records make the prompt transformation inspectable. They are model-reported
interpretations, not verification of the generated pixels. This layer uses a
separate request for each portrait and does not maintain multi-turn chat
history. It adds network latency and conversation-model cost to image
generation, using the same `pyWitnessAI[generation]` extra and `OPENAI_API_KEY`.

For an already configured OpenAI client, or another processor implementation,
pass `prompt_processor` instead:

```python
from openai import OpenAI
from pyWitnessAI import FaceGenerator, OpenAIPromptProcessor

client = OpenAI()
generator = FaceGenerator(
    description="a man with short hair and blue eyes",
    prompt_processor=OpenAIPromptProcessor(model="gpt-5-mini", client=client),
)
faces = generator.generate()
```

Do not supply both `conversation_model` and `prompt_processor`. A custom
processor implements `process(PromptProcessingRequest) -> PromptProcessingResult`;
both contract types are exported from `pyWitnessAI`. The custom processor is
responsible for its provider calls and structured response. The generator
validates its result before dispatching the image request.

## Share a parser with selection

```python
from pyWitnessAI import FaceGenerator, FaceSearch, SchemaDescriptionParser

parser = SchemaDescriptionParser()
generator = FaceGenerator(description="a man with blue eyes", parser=parser)
prompts = generator.generation_prompts()  # No files or provider calls
search = FaceSearch.from_folder("./existing_faces", parser=parser)
```

`SchemaDescriptionParser` implements `parse(text) -> ParsedDescription`. Its English rules retain explicitly recognized attributes and do not invent definite labels from unknown, uncertain, or negated text. In direct mode, if no usable description remains, generation raises an error. A custom parser can support another language or vocabulary. With the optional conversation layer, the original description is available even when the schema parser retains no usable text.

You can also choose `"raw"`, `"lenient"`, or `"strict"`. The generator uses that parser's text inside shared portrait instructions. It does not parse the text again to populate the manifest. Canonical attributes use `family == dimension == schema field name`; other attribute columns remain empty.

For a full eyewitness description, use `RawDescriptionParser(lowercase=False)`
to retain all wording and capitalization while collapsing whitespace. The
default schema parser retains only recognized canonical attributes, so it can
omit details such as a particular hair parting or observed behavior.

```python
from pyWitnessAI import FaceGenerator, RawDescriptionParser

witness_description = (
    "1) Dark hair (dark brown/black) / "
    "2) Medium length hair sort of parted down the middle / "
    "3) prominent nose (big nose) / 4) He looked nervous"
)
generator = FaceGenerator(
    description=witness_description,
    parser=RawDescriptionParser(lowercase=False),
    n=5,  # TP fillers; use n=6 for an all-filler TA lineup.
)
prompts = generator.generation_prompts()
```

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

## Optional filler profiles

Use `filler_profiles` to supply a different supplementary design for each
generated filler. It must be a sequence of exactly `n` distinct, nonempty
strings, and an explicit `description` is required. The default `None` keeps the
existing prompts and generation behavior.

Each profile appears in a separate `Additional design requirements` block.
Witness-described traits take priority: researchers should supply only
dimensions the witness did not describe. Direct mode expresses that priority
in the prompt without automatically detecting conflicts. The optional
conversation layer is instructed to resolve and report conflicts. When profiles are supplied, the description-only
template omits its generic request for facial features and hair styles to
differ from the other lineup members; the explicit profiles guide those
differences instead.

Profiles are separate from the witness description and are not processed as
witness traits by the description parser. `RawDescriptionParser(lowercase=False)`
continues to preserve the witness's full wording and capitalization, with
whitespace normalized. For example, if the witness described hair, eyes, nose
and facial hair but left face shape unmentioned:

```python
from pyWitnessAI import FaceGenerator, RawDescriptionParser, create_lineup

# Reuse the full witness_description from the raw-parser example above.
profiles = [
    "A narrow oval face.",
    "A broad square face.",
    "A round face.",
    "A long rectangular face.",
]
N = len(profiles)  # Total lineup size; use any N >= 2 with N profiles.
parser = RawDescriptionParser(lowercase=False)

# Preview one prompt per planned filler without calling a provider.
generator = FaceGenerator(
    description=witness_description,
    parser=parser,
    n=N - 1,
    filler_profiles=profiles[:N - 1],
)
prompts = generator.generation_prompts()

# These calls generate portraits and therefore contact the image provider.
tp = create_lineup(
    suspect="./suspect.png", target="present", size=N,
    description=witness_description, parser=parser,
    filler_profiles=profiles[:N - 1],
    output_dir="./profile_tp_fillers", seed=11,
)
ta = create_lineup(
    suspect=None, target="absent", size=N,
    description=witness_description, parser=parser,
    filler_profiles=profiles[:N],
    output_dir="./profile_ta_fillers", seed=12,
)
tp.save("./profile_tp_lineup")
ta.save("./profile_ta_lineup")
```

`FaceGenerator.n` counts generated portraits (fillers in this example);
`create_lineup.size` counts all lineup members. TP needs `N - 1` profiles because the suspect supplies one
member. An all-filler TA lineup uses `suspect=None` and needs `N` profiles. The
caller chooses the profiles explicitly; `create_lineup` requires the exact
filler count and does not select an extra TA profile automatically. Profiles
can also accompany `reference_image` when an explicit description is supplied.

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

The supplementary profile is recorded separately as `GeneratedFace.filler_profile`
and the manifest's `filler_profile` column. `original_description` retains the
original witness description; profiles do not populate witness attribute
columns.

When processing is enabled, `prompt` records the actual final image prompt and
`base_prompt` preserves its locally constructed input. Processing provenance
uses `prompt_processing_provider`, `prompt_processing_model`,
`prompt_processing_summary`, `prompt_processing_constraints`, and
`prompt_processing_conflicts`. The constraints and conflicts columns contain
JSON-encoded string lists. Canonical attribute columns continue to reflect the
description parser or explicit sampling; model-extracted constraints are kept
in the separate processing fields. Processed records from ordinary text use
`attribute_label_source="parsed_attribute_hints"` because semantic conflict
resolution may omit a parsed trait from the final prompt. Authoritative
pre-parsed or sampled attributes retain `"requested"`; records without
attributes use `"unspecified"`. The processing
result's `prompt` includes the enforced portrait instructions and matches the
prompt sent to the image backend.

In direct mode, `attribute_label_source="requested"` describes generation conditions. `attributes_verified=False` means those attributes have not been checked in the output pixels. Reference-only generation does not infer attributes from the reference. These records support condition-based queries and experiment design; they are not automatically human-verified visual labels.

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
| `FaceGenerator.py` | Prompt planning, optional processing cache, image dispatch, results and manifest records | Description, FaceAttributeSchema, prompt processor, backend, GeneratedFaceDataset |
| `Description.py` | Reusable parsing strategies | FaceAttributeSchema and optional contrast rules |
| `FaceAttributeSchema.py` | Canonical attributes, sampling values and portrait templates | Standard library |
| `face_generation_backend.py` | Provider-independent request, base class, validation and registry | Standard library; loads the chosen adapter when requested |
| `prompt_processing.py` | Provider-independent prompt processing request and result contracts | Standard library |
| `API/openai.py` | OpenAI image model defaults, lazy client calls, reference uploads and response decoding | Generic backend interface; OpenAI SDK when used |
| `API/openai_prompt_processor.py` | OpenAI Responses prompt processing with structured results and reference input | Prompt processing interface; OpenAI SDK when used |
| `GeneratedFaceDataset.py` | Manifest storage, requested-attribute matching and dataset generation | FaceAttributeSchema, backend; FaceGenerator when generating |
