# FaceStandardizer

Standardize real and generated portraits through one local pixel pipeline before
retrieval or experimental presentation. Original files are never overwritten.

```bash
python -m pip install -e ".[standardization]"
```

```python
from pyWitnessAI import FaceStandardizer, FaceSearch

standardizer = FaceStandardizer()
catalog = standardizer.run("./portraits", "./standardized_v1")
print(catalog.manifest[["image_id", "status", "quality_flags"]])
# FaceSearch uses catalog.ready; review/error rows are excluded.
search = FaceSearch(catalog)
```

The basic call crops around the image center to the output aspect ratio, then
resizes uniformly. It preserves the source background, does not rotate the image,
and does not load or run face detection or foreground segmentation. Opaque sources
fill the output without padding.

To control face size while keeping the original orientation, supply an eye-distance
target in center mode:

```python
standardizer = FaceStandardizer(alignment="center", eye_distance=0.30)
catalog = standardizer.run("./portraits", "./scaled_v1")
```

This measures the eyes and sets their output separation to `eye_distance * width`,
scaling uniformly about the image center without rotating. Larger values enlarge
the face; smaller values shrink it and may expose padding, filled using
`background`. `eye_height` does not affect center mode. Omitting `eye_distance`
(or passing `None`) keeps the basic center crop and does not require face detection
unless another option needs landmarks.

To prioritize filling the output over the exact eye-distance target:

```python
standardizer = FaceStandardizer(
    alignment="center", eye_distance=0.20, fill_canvas=True,
)
```

In center mode, `fill_canvas=True` uses the larger of the requested eye-based
scale and `max(output_width / input_width, output_height / input_height)`.
This keeps opaque source images covering the output without adding padding,
while preserving the image center, proportions, and original orientation. Some
portraits will consequently have an eye separation larger than the requested
fraction. Per-image `output_eye_distance` records the actual distance in pixels,
and `eye_distance_clamped` records whether the cover scale increased it.
Source transparency and removed backgrounds can still use the configured fill;
this option only prevents padding outside the source rectangle. It is supported
only with `alignment="center"`. Without `eye_distance`, it leaves the basic,
model-free center crop unchanged. The default `fill_canvas=False` preserves the
exact eye-distance target and allows padding when needed.

To align portraits using their eyes, opt in explicitly:

```python
standardizer = FaceStandardizer(alignment="eyes")
catalog = standardizer.run("./portraits", "./eye_aligned_v1")
```

In eye alignment mode, areas outside the transformed source use a solid color
estimated from that image's border by default. To choose one color for all images:

```python
standardizer = FaceStandardizer(alignment="eyes", background=(224, 224, 224))
```

To remove the source background and replace it with the configured color, opt in:

```python
standardizer = FaceStandardizer(remove_background=True)
catalog = standardizer.run("./portraits", "./background_removed_v1")
```

Use a new or empty output directory for each run. `run()` accepts an image folder,
a single path, a list of paths, an `ImageCatalog`, or a manifest DataFrame.
Generated images enter through `generator.to_manifest()` or
`dataset.load_manifest()`; they do not need a separate standardization method.

To combine sources, create one manifest with absolute paths and unique IDs:

```python
import pandas as pd

images = pd.DataFrame([
    {"image_id": "library_a:001", "image_path": "/data/a/001.jpg",
     "source_dataset": "library_a", "is_synthetic": False},
    {"image_id": "generated:001", "image_path": "/data/generated/001.png",
     "source_dataset": "generated", "is_synthetic": True},
])
catalog = standardizer.run(images, "./mixed_v1")
```

Source labels, identity IDs, descriptions, and other metadata are retained.
Unknown real/synthetic provenance stays unknown. Requested generation attributes
remain requested attributes; processing does not verify their accuracy.

## Processing and parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `size` | `(256, 320)` | Output width and height in pixels |
| `alignment` | `"center"` | Center crop and uniform resize; `"eyes"` enables eye alignment |
| `eye_distance` | `None` (`0.25` in `"eyes"` mode) | Target eye separation as a fraction of output width; enables eye-based scaling without rotation in `"center"` mode |
| `fill_canvas` | `False` | Center mode only: increase the eye-based scale if needed to fill the output with the source rectangle; actual eye distance may exceed the target |
| `eye_height` | `0.47` | In eye alignment mode, eye midpoint height as a fraction of output height |
| `background` | `"auto"` | Estimate a solid sRGB color from each source border, or supply an RGB tuple for transparent areas, padding, and any removed background |
| `remove_background` | `False` | Enable foreground segmentation and background replacement |
| `detail_eye_distance` | `None` | Optional sampling ceiling measured in eye-distance pixels |
| `enhancer` | `None` | Optional local restoration callable; all enhanced outputs require review |
| `model_dir` | `~/.cache/pyWitnessAI/portraits` | Cache for local model files used by the chosen processing options |

EXIF orientation and embedded color profiles are applied first. Untagged RGB is
assumed sRGB. With `alignment="center"`, the crop is centered on the image and
matches the requested output aspect ratio before uniform resizing. If
`eye_distance` is supplied, the scale instead equals the target eye distance
divided by the measured input eye distance; the image center stays fixed and the
original eye-line angle is retained. With `fill_canvas=True`, this scale is bounded
below by the scale needed to cover the output. The eye-distance target is measured along the tilted eye
line, not just its horizontal projection. With
`alignment="eyes"`, the detector supplies five landmarks; an eye-based similarity
transform translates, rotates, and scales the whole image equally in both axes.
Both modes retain the original background by default. `background="auto"`
estimates one RGB color per image from the visible border of the oriented source,
ignoring fully transparent pixels. If no visible border is available, it uses
`(224, 224, 224)`. An explicit RGB tuple overrides estimation. This solid fill
approximates the background color; it does not reconstruct gradients or complex
backgrounds. Source transparency and areas outside the source image use the
resolved color. With
`remove_background=True`, the segmented foreground is composited onto that color.
A supplied reviewed mask also enables this composition for that image. Matching
geometry is applied to the image and its mask. Large reductions are prefiltered.
Color output embeds an sRGB profile; source alpha is applied once.

The optional `detail_eye_distance=48` setting downsamples the standardized portrait to
approximately 48 pixels between the eyes, then resizes it to the output canvas.
In center mode, this setting or an enhancer requires landmark measurement, but
does not change the crop geometry selected by `eye_distance`. Explicitly supplied landmarks also
provide measurements in either mode. The 48-pixel ceiling is an example, not a
universal standard. A source with fewer native
eye-distance pixels is flagged. This ceiling does not equalize optical blur,
JPEG artifacts, or noise. `None` preserves the detail supported by the output
canvas. Without an enhancer, the pipeline does not equalize skin tone, recover
missing hair, change expressions, or synthesize detail.
Clothing within the crop remains visible. When background removal is enabled,
the foreground model is not a head-only parsing model. Choose the crop and review
it for the intended experiment.

## Outputs and review

Each run writes `images/`, `masks/`, `manifest.csv`, and `standardization.json`.
The manifest preserves all rows and records input paths/hashes, alignment mode,
measured native eye distance, scale, roll, landmarks, transform matrix, and quality
flags. Per-image `background_color` is a JSON RGB array; `background_source` is
`"auto"` or `"supplied"`. The JSON records parameters, software versions, and hashes
of models actually loaded.
The recipe retains `background="auto"` when selected, rather than a resolved
per-image color, and records `alignment` and `remove_background`; per-image metadata
records `background_removed` and `mask_source`. Without segmentation, `mask_source` is
`"source_alpha"` and `background_removed` is `False`.
The recipe and per-image metadata also record `fill_canvas`; per-image
`eye_distance_clamped` indicates that the coverage requirement increased the
requested eye distance. `output_eye_distance` always reports the actual result.

`StandardizedFace` is the small return container used by `transform()`; its fields
are `image`, `mask`, `metadata`, and `baseline` (the unenhanced standardized image,
or `None` when enhancement is disabled).
The `mask` field and saved `masks/` files are retained in both modes. By default,
they represent transformed source alpha and image coverage, not person segmentation.
With background removal or a supplied reviewed mask, they represent the transformed
foreground mask combined with source alpha.

| Status | Meaning |
| --- | --- |
| `ready` | Processing completed without automated QC flags |
| `review` | An output exists, but AI enhancement, low native resolution, possible pose asymmetry, or a landmark/segmented-foreground boundary needs review |
| `error` | Reading, measurement, segmentation, enhancement, or writing failed |

Input rows already marked unavailable are retained without processing. A failed
row never falls back to its original photo as a usable standardized output.
Missing dependencies, invalid configuration, and model-download errors fail before
the batch starts. Head and foreground-boundary checks run only when an actual
segmentation mask is available. Eye-resolution, landmark, and pose checks run
only when landmarks were measured or supplied; a basic center crop does not
perform these checks. `ready` is not certification of lineup fairness or, when
background removal is enabled, a clean hair mask: inspect all pilot portraits and masks,
including those without flags.
Only explicitly reviewed rows should be promoted from `review` to `ready`, retaining
their flags and a reviewer record. Do not promote them by merely rechecking PNG
readability with `validate_images=True`.

Reload saved results with `ImageCatalog.from_manifest(pd.read_csv(".../manifest.csv"))`.
Create subsequent processing versions from the originals; reprocessing a
standardized manifest is rejected. New output paths naturally invalidate the
existing CLIP image-index fingerprint, so no second cache implementation is needed.

## Optional AI enhancement

```bash
python -m pip install -e ".[enhancement]"
```

```python
from pyWitnessAI import CodeFormerEnhancer, FaceStandardizer

standardizer = FaceStandardizer(
    enhancer=CodeFormerEnhancer(fidelity=1.0, strength=0.5),
)
catalog = standardizer.run(images, "./enhanced_v1")
print(catalog.manifest[["image_path", "baseline_image_path", "quality_flags"]])
```

`CodeFormerEnhancer` has one job: restore a softly blended face region and return
RGB pixels at the original dimensions. `FaceStandardizer` measures the original
once, then applies the same original landmarks, source alpha or foreground mask,
crop, background handling, and detail ceiling to both routes. Enhanced runs
additionally save `baseline/`;
each row's `baseline_image_path` identifies its unenhanced standardized counterpart.
Compare these paired images to isolate the effect of restoration. `transform()`
provides that counterpart as `result.baseline` without writing any files.

| Enhancement parameter | Default | Meaning |
| --- | --- | --- |
| `fidelity` | `1.0` | CodeFormer fidelity weight in [0, 1]; larger values prioritize input fidelity |
| `strength` | `0.5` | Maximum weight applied to model pixel changes in (0, 1] |
| `device` | `"cpu"` | Explicit `cpu`, `cuda`, or `cuda:<index>`; no silent device fallback |
| `model_dir` | `~/.cache/pyWitnessAI/portraits/codeformer` | Verified source files, license, and checkpoint cache |

Alignment to the model's 512-pixel canvas uses the original eyes and a similarity
transform. A feathered ellipse blends the model's pixel changes back into the source;
pixels outside that region stay unchanged before standardization. It is a fixed
blend region, not anatomical face parsing: check hairline, beard, ears and edges.
It does not relight the whole portrait or harmonize skin tones. Shared framing,
and optionally background replacement, contribute to consistent presentation.
Returning to the original canvas and applying a detail ceiling can reduce the
apparent enhancement. Only
the restoration residual is reprojected, so an unchanged model output does not
introduce an extra round of blur into the original.

Even fidelity 1 cannot guarantee unchanged identity, marks, expression, age cues,
or true recovery of lost detail. The model can invent plausible texture.
Every enhanced output receives `enhancement_requires_review`,
`enhancement_applied=True`, and `status=review`; `catalog.ready` remains empty
until explicit review. The original `is_synthetic` value describes the source
and is retained; enhancement is recorded separately. Model parameters, source
revision/checksums, checkpoint checksum, device, and software versions are saved
in `standardization.json`. Input hashes continue to identify the original files.

For lineup preparation, review every original/baseline/enhanced triplet and then
pilot source cues and selection bias. Use the same chosen processing condition
for target and fillers; clearer appearance alone is not evidence of suitability.
No similarity-score threshold automatically approves enhanced identities.

The adapter uses [official CodeFormer](https://github.com/sczhou/CodeFormer),
pinned to revision `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b` and the v0.1.0
checkpoint. Larger fidelity weights favor input fidelity according to the
[upstream testing guide](https://github.com/sczhou/CodeFormer#testing).
On first use it downloads the checkpoint (about 377 MB), three necessary inference
source files, and their [S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/LICENSE)
for non-commercial use. SHA256 checks precede loading; photos are processed locally.
The adapter isolates upstream imports and does not install BasicSR training,
torchvision, a second detector, or a second segmentation pipeline.

To use another restorer, supply `enhancer(rgb_image, *, landmarks)` returning a
same-size PIL RGB image. It receives copies of the oriented sRGB image composited
onto the chosen background and five original measured points. For transparent
inputs its pixel changes are applied before the shared foreground/alpha composition.
An optional `prepare()` method checks dependencies before a
batch writes outputs; an optional `recipe()` returns JSON-serializable model
settings for provenance. Plain callables record their qualified name, so supply
`recipe()` when experiment reproducibility requires additional configuration.

## Reviewed measurements and alternative local models

```python
result = standardizer.transform("./portrait.jpg", landmarks=points, mask=mask)
result.image.save("./preview.png")
```

`points` is a finite 5x2 array in oriented input pixel coordinates: eyes, nose,
mouth corners. `mask` is a same-size PIL `L` image, from 0 to 255. In a batch, the
optional `landmarks` column accepts lists or JSON and `mask_path` identifies a
reviewed mask. Relative mask paths use the input catalog root. Supplied measurements
bypass their corresponding model. Explicit `mask=` and `mask_path` inputs apply
their reviewed masks even when `remove_background=False`; this opts those images
into background replacement. For other models, pass `detector(image)` and/or
`segmenter(image)` callables to the constructor with these same return conventions.
The custom segmenter is used only when `remove_background=True` and no reviewed
mask was supplied.
No provider registry or extra backend hierarchy is needed.

Default landmark inference uses [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet).
Opting into background removal uses the fixed
[U2Net human segmentation model](https://github.com/xuebinqin/U-2-Net), unless a
custom segmenter or reviewed mask is supplied. Each model's weights are downloaded
and SHA256-checked only when needed; images remain local.
U2Net preprocessing/postprocessing was cross-checked against
[rembg 2.0.61](https://github.com/danielgatis/rembg/tree/v2.0.61/rembg/sessions).
The broad rembg dependency is not required. For experimental mixing, follow pixel
processing with an independent pilot of source cues and selection bias; see
[Bell et al. (2024)](https://www.nature.com/articles/s41598-024-63004-z).
