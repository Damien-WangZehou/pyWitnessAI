# Face attribute schema safety

`DEFAULT_FACE_ATTRIBUTE_SCHEMA` is a conservative, parsing-only schema. It can
extract an asserted attribute label, but every built-in `contrasts` mapping is
empty. This is intentional: demographic categories, nominal colours, face
shapes, expressions, and clothing do not have a universal single “opposite”.

The parser also fails closed:

- locally negated or uncertain matches are omitted;
- two different values in the same scalar field cause that field to abstain;
- numeric age requires `aged`, `years old`, or `y/o` context;
- race/ethnicity words must modify a person, so `black hair` and `white shirt`
  cannot create race labels;
- vague `visible hair` is not invented merely because the word `hair` occurs.

## Explicit benchmark contrasts

`FillerSelectionBenchmark` requires an explicit experimental contrast. Prefer
the `(feature, positive_label)` key so the configuration cannot silently carry
over to another positive value:

```python
from pyWitnessAI import FillerSelectionBenchmark

benchmark = FillerSelectionBenchmark(
    "a person with a long nose and thin lips",
    contrast_overrides={
        ("nose", "long nose"): "short nose",
        ("mouth", "thin lips"): "full lips",
    },
)
```

If only a preregistered subset should be benchmarked, set
`missing_contrast="skip"` explicitly. Misspelled or out-of-scope overrides are
errors by default; a deliberately reusable registry must opt into
`unused_contrast="warn"`. The stage plan, result rows, and metadata record the
resolved, skipped, and unused entries. `contrast_config_sha256` fingerprints
the serialized schema definitions and contrast design, while
`run_config_sha256` separates selector/model/generator settings into distinct
result folders.

For a binary `AttributeRetrievalBenchmark`, pass `negative_label` explicitly:

```python
from pyWitnessAI import AttributeProbeSpec

probe = AttributeProbeSpec.from_schema(
    "nose",
    "long nose",
    negative_label="short nose",
)
```

For multi-category Positive–Negative retrieval, do not reduce categories to an
arbitrary single negative. Use a versioned, candidate-pool-specific registry
that averages concrete alternative anchors and explicitly records its
validation status.
The current CFD White-Male-Neutral registry is an offline pilot registry, not a
general or independently validated production schema.

## Legacy mapping table

`LEGACY_FACE_ATTRIBUTE_SCHEMA_V1` retains the former single-negative lookup
table only for auditing a historical canonical-label contrast. It reuses the
hardened parser and is **not** an exact snapshot of the old end-to-end parser.
It is never selected by default and should not be used to design a new
experiment.
