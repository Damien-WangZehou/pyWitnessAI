from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from pyWitnessAI.AttributeRetrievalBenchmark import AttributeProbeSpec, AttributeRetrievalBenchmark
from pyWitnessAI.Description import DescriptionAttribute, ParsedDescription, RawDescriptionParser, SchemaDescriptionParser
from pyWitnessAI.FaceAttributeSchema import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    FaceAttributeDefinition,
    FaceAttributeSchema,
    FaceDescriptionSchema,
)
from pyWitnessAI.FaceGenerator import FaceGenerator
from pyWitnessAI.GeneratedFaceDataset import GeneratedFaceDataset, SCHEMA_COLUMNS
from pyWitnessAI.face_generation_backend import (
    ImageGenerationBackend,
    ImageGenerationRequest,
    register_image_generation_backend,
)


class TinyFaceBackend(ImageGenerationBackend):
    provider = "tiny-face-test"
    default_model = "tiny-portrait-v1"
    supports_reference_image = True

    def __init__(self, *, fail_at: int | None = None, write_image: bool = True, partial_failure: bool = False):
        self.requests: list[ImageGenerationRequest] = []
        self.fail_at = fail_at
        self.write_image = write_image
        self.partial_failure = partial_failure

    def generate(self, request: ImageGenerationRequest) -> None:
        self.requests.append(request)
        if request.index == self.fail_at:
            if self.partial_failure:
                request.output_path.write_bytes(b"incomplete-image")
            raise RuntimeError("simulated provider failure")
        if self.write_image:
            Image.new("RGB", (8, 8), (request.index, 80, 120)).save(request.output_path)


@pytest.fixture
def reference_image(tmp_path: Path) -> Path:
    path = tmp_path / "reference.png"
    Image.new("RGB", (10, 10), (90, 120, 160)).save(path)
    return path


def custom_schema() -> FaceAttributeSchema:
    return FaceAttributeSchema(
        [
            FaceAttributeDefinition(
                "skin_marking",
                1,
                patterns=((r"\bfreckles\b", "freckled skin"),),
                sample_values=("freckled skin", "clear skin"),
            )
        ],
        name="test-skin-schema",
        version="test-1",
    )


@pytest.mark.parametrize(
    ("description", "expected_attributes"),
    [
        (
            "a white dude with beard and blue eyes, thin lips",
            {"gender": "male", "race": "White", "facial_hair": "beard", "eyes": "blue eyes", "mouth": "thin lips"},
        ),
        (
            "a male person with no visible teeth",
            {"gender": "male", "teeth": "no visible teeth"},
        ),
        (
            "a middle-aged white male with blue eyes",
            {"gender": "male", "race": "White", "age": "middle-aged adult", "eyes": "blue eyes"},
        ),
        (
            "a male with long curly hair, brown bushy eyebrows, and an aquiline nose",
            {"gender": "male", "hair": "long hair", "hair_texture": "curly hair", "eyebrow_color": "brown eyebrows", "eyebrows": "bushy eyebrows", "nose": "aquiline nose"},
        ),
    ],
)
def test_default_schema_preserves_requested_attributes_in_portrait_prompts(
    tmp_path: Path, description: str, expected_attributes
):
    generator = FaceGenerator(TinyFaceBackend(), tmp_path / "faces", description=description)
    result = generator.generate()[0]

    assert result.attributes == expected_attributes
    assert generator.parsed_description.model_text in result.prompt
    assert all(value in result.prompt for value in expected_attributes.values())
    assert result.prompt.startswith("Make a photo of an under average-looking man")


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("use_reference", [False, True])
def test_witness_lineup_uses_simple_single_portrait_prompts(
    tmp_path: Path, reference_image: Path, n: int, use_reference: bool,
):
    description = (
        "1) Dark hair (dark brown/black) / "
        "2) Medium length hair sort of parted down the middle so that it was out of his face / "
        "3) prominent nose (big nose) / 4) Dark eyes / 5) He looked nervous  / "
        "6) No facial hair / 7) very stern  / "
        "8) hard to estimate height since there was nothing to base it off of looking from a computer screen / "
        "9) Dark coat on / "
        "10) kept looking around to make sure no one was paying attention to what he was doing"
    )
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", n, description=description,
        parser=RawDescriptionParser(lowercase=False),
        reference_image=reference_image if use_reference else None,
    )
    core = " ".join(description.split())
    prefix = (
        "Make a photo of a man based on this image, with the following features: "
        if use_reference else
        "Make a photo of an under average-looking man, with facial features "
        "and hair style completely different from each other, "
        "but matching the following description: "
    )
    expected = (
        prefix + core + ". Make it an image portrait, only from shoulders above. "
        "Make the background gray. "
        "The people should be looking at the camera with frontal face. "
        "Do not emphasize or depict body actions, hand placement, or clothing interactions."
    )
    assert generator.generation_prompts() == [expected] * n
    results = generator.generate()

    assert len(results) == len(backend.requests) == n
    assert [request.index for request in backend.requests] == list(range(1, n + 1))
    assert all(request.total == n and request.prompt == expected for request in backend.requests)
    assert all(request.reference_image == (reference_image.resolve() if use_reference else None)
               for request in backend.requests)
    assert all(result.attributes == {} and result.description == core for result in results)


def test_registered_backend_name_creates_images_using_its_default_model(tmp_path: Path):
    register_image_generation_backend("portrait-registry-test", TinyFaceBackend, overwrite=True)
    generator = FaceGenerator("portrait-registry-test", tmp_path / "faces", description="blue eyes")

    result = generator.generate()[0]

    assert isinstance(generator.backend, TinyFaceBackend)
    assert generator.backend.requests[0].model == TinyFaceBackend.default_model
    assert result.provider == TinyFaceBackend.provider
    assert result.model == TinyFaceBackend.default_model
    assert Path(result.path).is_file()
    assert (generator.output_dir / "manifest.csv").is_file()


@pytest.mark.parametrize(
    ("description", "use_reference", "expected_mode"),
    [
        (None, False, "random_description"),
        ("a male with blue eyes", False, "description"),
        (None, True, "reference"),
        ("a male with blue eyes", True, "description_reference"),
    ],
)
def test_four_input_modes_dispatch_n_images_and_record_conditions(
    tmp_path: Path, reference_image: Path, description, use_reference, expected_mode
):
    backend = TinyFaceBackend()
    output_dir = tmp_path / "generated"
    generator = FaceGenerator(
        backend,
        output_dir,
        3,
        description=description,
        reference_image=reference_image if use_reference else None,
        similarity=37,
        seed=91,
    )

    prompts = generator.generation_prompts()
    assert len(prompts) == 3
    assert generator.generation_mode == expected_mode
    assert backend.requests == []
    assert not output_dir.exists()

    results = generator()

    assert len(results) == len(backend.requests) == 3
    assert [request.index for request in backend.requests] == [1, 2, 3]
    assert all(request.total == 3 for request in backend.requests)
    assert [request.prompt for request in backend.requests] == prompts
    assert generator.image_paths() == [Path(result.path) for result in results]
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert len(manifest) == 3
    assert manifest.image_id.is_unique
    assert set(manifest.image_id) == {result.image_id for result in results}
    assert set(manifest.batch_id) == {generator.batch_id}
    assert set(manifest.generation_mode) == {expected_mode}
    assert manifest.attributes_verified.eq(False).all()
    assert set(manifest.attribute_schema_name) == {DEFAULT_FACE_ATTRIBUTE_SCHEMA.name}
    for result in results:
        with Image.open(result.path) as image:
            assert image.size == (8, 8)
        row = manifest.loc[manifest.image_id == result.image_id].iloc[0]
        assert row["prompt"] == result.prompt
        assert row["prompt_hash"]
        assert Path(row["image_path"]).is_absolute()
        for attribute, value in result.attributes.items():
            column = attribute if attribute in SCHEMA_COLUMNS else f"attribute__{attribute}"
            assert str(row[column]) == value
    if description is not None:
        assert manifest.gender.eq("male").all()
        assert manifest.eyes.eq("blue eyes").all()
        assert manifest.original_description.eq(description).all()
        assert manifest.attribute_label_source.eq("requested").all()
    if use_reference:
        assert all(request.reference_image == reference_image.resolve() for request in backend.requests)
        assert manifest.reference_image_path.eq(str(reference_image.resolve())).all()
        assert manifest.requested_similarity.eq(37).all()
        assert all(("37/100" in prompt) is (description is None) for prompt in prompts)
    else:
        assert all(request.reference_image is None for request in backend.requests)
        assert manifest.reference_image_path.isna().all()
        assert manifest.requested_similarity.isna().all()


def test_reference_only_does_not_invent_image_attribute_labels(tmp_path: Path, reference_image: Path):
    generator = FaceGenerator(TinyFaceBackend(), tmp_path / "faces", reference_image=reference_image)
    result = generator.generate()[0]

    assert result.attributes == {}
    assert result.prompt.startswith("Make a photo of a person based on this image.")
    row = generator.to_manifest().iloc[0]
    assert row["attribute_label_source"] == "unspecified"
    assert row["attributes_verified"] == False
    assert all(pd.isna(row[column]) or row[column] == "" for column in SCHEMA_COLUMNS)


def test_unconditional_without_manifest_never_samples_or_parses_schema(tmp_path: Path, monkeypatch):
    schema = custom_schema()

    def unexpected_schema_use(*args, **kwargs):
        pytest.fail("Unconditional unrecorded generation must leave attributes to the image provider.")

    monkeypatch.setattr(schema, "sample", unexpected_schema_use)
    monkeypatch.setattr(schema, "parse", unexpected_schema_use)
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", 2, write_manifest=False, attribute_schema=schema
    )
    results = generator.generate()

    assert generator.generation_mode == "random"
    assert all(result.attributes == {} and result.description == "" for result in results)
    assert all("Randomly choose" in request.prompt for request in backend.requests)
    assert sorted(path.suffix for path in generator.output_dir.iterdir()) == [".png", ".png"]


def test_recorded_random_descriptions_are_explicit_and_seed_reproducible(tmp_path: Path):
    first = FaceGenerator(TinyFaceBackend(), tmp_path / "first", 4, seed=42)
    second = FaceGenerator(TinyFaceBackend(), tmp_path / "second", 4, seed=42)

    assert first.generation_prompts() == second.generation_prompts()
    assert not first.output_dir.exists()
    first_results = first.generate()
    second_results = second.generate()

    assert [result.attributes for result in first_results] == [result.attributes for result in second_results]
    assert all(result.attributes and result.description for result in first_results)
    assert {result.image_id for result in first_results}.isdisjoint(result.image_id for result in second_results)
    assert all("Make the background gray." in result.prompt for result in first_results)
    assert all("frontal face" in result.prompt for result in first_results)
    manifest = first.to_manifest()
    assert manifest.attribute_label_source.eq("requested").all()
    assert manifest.description_seed.eq("42").all()


@pytest.mark.parametrize("use_reference", [False, True])
def test_disabling_manifest_preserves_explicit_description_and_reference(
    tmp_path: Path, reference_image: Path, use_reference: bool
):
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend,
        tmp_path / "faces",
        description="blue eyes",
        reference_image=reference_image if use_reference else None,
        write_manifest=False,
    )
    result = generator.generate()[0]

    assert result.attributes == {"eyes": "blue eyes"}
    assert "blue eyes" in backend.requests[0].prompt
    assert backend.requests[0].reference_image == (reference_image.resolve() if use_reference else None)
    assert not (generator.output_dir / "manifest.csv").exists()


@pytest.mark.parametrize("schema_from_parser", [False, True])
def test_custom_schema_controls_parsing_and_attribute_column_names(tmp_path: Path, schema_from_parser):
    schema = custom_schema()
    options = {"parser": SchemaDescriptionParser(schema)} if schema_from_parser else {"attribute_schema": schema}
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", 2, description="freckles", **options,
    )
    results = generator.generate()
    manifest = pd.read_csv(generator.output_dir / "manifest.csv")

    assert all(result.attributes == {"skin_marking": "freckled skin"} for result in results)
    assert all("freckled skin" in result.prompt for result in results)
    assert manifest["attribute__skin_marking"].eq("freckled skin").all()
    assert manifest.attribute_schema_name.eq("test-skin-schema").all()
    assert manifest.attribute_schema_version.eq("test-1").all()
    dataset = GeneratedFaceDataset(generator.output_dir)
    matches = dataset.find("freckles", parser=generator.parser, n=1)
    assert matches.image_id.iloc[0] in manifest.image_id.tolist()
    assert matches["attribute__skin_marking"].tolist() == ["freckled skin"]
    with pytest.raises(ValueError, match="No schema attributes"):
        dataset.find("unrecognised feature", parser=generator.parser)
    assert len(backend.requests) == 2


def test_random_generation_uses_only_custom_schema_sample_values(tmp_path: Path):
    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces", 4, attribute_schema=custom_schema(), seed=17,
    )
    results = generator.generate()

    for result in results:
        assert set(result.attributes) == {"skin_marking"}
        assert result.attributes["skin_marking"] in {"freckled skin", "clear skin"}
        assert result.attributes["skin_marking"] in result.prompt


@pytest.mark.parametrize("model_text", ["freckled skin, a silver scarf", "a portrait with a silver scarf"])
def test_custom_parser_output_controls_prompt_and_requested_attributes(tmp_path: Path, model_text: str):
    class CustomParser:
        name = "test-parser"

        def parse(self, text: str) -> ParsedDescription:
            return ParsedDescription(
                text, text, model_text,
                (DescriptionAttribute("freckled skin", "skin_marking", "skin_marking", "freckled skin"),),
            )

    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces", description="use my description policy",
        parser=CustomParser(), attribute_schema=custom_schema(),
    )
    result = generator.generate()[0]

    assert model_text in result.prompt
    assert "freckled skin" in result.prompt
    assert "use my description policy" not in result.prompt
    assert result.attributes == {"skin_marking": "freckled skin"}
    manifest = pd.read_csv(generator.output_dir / "manifest.csv")
    assert manifest.iloc[0]["parser"] == "test-parser"
    assert manifest.iloc[0]["attribute__skin_marking"] == "freckled skin"
    assert manifest.iloc[0]["attribute_label_source"] == "requested"


def test_raw_parser_keeps_description_without_reparsing_schema_labels(tmp_path: Path, monkeypatch):
    schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend([], name="raw-policy-test")

    def unexpected_parse(*args, **kwargs):
        pytest.fail("The generator must not run schema parsing after the chosen raw parser.")

    monkeypatch.setattr(schema, "parse", unexpected_parse)
    description = "  BLUE  eyes\nand a silver scarf  "
    model_text = "blue eyes and a silver scarf"
    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces", description=description,
        parser="raw", attribute_schema=schema,
    )
    result = generator.generate()[0]
    manifest = pd.read_csv(generator.output_dir / "manifest.csv")

    assert model_text in result.prompt
    assert result.description == model_text
    assert result.attributes == {}
    assert manifest.iloc[0]["original_description"] == description
    assert manifest.iloc[0]["attribute_label_source"] == "unspecified"
    assert manifest[list(SCHEMA_COLUMNS)].isna().all().all()


def test_preparsed_explicit_attributes_bypass_parser_and_preserve_experimental_values(tmp_path: Path):
    class UnexpectedParser:
        def parse(self, text: str) -> ParsedDescription:
            pytest.fail("An already parsed description must not be parsed again.")

    label = "opaque experimental eye label"
    parsed = ParsedDescription.from_attributes({"eyes": label}, original_text="source wording")
    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces", description=parsed, parser=UnexpectedParser(),
    )
    result = generator.generate()[0]
    row = pd.read_csv(generator.output_dir / "manifest.csv").iloc[0]

    assert generator.parsed_description is parsed
    assert label in result.prompt
    assert result.attributes == {"eyes": label}
    assert row["eyes"] == label
    assert row["original_description"] == "source wording"
    assert row["parser"] == "preparsed"


def test_preparsed_raw_text_survives_the_default_schema_policy(tmp_path: Path):
    parsed = RawDescriptionParser().parse("unsupported generic words")
    generator = FaceGenerator(TinyFaceBackend(), tmp_path / "faces", description=parsed)
    result = generator.generate()[0]

    assert parsed.model_text in result.prompt
    assert result.description == parsed.model_text
    assert result.attributes == {}
    assert generator.to_manifest().iloc[0]["parser"] == "preparsed"


def test_schema_parser_rejects_a_different_explicit_attribute_schema(tmp_path: Path):
    backend = TinyFaceBackend()
    parser = SchemaDescriptionParser(custom_schema())

    with pytest.raises(ValueError, match="same attribute_schema"):
        FaceGenerator(
            backend, tmp_path / "faces", description="freckles", parser=parser,
            attribute_schema=DEFAULT_FACE_ATTRIBUTE_SCHEMA,
        )

    assert backend.requests == []
    assert not (tmp_path / "faces").exists()


def test_custom_parser_conflicting_uncertain_and_rejected_attributes_are_not_recorded(tmp_path: Path):
    class EvidenceParser:
        def parse(self, text: str) -> ParsedDescription:
            return ParsedDescription(
                text, text, "an adult portrait",
                (
                    DescriptionAttribute("blue eyes", "eyes", "eyes", "blue eyes"),
                    DescriptionAttribute("brown eyes", "eyes", "eyes", "brown eyes"),
                    DescriptionAttribute("maybe short hair", "hair", "hair", "short hair", uncertain=True),
                    DescriptionAttribute("wide nose", "nose", "nose", "wide nose", accepted=False),
                    DescriptionAttribute("bushy eyebrows", "eyebrows", "eyebrows", "bushy eyebrows"),
                ),
            )

    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces", description="use these observations", parser=EvidenceParser(),
    )
    result = generator.generate()[0]
    manifest = pd.read_csv(generator.output_dir / "manifest.csv")

    assert result.attributes == {"eyebrows": "bushy eyebrows"}
    assert "bushy eyebrows" in result.prompt
    assert all(value not in result.prompt for value in ("blue eyes", "brown eyes", "short hair", "wide nose"))
    assert manifest.eyebrows.tolist() == ["bushy eyebrows"]
    assert manifest[["eyes", "hair", "nose"]].isna().all().all()


@pytest.mark.parametrize(("parser", "retain_uncertain"), [("raw", True), ("lenient", True), ("strict", False)])
def test_description_policies_preserve_prompt_choice_without_asserting_uncertain_labels(
    tmp_path: Path, parser: str, retain_uncertain: bool
):
    generator = FaceGenerator(
        TinyFaceBackend(), tmp_path / "faces",
        description="short hair, a star-shaped birthmark, maybe blue eyes", parser=parser,
    )
    result = generator.generate()[0]

    assert "short hair" in result.prompt
    assert ("maybe blue eyes" in result.prompt) is retain_uncertain
    assert "eyes" not in result.attributes
    if parser in {"raw", "lenient"}:
        assert "star-shaped birthmark" in result.prompt
    if parser == "strict":
        assert any(item.uncertain and not item.accepted for item in generator.parsed_description.attributes)


def test_strict_parser_with_only_uncertain_details_does_not_dispatch_an_empty_prompt(tmp_path: Path):
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", description="maybe blue eyes", parser="strict",
    )

    with pytest.raises(ValueError, match="empty model input"):
        generator.generate()

    assert backend.requests == []
    assert generator.parsed_description.original_text == "maybe blue eyes"
    assert not generator.output_dir.exists()


def test_reference_capability_is_checked_before_output_or_backend_calls(tmp_path: Path, reference_image: Path):
    backend = TinyFaceBackend()
    backend.supports_reference_image = False
    generator = FaceGenerator(backend, tmp_path / "faces", reference_image=reference_image)

    with pytest.raises(ValueError, match="does not support reference"):
        generator.generate()

    assert backend.requests == []
    assert not generator.output_dir.exists()


@pytest.mark.parametrize("similarity", [0, 100])
def test_reference_similarity_endpoints_are_preserved(tmp_path: Path, reference_image: Path, similarity):
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", reference_image=reference_image, similarity=similarity
    )
    generator.generate()

    assert backend.requests[0].similarity == similarity
    assert f"{similarity}/100" in backend.requests[0].prompt
    assert generator.to_manifest().iloc[0]["requested_similarity"] == similarity


@pytest.mark.parametrize("similarity", [-1, 101, float("nan"), float("inf"), -float("inf"), True, None, "invalid"])
def test_invalid_similarity_is_rejected_without_writing(tmp_path: Path, similarity):
    backend = TinyFaceBackend()

    with pytest.raises(ValueError, match="similarity"):
        FaceGenerator(backend, tmp_path / "faces", similarity=similarity)

    assert backend.requests == []
    assert not (tmp_path / "faces").exists()


def test_missing_reference_is_rejected_before_generation(tmp_path: Path):
    backend = TinyFaceBackend()
    generator = FaceGenerator(backend, tmp_path / "faces", reference_image=tmp_path / "missing.png")

    with pytest.raises(FileNotFoundError, match="Reference image"):
        generator.generate()

    assert backend.requests == []
    assert not generator.output_dir.exists()


def test_new_batches_append_unique_rows_and_repeated_batch_reuses_images(tmp_path: Path):
    output_dir = tmp_path / "faces"
    first_backend, second_backend = TinyFaceBackend(), TinyFaceBackend()
    first = FaceGenerator(first_backend, output_dir, 2, description="blue eyes")
    second = FaceGenerator(second_backend, output_dir, 1, description="brown eyes")
    first_results = first.generate()
    second_results = second.generate()
    first.generate()
    manifest = pd.read_csv(output_dir / "manifest.csv")

    assert len(manifest) == 3
    assert manifest.image_id.is_unique
    assert manifest.batch_id.nunique() == 2
    assert {result.image_id for result in first_results}.isdisjoint(result.image_id for result in second_results)
    assert manifest.loc[manifest.batch_id == first.batch_id, "eyes"].eq("blue eyes").all()
    assert manifest.loc[manifest.batch_id == second.batch_id, "eyes"].eq("brown eyes").all()
    assert len(first_backend.requests) == 2
    assert len(second_backend.requests) == 1


def test_partial_failure_keeps_only_successes_and_retry_regenerates_incomplete_image(tmp_path: Path):
    backend = TinyFaceBackend(fail_at=2, partial_failure=True)
    generator = FaceGenerator(backend, tmp_path / "faces", 3, description="blue eyes")

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        generator.generate()

    manifest = pd.read_csv(generator.output_dir / "manifest.csv")
    assert len(manifest) == len(generator.results) == 1
    assert manifest.image_id.tolist() == [generator.results[0].image_id]
    assert manifest.batch_index.tolist() == [1]
    assert [request.index for request in backend.requests] == [1, 2]
    assert all(Path(path).is_file() for path in manifest.image_path)
    assert not backend.requests[-1].output_path.exists()

    backend.fail_at = None
    results = generator.generate()
    recovered = pd.read_csv(generator.output_dir / "manifest.csv")
    assert len(results) == len(recovered) == 3
    assert recovered.image_id.is_unique
    assert [request.index for request in backend.requests] == [1, 2, 2, 3]
    for result in results:
        with Image.open(result.path) as image:
            assert image.size == (8, 8)


def test_backend_that_writes_no_image_cannot_create_a_success_manifest_row(tmp_path: Path):
    generator = FaceGenerator(
        TinyFaceBackend(write_image=False), tmp_path / "faces", description="blue eyes"
    )

    with pytest.raises(RuntimeError, match="output image"):
        generator.generate()

    assert generator.results == []
    assert not (generator.output_dir / "manifest.csv").exists()


def test_dry_run_is_a_preview_without_files_or_backend_generation(tmp_path: Path, reference_image: Path):
    backend = TinyFaceBackend()
    generator = FaceGenerator(
        backend, tmp_path / "faces", 2, description="blue eyes", reference_image=reference_image
    )

    assert generator.generate(dry_run=True) == []
    assert backend.requests == []
    assert generator.results == []
    assert not generator.output_dir.exists()


def test_manifest_supports_dataset_filtering_and_attribute_benchmark_without_ai(tmp_path: Path):
    output_dir = tmp_path / "faces"
    FaceGenerator(TinyFaceBackend(), output_dir, 2, description="a male with blue eyes").generate()
    FaceGenerator(TinyFaceBackend(), output_dir, 2, description="a male with brown eyes").generate()
    dataset = GeneratedFaceDataset(output_dir)
    matches = dataset.select(FaceDescriptionSchema.from_attributes("", {"eyes": "blue eyes"}), match="contains")

    assert len(matches) == 2
    assert matches.eyes.eq("blue eyes").all()
    assert matches.attribute_label_source.eq("requested").all()
    assert set(dataset.find("blue eyes").image_id) == set(matches.image_id)
    with pytest.raises(ValueError, match="No schema attributes"):
        dataset.find("an unrecognised trait")

    def metadata_selector(manifest: pd.DataFrame, probe: AttributeProbeSpec) -> pd.DataFrame:
        return pd.DataFrame({
            "image_id": manifest.image_id,
            "selector_score": manifest.eyes.eq(probe.positive_label).astype(float),
        })

    benchmark = AttributeRetrievalBenchmark(
        output_dir / "manifest.csv",
        AttributeProbeSpec("eyes", "blue eyes", "brown eyes", ("blue eyes",)),
        label_column="eyes", selector=metadata_selector, results_root=tmp_path / "benchmark",
    )
    ranked = benchmark.run(display=False)

    assert len(ranked) == 4
    assert ranked.head(2).eyes.eq("blue eyes").all()
    assert ranked.attribute_label_source.eq("requested").all()
    assert ranked.attributes_verified.eq(False).all()
    assert benchmark.statistics(display=False).iloc[0]["precision_at_k"] == 1.0
