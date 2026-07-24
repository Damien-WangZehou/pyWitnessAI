from pathlib import Path
import csv
import re

from pyWitnessAI import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    FaceAttributeDefinition,
    FillerGenerator,
    ImageGenerationBackend,
    ImageGenerationRequest,
    register_image_generation_backend,
)


class _TinyImageBackend(ImageGenerationBackend):
    provider = "tiny-test"
    default_model = "tiny-face-model"
    known_models = ("tiny-face-model",)

    def __init__(self) -> None:
        self.requests: list[ImageGenerationRequest] = []

    def generate(self, request: ImageGenerationRequest) -> None:
        self.requests.append(request)
        request.output_path.write_bytes(b"synthetic-test-image")


def test_filler_generator_parses_common_face_description(tmp_path: Path):
    generator = FillerGenerator(
        "a white dude with beard and blue eyes, thin lips",
        2,
        tmp_path,
        "gpt-image-2",
    )

    assert generator.schema.gender == "male"
    assert generator.schema.race == "White"
    assert generator.schema.facial_hair == "beard"
    assert generator.schema.eyes == "blue eyes"
    assert generator.schema.mouth == "thin lips"

    prompts = generator.generation_prompts()
    assert len(prompts) == 2
    assert "White male adult person" in prompts[0]
    assert "blue eyes" in prompts[0]
    assert "not a real person" in prompts[0]


def test_filler_generator_treats_clip_model_name_as_retrieval_model(tmp_path: Path):
    generator = FillerGenerator(
        "a white dude with beard and blue eyes",
        1,
        tmp_path,
        "clip-ViT-B-16",
    )

    assert generator.clip_model == "clip-ViT-B-16"
    assert generator.model == "gpt-image-2"


def test_filler_generator_parses_negated_visible_teeth(tmp_path: Path):
    generator = FillerGenerator("a male person with no visible teeth", 1, tmp_path)

    assert generator.schema.teeth == "no visible teeth"
    assert "no visible teeth" in generator.generation_prompts()[0]


def test_filler_generator_uses_age_in_subject_phrase(tmp_path: Path):
    generator = FillerGenerator("a middle-aged white male with blue eyes", 1, tmp_path)

    prompt = generator.generation_prompts()[0]
    assert generator.schema.age == "middle-aged adult"
    assert "middle-aged adult White male person" in prompt
    assert "with middle-aged adult" not in prompt


def test_filler_generator_preserves_compound_eyebrow_and_hawk_nose_attributes(tmp_path: Path):
    generator = FillerGenerator(
        "a male with long curly hair, brown bushy eyebrows, and an aquiline nose",
        1,
        tmp_path,
    )

    assert generator.schema.hair == "long hair"
    assert generator.schema.hair_texture == "curly hair"
    assert generator.schema.eyebrow_color == "brown eyebrows"
    assert generator.schema.eyebrows == "bushy eyebrows"
    assert generator.schema.nose == "hawk nose"
    assert "long curly hair" in generator.generation_prompts()[0]
    assert "brown bushy eyebrows, hawk nose" in generator.generation_prompts()[0]


def test_filler_generator_accepts_extended_attribute_schema(tmp_path: Path):
    schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
        FaceAttributeDefinition(
            "skin_marking",
            65,
            patterns=((r"\bfreckled skin\b", "freckled skin"),),
            contrasts={"freckled skin": "unfreckled skin"},
        )
    )
    generator = FillerGenerator(
        "a male with freckled skin",
        1,
        tmp_path,
        attribute_schema=schema,
    )

    assert generator.schema.attribute_values()["skin_marking"] == "freckled skin"
    assert "freckled skin" in generator.generation_prompts()[0]


def test_filler_generator_dry_run_does_not_write_images(tmp_path: Path):
    generator = FillerGenerator("a male person", 1, tmp_path)

    result = generator.generate(dry_run=True)

    assert result == []
    assert not list(tmp_path.glob("*.png"))


def test_filler_generator_accepts_backend_instance(tmp_path: Path):
    backend = _TinyImageBackend()
    generator = FillerGenerator("a male person", 1, tmp_path, backend=backend)

    results = generator.generate()

    assert len(results) == 1
    assert backend.requests[0].model == "tiny-face-model"
    assert backend.requests[0].prompt.startswith("Photorealistic synthetic fictional adult face portrait")
    assert Path(results[0].path).read_bytes() == b"synthetic-test-image"
    assert results[0].provider == "tiny-test"


def test_filler_generator_batch_naming_records_image_ids(tmp_path: Path):
    backend = _TinyImageBackend()
    generator = FillerGenerator(
        "a male person",
        2,
        tmp_path,
        backend=backend,
        naming_strategy="batch",
    )

    results = generator.generate()

    assert len(results) == 2
    assert results[0].batch_id.startswith("batch_")
    assert re.match(r"gf_\d{8}_\d{6}_[0-9a-f]{8}_0001\.png", Path(results[0].path).name)
    assert results[0].image_id == Path(results[0].path).stem

    with (tmp_path / "generation_manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["image_id"] == results[0].image_id
    assert rows[0]["batch_id"] == results[0].batch_id
    assert rows[0]["prompt_hash"]


def test_filler_generator_accepts_registered_backend(tmp_path: Path):
    register_image_generation_backend("tiny-test", _TinyImageBackend, overwrite=True)
    generator = FillerGenerator("a male person", 1, tmp_path, provider="tiny-test")

    results = generator.generate()

    assert len(results) == 1
    assert results[0].model == "tiny-face-model"
    assert results[0].provider == "tiny-test"
    assert (tmp_path / "generation_manifest.csv").exists()


def test_filler_generator_rejects_underage_description(tmp_path: Path):
    try:
        FillerGenerator("a 16 years old male person", 1, tmp_path)
    except ValueError as exc:
        assert "adult" in str(exc)
    else:
        raise AssertionError("Expected an underage description to be rejected.")
