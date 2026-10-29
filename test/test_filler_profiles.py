"""Offline coverage of explicit, per-filler supplemental design requirements."""

from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from pyWitnessAI.Description import RawDescriptionParser
from pyWitnessAI.FaceGenerator import FaceGenerator
from pyWitnessAI.GeneratedFaceDataset import SCHEMA_COLUMNS
from pyWitnessAI.LineupSession import create_lineup
from pyWitnessAI.face_generation_backend import ImageGenerationBackend


WITNESS_DESCRIPTION = (
    "1) Dark hair (dark brown/black) / "
    "2) Medium length hair sort of parted down the middle so that it was out of his face / "
    "3) prominent nose (big nose) / 4) Dark eyes / 5) He looked nervous / "
    "6) No facial hair / 7) very stern / "
    "8) hard to estimate height since there was nothing to base it off of looking from a computer screen / "
    "9) Dark coat on / "
    "10) kept looking around to make sure no one was paying attention to what he was doing"
)
PROFILES = (
    "Oval face, narrow jaw, thin lips",
    "Round face, broad jaw, full lips",
    "Angular face, pointed chin, medium lips",
    "Square face, wide chin, small ears",
    "Long face, rounded chin, large ears",
    "Heart-shaped face, narrow chin, broad mouth",
    "Diamond-shaped face, prominent cheekbones, small mouth",
)


class ProfileBackend(ImageGenerationBackend):
    provider = "filler-profile-test"
    default_model = "tiny-profile-portrait"
    supports_reference_image = True

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        Image.new("RGB", (8, 10), (request.index, 90, 130)).save(request.output_path)


@pytest.fixture
def backend():
    return ProfileBackend()


@pytest.fixture
def portrait(tmp_path):
    path = tmp_path / "existing.png"
    Image.new("RGB", (8, 10), (180, 50, 80)).save(path)
    return path.resolve()


@pytest.mark.parametrize("use_reference", [False, True])
def test_profiles_preserve_full_raw_description_and_map_to_images_and_manifest(
    tmp_path, backend, portrait, use_reference,
):
    profiles = PROFILES[:3]
    output = tmp_path / "generated"
    generator = FaceGenerator(
        backend, output, n=len(profiles), description=WITNESS_DESCRIPTION,
        parser=RawDescriptionParser(lowercase=False), filler_profiles=profiles,
        reference_image=portrait if use_reference else None,
    )
    prompts = generator.generation_prompts()
    assert backend.requests == []
    assert not output.exists()

    results = generator.generate()
    assert len(results) == len(backend.requests) == len(profiles)
    assert len(set(prompts)) == len(profiles)
    records = pd.read_csv(output / "manifest.csv").set_index("image_id")
    for result, request, profile in zip(results, backend.requests, profiles):
        assert request.prompt == result.prompt == prompts[result.index - 1]
        assert request.total == len(profiles)
        assert request.reference_image == (portrait if use_reference else None)
        assert result.prompt.count(WITNESS_DESCRIPTION) == 1
        assert result.prompt.count(profile) == 1
        assert all(other not in result.prompt for other in profiles if other != profile)
        assert "Additional design requirements" in result.prompt
        assert "priority" in result.prompt.lower()
        assert "other people in the lineup" not in result.prompt
        assert result.filler_profile == profile
        assert result.description == WITNESS_DESCRIPTION
        assert result.attributes == {}
        assert Path(result.path).is_file()
        row = records.loc[result.image_id]
        assert row["filler_profile"] == profile
        assert row["original_description"] == WITNESS_DESCRIPTION
        assert row["verbal_description"] == WITNESS_DESCRIPTION
        assert row["prompt"] == result.prompt
        assert row["attributes_verified"] == False
    # Supplemental traits remain separate from witness-derived schema labels.
    assert records[list(SCHEMA_COLUMNS)].isna().all().all()
    assert generator.to_manifest().filler_profile.tolist() == list(profiles)


def test_profile_input_is_copied_before_caller_mutation(tmp_path, backend):
    profiles = list(PROFILES[:2])
    generator = FaceGenerator(
        backend, tmp_path / "generated", n=2, description="dark hair",
        parser="raw", filler_profiles=profiles,
    )
    profiles[0] = "Caller changed this requirement"
    profiles.append("Caller added another requirement")

    assert generator.filler_profiles == PROFILES[:2]
    prompts = generator.generation_prompts()
    assert len(prompts) == 2
    assert all(profile in prompt for profile, prompt in zip(PROFILES, prompts))
    assert all("Caller" not in prompt for prompt in prompts)


def test_omitting_profiles_preserves_existing_prompt_and_empty_profile_metadata(tmp_path, backend):
    options = dict(n=2, description="a man with dark hair", parser="raw", seed=42)
    ordinary = FaceGenerator(backend, tmp_path / "ordinary", **options)
    explicit_none = FaceGenerator(backend, tmp_path / "explicit_none", filler_profiles=None, **options)

    assert ordinary.generation_prompts() == explicit_none.generation_prompts()
    assert all("other people in the lineup" in prompt for prompt in ordinary.generation_prompts())
    assert all("Additional design requirements" not in prompt for prompt in ordinary.generation_prompts())
    results = ordinary.generate()
    assert all(result.filler_profile == "" for result in results)
    assert ordinary.to_manifest().filler_profile.eq("").all()


def test_profiles_do_not_override_the_selected_description_parser(tmp_path, backend):
    generator = FaceGenerator(
        backend, tmp_path / "generated", description="short hair, maybe blue eyes",
        parser="strict", filler_profiles=["Oval face and thin lips"],
    )
    result = generator.generate()[0]

    assert result.description == generator.parsed_description.model_text
    assert "short hair" in result.prompt
    assert "maybe blue eyes" not in result.prompt
    assert "eyes" not in result.attributes
    assert "face_shape" not in result.attributes
    assert "mouth" not in result.attributes


@pytest.mark.parametrize("profiles", [
    [], ["", "round face"], ["   ", "round face"],
    ["oval face", "oval face"], ["oval face"],
    ["oval face", "round face", "square face"],
    "oval face", b"oval face", ["oval face", None], ["oval face", 3],
])
def test_invalid_profiles_fail_before_generation_or_files(tmp_path, backend, profiles):
    output = tmp_path / "generated"
    with pytest.raises((TypeError, ValueError)):
        FaceGenerator(
            backend, output, n=2, description="dark hair", parser="raw",
            filler_profiles=profiles,
        ).generate()
    assert backend.requests == []
    assert not output.exists()


@pytest.mark.parametrize("use_reference", [False, True])
def test_profiles_require_an_explicit_description(tmp_path, backend, portrait, use_reference):
    output = tmp_path / "generated"
    with pytest.raises((TypeError, ValueError)):
        FaceGenerator(
            backend, output, filler_profiles=["Oval face"],
            reference_image=portrait if use_reference else None,
        ).generate()
    assert backend.requests == []
    assert not output.exists()


@pytest.mark.parametrize("size", [4, 7])
@pytest.mark.parametrize("target", ["present", "absent"])
def test_create_lineup_uses_one_profile_per_required_filler(tmp_path, backend, portrait, size, target):
    needed = size - int(target == "present")
    profiles = PROFILES[:needed]
    output = tmp_path / "generated"
    lineup = create_lineup(
        suspect=portrait if target == "present" else None, target=target, size=size,
        description=WITNESS_DESCRIPTION, filler_profiles=profiles,
        generator_backend=backend, output_dir=output, seed=42,
    )

    assert len(lineup) == size
    assert lineup.target == target
    assert len(backend.requests) == needed
    assert all(request.total == needed for request in backend.requests)
    assert all(profile in request.prompt for profile, request in zip(profiles, backend.requests))
    assert all(WITNESS_DESCRIPTION.lower() in request.prompt for request in backend.requests)
    assert list(lineup.roles.values()).count("filler") == needed
    assert (lineup.suspect_position is not None) == (target == "present")
    assert (str(portrait) in lineup.source_paths.values()) == (target == "present")
    records = pd.read_csv(output / "manifest.csv").sort_values("batch_index")
    assert records.filler_profile.tolist() == list(profiles)


@pytest.mark.parametrize("target, profiles", [
    ("present", PROFILES[:4]), ("absent", PROFILES[:3]),
])
def test_create_lineup_rejects_profile_counts_for_the_wrong_condition(
    tmp_path, backend, portrait, target, profiles,
):
    output = tmp_path / "generated"
    with pytest.raises((TypeError, ValueError)):
        create_lineup(
            suspect=portrait if target == "present" else None, target=target, size=4,
            description="dark hair", filler_profiles=profiles,
            generator_backend=backend, output_dir=output,
        )
    assert backend.requests == []
    assert not output.exists()


def test_create_lineup_rejects_profiles_with_an_existing_pool(tmp_path, backend, portrait):
    output = tmp_path / "generated"
    other = tmp_path / "other_existing.png"
    Image.new("RGB", (8, 10), (80, 50, 180)).save(other)
    with pytest.raises((TypeError, ValueError)):
        create_lineup(
            filler_pool=[portrait, other], filler_profiles=["Oval face", "Round face"],
            size=2, generator_backend=backend, output_dir=output,
        )
    assert backend.requests == []
    assert not output.exists()
