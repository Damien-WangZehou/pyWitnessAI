import pytest

from pyWitnessAI import (
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    LEGACY_FACE_ATTRIBUTE_SCHEMA_V1,
)


def test_default_schema_is_parsing_only_and_legacy_contrasts_require_opt_in():
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.version == "2.0.0-safe-parsing-only"
    assert all(
        not definition.contrasts
        for definition in DEFAULT_FACE_ATTRIBUTE_SCHEMA.definitions
    )
    assert LEGACY_FACE_ATTRIBUTE_SCHEMA_V1.contrast_for("race", "Black") == "White"


def test_colours_on_hair_and_clothing_never_invent_race():
    ordinary = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a man with black hair wearing a white shirt"
    )
    explicit = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a Black man with black hair wearing a white shirt"
    )

    assert ordinary["gender"] == "male"
    assert ordinary["hair_color"] == "black hair"
    assert ordinary["clothing"] == "white shirt"
    assert "race" not in ordinary
    assert explicit["race"] == "Black"
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("an East Asian man")["race"] == "East Asian"
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a South Asian woman")["race"] == "South Asian"


def test_negated_and_uncertain_values_are_not_returned_as_affirmed():
    negated = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "not a man with no blue eyes and no glasses"
    )
    qualified = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "brown eyes, possibly blue eyes"
    )
    unable = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "I cannot say whether he had green eyes"
    )

    assert not {"gender", "eyes", "accessories"}.intersection(negated)
    assert qualified["eyes"] == "brown eyes"
    assert "eyes" not in unable


def test_age_requires_explicit_age_context():
    assert "age" not in DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a man with a 30 cm scar")
    assert "age" not in DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a man with an old scar")
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a man aged 30")["age"] == (
        "around 30 years old"
    )
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a 30-year-old man")["age"] == (
        "around 30 years old"
    )

    with pytest.raises(ValueError, match="adult ages"):
        DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a 16 years old male person")


def test_vague_hair_presence_and_cross_feature_texture_leakage_are_removed():
    colour_only = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a man with brown hair")
    unrelated = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a man with a curly beard and black hair"
    )

    assert colour_only["hair_color"] == "brown hair"
    assert "hair" not in colour_only
    assert "hair_texture" not in unrelated


def test_scalar_fields_abstain_when_multiple_values_or_dimensions_match():
    conflict = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a person with blue eyes and brown eyes, thick arched eyebrows, "
        "and a broad long nose, wearing glasses and a hat"
    )

    assert "eyes" not in conflict
    assert "eyebrows" not in conflict
    assert "nose" not in conflict
    assert "accessories" not in conflict


def test_overbroad_aliases_are_kept_as_distinct_labels():
    parsed = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a person with an aquiline nose, a long face, a stocky build, and prominent ears"
    )

    assert parsed["nose"] == "aquiline nose"
    assert parsed["face_shape"] == "long face"
    assert parsed["build"] == "stocky build"
    assert parsed["ears"] == "prominent ears"


@pytest.mark.parametrize(
    "description", ("a non-white man", "a non white man", "a non–white man")
)
def test_non_demographic_labels_never_become_affirmative(description):
    parsed = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(description)

    assert parsed["gender"] == "male"
    assert "race" not in parsed


@pytest.mark.parametrize(
    "description",
    (
        "blue or green eyes",
        "blue/green eyes",
        "a Black or White man",
        "a South or East Asian man",
    ),
)
def test_disjunctive_attributes_abstain(description):
    parsed = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(description)

    if "eyes" in description:
        assert "eyes" not in parsed
    else:
        assert "race" not in parsed


@pytest.mark.parametrize(
    "description",
    (
        "blue eyes, maybe",
        "blue eyes, maybe. He had black hair.",
        "he could have blue eyes",
        "uncertain whether he had blue eyes",
    ),
)
def test_prefix_and_postposed_uncertainty_abstain(description):
    assert "eyes" not in DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(description)


def test_all_explicit_numeric_ages_are_validated_before_coordination_abstains():
    with pytest.raises(ValueError, match="adult ages"):
        DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("16 or 30 years old male")
    with pytest.raises(ValueError, match="adult ages"):
        DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a 16-year-old or 30-year-old male")
    with pytest.raises(ValueError, match="adult ages"):
        DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a man aged 30 to 16")
    with pytest.raises(ValueError, match="adult ages"):
        DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("between 16 and 30 years old")

    adults = DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "a 30-year-old or 40-year-old male"
    )
    assert "age" not in adults


def test_emotion_words_require_visible_expression_context():
    assert "expression" not in DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse(
        "an angry scar on his cheek"
    )
    assert "expression" not in DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("he sounded sad")
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("he had an angry expression")[
        "expression"
    ] == "angry expression"
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("he looked sad")["expression"] == (
        "sad expression"
    )


def test_beard_size_and_density_are_not_relabelled_as_full_coverage():
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a big beard")["facial_hair"] == (
        "big beard"
    )
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a thick beard")["facial_hair"] == (
        "thick beard"
    )
    assert DEFAULT_FACE_ATTRIBUTE_SCHEMA.parse("a full beard")["facial_hair"] == (
        "full beard"
    )
