import pandas as pd

from pyWitnessAI.cfd_clip_pilot import (
    ClipRetrievalProbe,
    DescriptionLadder,
    filter_expression_images,
    find_manifest_row,
)


def test_description_ladder_builds_cumulative_prompts():
    ladder = DescriptionLadder.from_steps(
        [
            ("gender", "female", "subject"),
            ("hair", "long brown hair", "detail"),
            ("race", "White", "subject"),
        ],
        base_details=("a happy open-mouth smile",),
        base_feature="expression_only",
    )

    frame = ladder.to_frame()

    assert frame["feature"].tolist() == [
        "expression_only",
        "gender",
        "hair",
        "race",
    ]
    assert frame.loc[0, "description"] == (
        "A frontal face photograph of a person with a happy open-mouth smile."
    )
    assert frame.loc[3, "description"] == (
        "A frontal face photograph of a White female person with "
        "a happy open-mouth smile, long brown hair."
    )


def test_filter_expression_images_removes_neutral_rows():
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "expression": ["N", "HO", "", "HC"],
        }
    )

    filtered = filter_expression_images(manifest)

    assert filtered["image_id"].tolist() == ["b", "d"]


def test_find_manifest_row_accepts_cfd_image_id_variants():
    manifest = pd.DataFrame(
        {
            "image_id": ["CFD-WF-025-006-HO"],
            "target_id": ["WF-025"],
            "image_path": ["face.jpg"],
        }
    )

    row = find_manifest_row(manifest, image_id="WF-025-006-HO")

    assert row["target_id"] == "WF-025"


def test_clip_retrieval_probe_can_run_ladder_without_target():
    class FakeIndex:
        manifest = pd.DataFrame(
            {
                "image_id": ["img-1"],
                "target_id": ["person-1"],
                "image_path": ["face.jpg"],
                "expression": ["HO"],
            }
        )

        def search_texts(self, texts, **kwargs):
            assert texts == ["A frontal face photograph of a female person."]
            return pd.DataFrame(
                {
                    "query_index": [0],
                    "rank": [1],
                    "image_index": [0],
                    "image_id": ["img-1"],
                    "target_id": ["person-1"],
                    "image_path": ["face.jpg"],
                    "clip_score": [0.9],
                }
            )

    ladder = DescriptionLadder.from_steps([("gender", "female", "subject")])
    probe = ClipRetrievalProbe(index=FakeIndex(), encoder=object())

    results = probe.run_ladder(ladder, candidate_columns=("expression",))

    assert results.loc[0, "feature"] == "gender"
    assert results.loc[0, "candidate_expression"] == "HO"
    assert "is_target_image" not in results.columns
