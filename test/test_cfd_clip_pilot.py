import numpy as np
import pandas as pd
import pytest

from pyWitnessAI.cfd_clip_pilot.cfd import (
    expression_from_image_stem,
    standardise_cfd_metadata,
    target_id_from_image_stem,
)
from pyWitnessAI.cfd_clip_pilot.description import build_proxy_descriptions, render_cfd_description
from pyWitnessAI.cfd_clip_pilot.evaluate import evaluate_retrieval
from pyWitnessAI.cfd_clip_pilot.index import ClipIndex
from pyWitnessAI.cfd_clip_pilot.lineup import build_filler_sets


class DummyTextEncoder:
    model_name = "dummy"

    def encode_texts(self, texts, batch_size=32, show_progress=False):
        vectors = []
        for text in texts:
            if "alpha" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "beta" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def _dummy_index():
    manifest = pd.DataFrame(
        {
            "image_id": ["img-a", "img-b", "img-c"],
            "target_id": ["target-a", "target-b", "target-c"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
        }
    )
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ClipIndex(embeddings, manifest)


def test_cfd_filename_parsing_removes_neutral_suffix():
    assert expression_from_image_stem("CFD-WM-001-001-N") == "N"
    assert target_id_from_image_stem("CFD-WM-001-001-N") == "WM-001"


def test_standardise_cfd_metadata_maps_common_columns():
    metadata = pd.DataFrame(
        {
            "Target": ["CFD-WM-001-001-N.jpg"],
            "Gender": ["Male"],
            "Race": ["White"],
            "Age": [27],
        }
    )

    standardised = standardise_cfd_metadata(metadata)

    assert standardised.loc[0, "target_id"] == "WM-001"
    assert standardised.loc[0, "gender"] == "Male"
    assert standardised.loc[0, "race"] == "White"
    assert standardised.loc[0, "age"] == 27


def test_render_cfd_description_uses_available_metadata():
    row = {"age": 27, "race": "White", "gender": "Male", "hair_colour": "Brown"}

    description = render_cfd_description(row)

    assert "around 27 years old" in description
    assert "white" in description
    assert "male" in description
    assert "hair colour: brown" in description


def test_build_proxy_descriptions_deduplicates_by_target():
    manifest = pd.DataFrame(
        {
            "target_id": ["target-a", "target-a", "target-b"],
            "image_id": ["img-a-1", "img-a-2", "img-b-1"],
            "age": [20, 21, 30],
        }
    )

    queries = build_proxy_descriptions(manifest)

    assert queries["query_id"].tolist() == ["target-a", "target-b"]
    assert len(queries) == 2


def test_evaluate_retrieval_reports_target_rank():
    queries = pd.DataFrame(
        {
            "query_id": ["q-beta"],
            "target_id": ["target-b"],
            "description": ["beta description"],
        }
    )

    results, per_query, summary = evaluate_retrieval(
        index=_dummy_index(),
        queries=queries,
        encoder=DummyTextEncoder(),
        top_k=2,
    )

    assert results.iloc[0]["target_id"] == "target-b"
    assert per_query.loc[0, "target_rank"] == 1
    assert summary["hit_rate_top_1"] == 1.0


def test_build_filler_sets_excludes_query_target():
    queries = pd.DataFrame(
        {
            "query_id": ["q-alpha"],
            "target_id": ["target-a"],
            "description": ["alpha description"],
        }
    )

    fillers = build_filler_sets(
        index=_dummy_index(),
        queries=queries,
        encoder=DummyTextEncoder(),
        top_k=2,
        filler_count=2,
    )

    assert "target-a" not in set(fillers["filler_target_id"])
    assert fillers["filler_position"].tolist() == [1, 2]


def test_clip_index_search_validates_top_k():
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        _dummy_index().search_vectors(np.asarray([[1.0, 0.0, 0.0]]), top_k=0)


def test_clip_index_search_preserves_metadata_and_stable_ties():
    index = _dummy_index()
    index.manifest["group"] = ["first", "second", "third"]

    results = index.search_vectors(
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        top_k=2,
    )

    assert results["image_id"].tolist() == ["img-a", "img-b"]
    assert results["group"].tolist() == ["first", "second"]
    assert results.columns[:7].tolist() == [
        "query_index",
        "rank",
        "image_index",
        "image_id",
        "target_id",
        "image_path",
        "clip_score",
    ]


def test_clip_index_rejects_missing_ids():
    manifest = _dummy_index().manifest.copy()
    manifest.loc[0, "image_id"] = pd.NA

    with pytest.raises(ValueError, match="image_id values must not be empty"):
        ClipIndex(np.eye(3, dtype=np.float32), manifest)


def test_clip_index_float64_build_round_trips_cache(tmp_path):
    class Float64ImageEncoder:
        model_name = "float64-test"

        def encode_images(self, image_paths, batch_size=32, show_progress=False):
            return np.asarray([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)

    manifest = _dummy_index().manifest.head(2)
    index = ClipIndex.build(manifest, encoder=Float64ImageEncoder())
    index.save(tmp_path / "index")
    loaded = ClipIndex.load(tmp_path / "index")

    assert loaded.image_embeddings.dtype == np.float32
    np.testing.assert_allclose(
        loaded.image_embeddings,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
