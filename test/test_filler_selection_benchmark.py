from pathlib import Path
import hashlib
import json
import re

import pandas as pd
import pytest
from PIL import Image

from pyWitnessAI import (
    BenchmarkStage,
    DEFAULT_FACE_ATTRIBUTE_SCHEMA,
    FaceAttributeDefinition,
    FaceDescriptionSchema,
    FillerSelectionBenchmark,
    ImageGenerationBackend,
    ImageGenerationRequest,
    SelectorBackend,
    SelectorQuery,
)


class _TinyBenchmarkBackend(ImageGenerationBackend):
    provider = "tiny-benchmark"
    default_model = "tiny-benchmark-model"

    def __init__(self) -> None:
        self.requests: list[ImageGenerationRequest] = []

    def generate(self, request: ImageGenerationRequest) -> None:
        self.requests.append(request)
        request.output_path.write_bytes(b"benchmark-image")


class _PerfectSelectorBackend(SelectorBackend):
    name = "perfect-test"
    model_name = "perfect-v1"

    def score(self, manifest: pd.DataFrame, query: SelectorQuery) -> pd.DataFrame:
        return manifest[["image_id"]].assign(
            selector_score=manifest["is_positive"].map({True: 1.0, False: -1.0})
        )


MALE_CONTRAST = {("gender", "male"): "female"}


def _write_dummy_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        image = Image.new("RGB", (8, 8), color=(index * 20 % 255, 80, 120))
        image.save(folder / f"face_{index:02d}.png")


def test_default_schema_fails_closed_without_an_explicit_contrast(tmp_path: Path):
    with pytest.raises(ValueError, match="parsing-only"):
        FillerSelectionBenchmark(
            "a male person",
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
        )


def test_missing_contrast_skip_is_explicit_and_provenance_is_hashed(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person with blue eyes",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
        missing_contrast="skip",
    )

    assert [stage.feature for stage in benchmark.stages] == ["gender"]
    assert benchmark.stages[0].contrast_source == "override:feature+positive"
    assert benchmark.contrast_audit["skipped_features"] == [
        {
            "feature": "eyes",
            "positive_label": "blue eyes",
            "reason": "no_explicit_contrast",
        }
    ]
    assert len(benchmark.contrast_config_sha256) == 64
    assert benchmark.case_slug.endswith(benchmark.contrast_config_sha256[:8])

    alternative = FillerSelectionBenchmark(
        "a male person with blue eyes",
        dataset_root=tmp_path / "dataset_alt",
        results_root=tmp_path / "results_alt",
        generate_missing=False,
        contrast_overrides={("gender", "male"): "non-binary"},
        missing_contrast="skip",
    )
    assert alternative.contrast_config_sha256 != benchmark.contrast_config_sha256
    assert alternative.case_slug != benchmark.case_slug


def test_contrast_hash_includes_complete_schema_definitions(tmp_path: Path):
    contrasts = {"first": "negative-first", "second": "negative-second"}
    first_schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
        FaceAttributeDefinition(
            "custom_probe",
            1,
            patterns=((r"\bprobe\b", "first"),),
            contrasts=contrasts,
        ),
        name="same-name",
        version="same-version",
    )
    second_schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
        FaceAttributeDefinition(
            "custom_probe",
            1,
            patterns=((r"\bprobe\b", "second"),),
            contrasts=contrasts,
        ),
        name="same-name",
        version="same-version",
    )

    first = FillerSelectionBenchmark(
        "probe",
        attribute_schema=first_schema,
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
    )
    second = FillerSelectionBenchmark(
        "probe",
        attribute_schema=second_schema,
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
    )

    assert first.stages[0].positive_label == "first"
    assert second.stages[0].positive_label == "second"
    assert first.contrast_config_sha256 != second.contrast_config_sha256
    assert first.case_slug != second.case_slug


def test_run_hash_separates_selector_models_without_changing_contrast_hash(tmp_path: Path):
    common = {
        "verbal_description": "a male person",
        "contrast_overrides": MALE_CONTRAST,
        "dataset_root": tmp_path / "dataset",
        "results_root": tmp_path / "results",
        "generate_missing": False,
    }
    small = FillerSelectionBenchmark(selector_model="clip-ViT-B-32", **common)
    large = FillerSelectionBenchmark(selector_model="clip-ViT-L-14", **common)

    assert small.contrast_config_sha256 == large.contrast_config_sha256
    assert small.case_slug == large.case_slug
    assert small.run_config_sha256 != large.run_config_sha256
    assert small.result_dir != large.result_dir


def test_unused_and_feature_only_overrides_fail_or_warn_explicitly(tmp_path: Path):
    overrides = {
        ("gender", "male"): "female",
        ("eyes", "blue eyes"): "brown eyes",
    }
    with pytest.raises(ValueError, match="Unused contrast_overrides"):
        FillerSelectionBenchmark(
            "a male person",
            contrast_overrides=overrides,
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
        )

    with pytest.warns(UserWarning, match="Unused contrast_overrides"):
        reusable = FillerSelectionBenchmark(
            "a male person",
            contrast_overrides=overrides,
            unused_contrast="warn",
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
        )
    assert reusable.contrast_audit["unused_overrides"][0]["feature"] == "eyes"

    with pytest.warns(DeprecationWarning, match="Feature-only"):
        legacy_key = FillerSelectionBenchmark(
            "a male person",
            contrast_overrides={"gender": "female"},
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
        )
    assert legacy_key.stages[0].contrast_source == "override:feature"


def test_casefold_equivalent_contrast_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="must differ"):
        FillerSelectionBenchmark(
            "a male person",
            contrast_overrides={("gender", "male"): " MALE "},
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
        )


def test_benchmark_stage_retains_old_positional_constructor(tmp_path: Path):
    schema = FaceDescriptionSchema(original_description="test")
    stage = BenchmarkStage(
        "single",
        1,
        "feature",
        "positive",
        "negative",
        "query",
        schema,
        schema,
        tmp_path,
    )
    assert stage.contrast_source == ""


def test_numeric_age_never_receives_an_automatic_old_young_contrast(tmp_path: Path):
    with pytest.raises(ValueError, match="age='around 30 years old'"):
        FillerSelectionBenchmark(
            "a 30 years old male person",
            dataset_root=tmp_path / "dataset",
            results_root=tmp_path / "results",
            generate_missing=False,
            contrast_overrides=MALE_CONTRAST,
        )


def test_benchmark_builds_single_feature_stage_from_schema(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a white dude with beard and blue eyes, thin lips",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        contrast_overrides={
            ("gender", "male"): "female",
            ("facial_hair", "beard"): "no facial hair",
            ("eyes", "blue eyes"): "brown eyes",
            ("race", "White"): "Asian",
            ("mouth", "thin lips"): "full lips",
        },
    )

    assert [stage.feature for stage in benchmark.stages] == [
        "gender",
        "facial_hair",
        "eyes",
        "race",
        "mouth",
    ]
    assert benchmark.stages[0].positive_label == "male"
    assert benchmark.stages[0].negative_label == "female"


def test_benchmark_includes_age_after_gender(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a young adult white male with long hair",
        n=2,
        mode="ladder",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        contrast_overrides={
            ("gender", "male"): "female",
            ("age", "young adult"): "older adult",
            ("hair", "long hair"): "short hair",
            ("race", "White"): "Asian",
        },
    )

    assert [stage.feature for stage in benchmark.stages[:3]] == ["gender", "age", "hair"]
    assert benchmark.stages[1].positive_label == "young adult"
    assert benchmark.stages[1].negative_label == "older adult"
    assert "young adult male person" in benchmark.stages[1].query


def test_benchmark_uses_extended_schema_for_custom_stage(tmp_path: Path):
    schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
        FaceAttributeDefinition(
            "skin_marking",
            65,
            patterns=((r"\bfreckled skin\b", "freckled skin"),),
            contrasts={"freckled skin": "unfreckled skin"},
        )
    )
    benchmark = FillerSelectionBenchmark(
        "a male with freckled skin and a hawk nose",
        n=1,
        mode="ladder",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        attribute_schema=schema,
        contrast_overrides={
            ("gender", "male"): "female",
            ("nose", "hawk nose"): "straight nose",
        },
    )

    stages = {stage.feature: stage for stage in benchmark.stages}
    assert stages["skin_marking"].positive_label == "freckled skin"
    assert stages["skin_marking"].negative_label == "unfreckled skin"
    assert stages["nose"].positive_label == "hawk nose"
    assert stages["nose"].negative_label == "straight nose"

    image_path = tmp_path / "freckled.png"
    Image.new("RGB", (8, 8), color=(80, 100, 120)).save(image_path)
    benchmark.dataset.import_images(
        [image_path],
        schema=stages["skin_marking"].positive_schema,
        verbal_description=stages["skin_marking"].query,
    )
    manifest = benchmark.dataset.load_manifest()
    assert manifest.loc[0, "attribute__skin_marking"] == "freckled skin"


def test_benchmark_stage_plan_reports_manifest_coverage(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        dataset_match="contains",
        contrast_overrides=MALE_CONTRAST,
    )

    stage = benchmark.stages[0]
    male = tmp_path / "male.png"
    female = tmp_path / "female.png"
    Image.new("RGB", (8, 8), color=(40, 80, 120)).save(male)
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(female)
    benchmark.dataset.import_images([male], schema=stage.positive_schema, verbal_description=stage.query)
    benchmark.dataset.import_images([female], schema=stage.negative_schema, verbal_description=stage.query)

    plan = benchmark.stage_plan()

    assert plan.loc[0, "n_positive_manifest"] == 1
    assert plan.loc[0, "n_negative_manifest"] == 1
    assert plan.loc[0, "positive_filter"] == "gender=male"


def test_benchmark_runs_with_custom_selector_and_updates_manifest(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        selector=lambda manifest, query, top_k: (
            manifest.sort_values("is_positive", ascending=False)
            .head(top_k)
            .assign(rank=range(1, min(top_k, len(manifest)) + 1), selector_score=1.0)
        ),
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
    )

    stage = benchmark.stages[0]
    _write_dummy_images(stage.stage_dir / "male", 2)
    _write_dummy_images(stage.stage_dir / "female", 2)

    results = benchmark.run(display=False)
    summary = benchmark.statistics(display=False)

    assert len(results) == 4
    assert summary.loc[0, "positives_in_top_k"] == 2
    assert bool(summary.loc[0, "perfect_top_k"]) is True
    assert (tmp_path / "dataset" / "manifest.csv").exists()

    manifest = pd.read_csv(tmp_path / "dataset" / "manifest.csv")
    assert set(manifest["source_label_role"]) == {"positive", "negative"}
    assert set(manifest["gender"]) == {"male", "female"}


def test_benchmark_runs_from_manifest_without_stage_folders(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        selector=lambda manifest, query, top_k: (
            manifest.sort_values("is_positive", ascending=False)
            .head(top_k)
            .assign(rank=range(1, min(top_k, len(manifest)) + 1), selector_score=1.0)
        ),
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
    )

    stage = benchmark.stages[0]
    positive_paths = []
    negative_paths = []
    for index in range(1, 3):
        positive = tmp_path / f"manifest_male_{index}.png"
        negative = tmp_path / f"manifest_female_{index}.png"
        Image.new("RGB", (8, 8), color=(40, index * 40, 120)).save(positive)
        Image.new("RGB", (8, 8), color=(120, index * 40, 40)).save(negative)
        positive_paths.append(positive)
        negative_paths.append(negative)

    benchmark.dataset.import_images(
        positive_paths,
        schema=stage.positive_schema,
        verbal_description=stage.query,
        source="test_manifest",
    )
    benchmark.dataset.import_images(
        negative_paths,
        schema=stage.negative_schema,
        verbal_description=stage.query,
        source="test_manifest",
    )

    results = benchmark.run(display=False)

    assert len(results) == 4
    assert set(results["label_role"]) == {"positive", "negative"}
    assert not (stage.stage_dir / "male").exists()


def test_benchmark_generates_missing_images_with_custom_backend(tmp_path: Path):
    backend = _TinyBenchmarkBackend()
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=1,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generator_backend=backend,
        selector=lambda manifest, query, top_k: manifest.head(top_k).assign(rank=1, selector_score=1.0),
        generate_missing=True,
        contrast_overrides=MALE_CONTRAST,
    )

    results = benchmark.run(display=False)

    assert len(backend.requests) == 2
    assert len(results) == 2
    assert {request.model for request in backend.requests} == {"tiny-benchmark-model"}

    manifest = pd.read_csv(tmp_path / "dataset" / "manifest.csv")
    assert set(manifest["source"]) == {"generated"}
    assert all(re.fullmatch(r"face_[0-9a-f]{32}_0001\.png", name) for name in manifest["image_name"])
    assert manifest["attribute_label_source"].eq("requested").all()
    assert manifest["attributes_verified"].eq(False).all()
    assert {Path(path).parts[0] for path in manifest["relative_image_path"]} == {"images"}
    assert not (tmp_path / "dataset" / "images" / "manifest.csv").exists()
    assert (tmp_path / "dataset" / "images").exists()
    assert (tmp_path / "dataset" / "batches").exists()


def test_benchmark_persists_contrast_audit_and_custom_schema_metadata(tmp_path: Path):
    backend = _TinyBenchmarkBackend()
    custom_schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA.extend(
        FaceAttributeDefinition(
            "skin_marking",
            65,
            patterns=((r"\bfreckled skin\b", "freckled skin"),),
            contrasts={"freckled skin": "even cheek complexion"},
        ),
        name="custom-skin-schema",
        version="pilot-1",
    )
    benchmark = FillerSelectionBenchmark(
        "a male person with freckled skin and blue eyes",
        n=1,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generator_backend=backend,
        selector=lambda manifest, query, top_k: manifest.assign(selector_score=1.0),
        generate_missing=True,
        attribute_schema=custom_schema,
        contrast_overrides=MALE_CONTRAST,
        missing_contrast="skip",
    )

    benchmark.run(display=False)

    manifest = benchmark.dataset.load_manifest()
    contrast = manifest.loc[manifest["attribute__skin_marking"] == "even cheek complexion"]
    assert len(contrast) == 1
    assert "even cheek complexion" in contrast.iloc[0]["prompt"]
    assert contrast.iloc[0]["attribute_label_source"] == "requested"

    metadata = json.loads((benchmark.result_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["contrast_audit"]["skipped_features"] == [
        {
            "feature": "eyes",
            "positive_label": "blue eyes",
            "reason": "no_explicit_contrast",
        }
    ]
    contrast_payload = json.dumps(
        metadata["contrast_configuration"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    run_payload = json.dumps(
        metadata["run_configuration"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(contrast_payload).hexdigest() == metadata["contrast_config_sha256"]
    assert hashlib.sha256(run_payload).hexdigest() == metadata["run_config_sha256"]

    batch_files = sorted((tmp_path / "dataset" / "batches").glob("*.json"))
    assert len(batch_files) == 4
    for batch_file in batch_files:
        batch = json.loads(batch_file.read_text(encoding="utf-8"))
        assert batch["attribute_schema_name"] == "custom-skin-schema"
        assert batch["attribute_schema_version"] == "pilot-1"
        assert any(item["name"] == "skin_marking" for item in batch["attribute_schema"])


def test_benchmark_reuses_successful_faces_after_a_partial_generation_failure(tmp_path: Path):
    class FailingBackend(_TinyBenchmarkBackend):
        def generate(self, request):
            super().generate(request)
            if len(self.requests) == 2:
                raise RuntimeError("provider interrupted the second image")

    backend = FailingBackend()
    benchmark = FillerSelectionBenchmark(
        "a male person", n=2, mode="single",
        dataset_root=tmp_path / "dataset", results_root=tmp_path / "results",
        generator_backend=backend, contrast_overrides=MALE_CONTRAST,
        selector=lambda manifest, query, top_k: manifest.assign(selector_score=1.0),
    )

    with pytest.raises(RuntimeError, match="provider interrupted"):
        benchmark.run(display=False)

    partial = benchmark.dataset.load_manifest()
    assert len(partial) == 1
    assert partial["gender"].tolist() == ["male"]
    assert not backend.requests[1].output_path.exists()
    batch_files = list(benchmark.dataset.batches_dir.glob("*.json"))
    assert len(batch_files) == 1
    assert json.loads(batch_files[0].read_text(encoding="utf-8"))["n"] == 1

    results = benchmark.run(display=False)

    complete = benchmark.dataset.load_manifest()
    assert len(results) == len(complete) == 4
    assert len(backend.requests) == 5  # Four successes, plus the failed request.
    assert set(partial["image_id"]) <= set(complete["image_id"])
    assert complete["image_id"].is_unique
    assert all(Path(path).is_file() for path in complete["image_path"])
    assert not (benchmark.dataset.images_dir / "manifest.csv").exists()


def test_benchmark_accepts_shared_selector_backend(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=1,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        selector=_PerfectSelectorBackend(),
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
    )
    stage = benchmark.stages[0]
    _write_dummy_images(stage.stage_dir / "male", 1)
    _write_dummy_images(stage.stage_dir / "female", 1)

    results = benchmark.run(display=False)

    assert results.iloc[0]["is_positive"]
    assert set(results["selector_backend"]) == {"perfect-test"}
    assert set(results["selector_model"]) == {"perfect-v1"}
    assert "selector_model: perfect-v1" in benchmark.summary_text()


def test_benchmark_adapts_rank_only_legacy_selector(tmp_path: Path):
    def rank_only_selector(manifest, query, top_k):
        output = manifest.sort_values("is_positive", ascending=False).head(top_k)[["image_id"]]
        return output.assign(rank=range(1, len(output) + 1))

    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=1,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        selector=rank_only_selector,
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
    )
    stage = benchmark.stages[0]
    _write_dummy_images(stage.stage_dir / "male", 1)
    _write_dummy_images(stage.stage_dir / "female", 1)

    results = benchmark.run(display=False)

    assert results["is_positive"].tolist() == [True, False]
    assert results["selector_score"].tolist() == [-1.0, -2.0]


@pytest.mark.parametrize(
    "sources, expected",
    [
        (None, "unspecified"),
        ([None, pd.NA, " ", ""], "unspecified"),
        (["requested", " requested ", None, ""], "requested"),
        (["requested", " manual ", pd.NA, "requested"], "manual|requested"),
    ],
)
def test_stage_summary_uses_label_sources_only_from_evaluated_images(tmp_path: Path, sources, expected: str):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        selector=_PerfectSelectorBackend(),
        generate_missing=False,
        contrast_overrides=MALE_CONTRAST,
    )
    records = []
    for index, gender in enumerate(("male", "female", "male", "female", "non-binary")):
        image = tmp_path / f"face_{index}.png"
        Image.new("RGB", (8, 8), (40, 80, 120)).save(image)
        row = {
            "image_id": f"face_{index}",
            "image_path": str(image),
            "gender": gender,
            "attributes_verified": False,
        }
        if sources is not None:
            row["attribute_label_source"] = sources[index] if index < 4 else "unselected"
        records.append(row)
    benchmark.dataset.add_records(records)

    results = benchmark.run(display=False)
    summary = benchmark.statistics(display=False).iloc[0]

    assert len(results) == 4
    assert summary["evaluation_label_source"] == expected
    assert summary["precision_at_k"] == 1.0
    assert bool(summary["perfect_top_k"]) is True


def test_stage_metrics_and_display_preserve_imperfect_retrieval(tmp_path: Path, monkeypatch):
    import matplotlib
    from matplotlib.figure import Figure

    matplotlib.use("Agg")
    ipython_display = pytest.importorskip("IPython.display")
    displayed = []
    monkeypatch.setattr(ipython_display, "display", displayed.append)

    def selector(manifest, query, top_k):
        return manifest[["image_id", "expected_score"]].rename(
            columns={"expected_score": "selector_score"},
        )

    benchmark = FillerSelectionBenchmark(
        "a male person", n=2, selector=selector,
        dataset_root=tmp_path / "dataset", results_root=tmp_path / "results",
        generate_missing=False, contrast_overrides=MALE_CONTRAST,
    )
    records = []
    for index, gender in enumerate(("male", "female", "male", "female")):
        path = tmp_path / f"face_{index}.png"
        Image.new("RGB", (8, 8), (40, 80, 120)).save(path)
        records.append({
            "image_id": f"face_{index}", "image_path": str(path),
            "gender": gender, "expected_score": 0.9 - index / 10,
        })
    benchmark.dataset.add_records(records)

    ranked = benchmark.run(display=True)
    summary = benchmark.statistics(display=False).iloc[0]

    assert ranked["is_positive"].tolist() == [True, False, True, False]
    assert summary["positives_in_top_k"] == 1
    assert summary["precision_at_k"] == 0.5
    assert summary["recall_at_k"] == 0.5
    assert bool(summary["perfect_top_k"]) is False
    assert summary["mean_positive_rank"] == 2
    assert summary["worst_positive_rank"] == 3
    assert summary["first_negative_rank"] == 2
    figures = [value for value in displayed if isinstance(value, Figure)]
    assert len(figures) == 1
    assert figures[0].axes[0].get_title() == "Rank 1 | 0.900\nmale"
    assert figures[0].axes[1].get_title() == "Rank 2 | 0.800\nfemale"
