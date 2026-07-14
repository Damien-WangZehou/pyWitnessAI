from pathlib import Path
import re

import pandas as pd
from PIL import Image

from pyWitnessAI import FillerSelectionBenchmark, ImageGenerationBackend, ImageGenerationRequest


class _TinyBenchmarkBackend(ImageGenerationBackend):
    provider = "tiny-benchmark"
    default_model = "tiny-benchmark-model"

    def __init__(self) -> None:
        self.requests: list[ImageGenerationRequest] = []

    def generate(self, request: ImageGenerationRequest) -> None:
        self.requests.append(request)
        request.output_path.write_bytes(b"benchmark-image")


def _write_dummy_images(folder: Path, count: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        image = Image.new("RGB", (8, 8), color=(index * 20 % 255, 80, 120))
        image.save(folder / f"face_{index:02d}.png")


def test_benchmark_builds_single_feature_stage_from_schema(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a white dude with beard and blue eyes, thin lips",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
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
    )

    assert [stage.feature for stage in benchmark.stages[:3]] == ["gender", "age", "hair"]
    assert benchmark.stages[1].positive_label == "young adult"
    assert benchmark.stages[1].negative_label == "older adult"
    assert "young adult male person" in benchmark.stages[1].query


def test_benchmark_stage_plan_reports_manifest_coverage(tmp_path: Path):
    benchmark = FillerSelectionBenchmark(
        "a male person",
        n=2,
        mode="single",
        dataset_root=tmp_path / "dataset",
        results_root=tmp_path / "results",
        generate_missing=False,
        dataset_match="contains",
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
    )

    results = benchmark.run(display=False)

    assert len(backend.requests) == 2
    assert len(results) == 2
    assert {request.model for request in backend.requests} == {"tiny-benchmark-model"}

    manifest = pd.read_csv(tmp_path / "dataset" / "manifest.csv")
    assert set(manifest["source"]) == {"generated"}
    assert all(re.match(r"gf_\d{8}_\d{6}_[0-9a-f]{8}_000[1]\.png", name) for name in manifest["image_name"])
    assert (tmp_path / "dataset" / "images").exists()
    assert (tmp_path / "dataset" / "batches").exists()
