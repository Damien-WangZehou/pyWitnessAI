from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

import pandas as pd

from .FaceAttributeSchema import DEFAULT_FACE_ATTRIBUTE_SCHEMA, FaceAttributeSchema
from .FaceSearch import (
    ClipSelectorBackend,
    SelectorBackend,
    SelectorQuery,
    normalise_selector_scores,
)
from .FillerGenerator import FaceDescriptionSchema, FillerGenerator, ImageGenerationBackend
from .GeneratedFaceDataset import DatasetMatchMode, GeneratedFaceDataset
from .ImageCatalog import discover_images

BenchmarkMode = Literal["single", "ladder"]
SelectorCallable = Callable[[pd.DataFrame, str, int], pd.DataFrame]

DEFAULT_DATASET_ROOT = Path("./data/generated_filler_benchmark")
DEFAULT_RESULTS_ROOT = Path("./build/filler_selection_benchmark")

__all__ = [
    "BenchmarkStage",
    "FillerSelectionBenchmark",
    "FillerSelectorBenchmark",
]

SCHEMA_FEATURE_ORDER = DEFAULT_FACE_ATTRIBUTE_SCHEMA.feature_order
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_CONTRASTS = {
    definition.name: dict(definition.contrasts)
    for definition in DEFAULT_FACE_ATTRIBUTE_SCHEMA.definitions
}


@dataclass(frozen=True)
class BenchmarkStage:
    mode: str
    step: int
    feature: str
    positive_label: str
    negative_label: str
    query: str
    positive_schema: FaceDescriptionSchema
    negative_schema: FaceDescriptionSchema
    stage_dir: Path

    @property
    def stage_name(self) -> str:
        return f"{self.step:02d}_{self.feature}"


class FillerSelectionBenchmark:
    """Generate controlled synthetic filler sets and benchmark a filler selector."""

    def __init__(
        self,
        verbal_description: str,
        n: int = 9,
        mode: BenchmarkMode = "ladder",
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
        generator_provider: str = "openai",
        generator_model: str | None = None,
        generator_backend: ImageGenerationBackend | None = None,
        generator_backend_kwargs: Mapping[str, object] | None = None,
        selector: str | SelectorCallable | SelectorBackend = "clip",
        selector_model: str = "clip-ViT-B-32",
        device: str | None = None,
        image_size: str = "1024x1024",
        image_quality: str = "medium",
        image_format: str = "png",
        generate_missing: bool = True,
        overwrite_generated: bool = False,
        dataset_match: DatasetMatchMode = "exact",
        contrast_overrides: dict[str, str] | None = None,
        attribute_schema: FaceAttributeSchema | None = None,
    ) -> None:
        if mode not in {"single", "ladder"}:
            raise ValueError("mode must be 'single' or 'ladder'.")
        if n < 1:
            raise ValueError("n must be >= 1.")
        if dataset_match not in {"exact", "contains"}:
            raise ValueError("dataset_match must be 'exact' or 'contains'.")

        self.verbal_description = verbal_description
        self.n = int(n)
        self.mode = mode
        self.dataset_root = Path(dataset_root)
        self.results_root = Path(results_root)
        self.generator_provider = (getattr(generator_backend, "provider", None) or generator_provider).strip().lower()
        self.generator_model = generator_model
        self.generator_backend = generator_backend
        self.generator_backend_kwargs = dict(generator_backend_kwargs or {})
        self.selector = selector
        self.selector_model = selector_model
        self.device = device
        self.image_size = image_size
        self.image_quality = image_quality
        self.image_format = image_format
        self.generate_missing = generate_missing
        self.overwrite_generated = overwrite_generated
        self.dataset_match = dataset_match
        self.contrast_overrides = contrast_overrides or {}
        self.attribute_schema = attribute_schema or DEFAULT_FACE_ATTRIBUTE_SCHEMA
        self.dataset = GeneratedFaceDataset(self.dataset_root)

        self.schema = FillerGenerator.parse_description(
            verbal_description,
            attribute_schema=self.attribute_schema,
        )
        self.case_slug = _case_slug(verbal_description)
        self.stages = self._build_stages()
        self.results_: pd.DataFrame = pd.DataFrame()
        self.summary_: pd.DataFrame = pd.DataFrame()

    @property
    def manifest_path(self) -> Path:
        return self.dataset.manifest_path

    @property
    def mode_dir(self) -> Path:
        if self.mode == "single":
            return self.dataset_root / "single_feature"
        return self.dataset_root / "ladder" / self.case_slug

    def run(
        self,
        *,
        generate_missing: bool | None = None,
        display: bool = True,
        rebuild_index: bool = False,
    ) -> pd.DataFrame:
        """Prepare images, run every stage, optionally display each lineup."""
        if generate_missing is None:
            generate_missing = self.generate_missing

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        all_results = []
        summaries = []

        for stage in self.stages:
            print(f"\nRunning {self.mode}/{stage.stage_name}: {stage.feature}")
            stage_manifest = self._ensure_stage_dataset(stage, generate_missing=generate_missing)
            if len(stage_manifest) < self.n * 2:
                raise FileNotFoundError(
                    f"{stage.stage_name} needs {self.n} positive and {self.n} negative images. "
                    f"Found {len(stage_manifest)} manifest-matched rows. Add matching rows to "
                    f"{self.dataset.manifest_path}, import images through GeneratedFaceDataset.import_images(), "
                    "or set generate_missing=True."
                )

            ranked = self._run_selector(stage, stage_manifest, rebuild_index=rebuild_index)
            summary = self._summarise_stage(stage, ranked)
            all_results.append(ranked)
            summaries.append(summary)
            self._display_stage(stage, ranked, summary, enabled=display)

        self.results_ = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        self.summary_ = pd.DataFrame(summaries)
        self._write_results()
        return self.results_

    def statistics(self, display: bool = True) -> pd.DataFrame:
        """Return the per-stage benchmark summary table."""
        if self.summary_.empty:
            print("No benchmark results yet. Run benchmark.run() first.")
            return self.summary_
        if display:
            try:
                from IPython.display import display as ipy_display

                ipy_display(self.summary_)
            except ImportError:
                print(self.summary_.to_string(index=False))
        return self.summary_

    def print_statistics(self) -> pd.DataFrame:
        """Compatibility alias for users who prefer an explicit print method."""
        return self.statistics(display=True)

    def stage_plan(self) -> pd.DataFrame:
        """Return the stage-level plan and current manifest coverage."""
        rows = []
        for stage in self.stages:
            positive = self.dataset.select(stage.positive_schema, match=self.dataset_match)
            negative = self.dataset.select(stage.negative_schema, match=self.dataset_match)
            rows.append(
                {
                    "mode": self.mode,
                    "stage": stage.stage_name,
                    "feature": stage.feature,
                    "positive_label": stage.positive_label,
                    "negative_label": stage.negative_label,
                    "positive_filter": _schema_filter_text(stage.positive_schema),
                    "negative_filter": _schema_filter_text(stage.negative_schema),
                    "n_positive_manifest": len(positive),
                    "n_negative_manifest": len(negative),
                    "dataset_match": self.dataset_match,
                    "optional_positive_folder": str(stage.stage_dir / _slug(stage.positive_label)),
                    "optional_negative_folder": str(stage.stage_dir / _slug(stage.negative_label)),
                    "query": stage.query,
                }
            )
        return pd.DataFrame(rows)

    def write_stage_plan(self, path: str | Path | None = None) -> pd.DataFrame:
        """Write the stage plan CSV and return it."""
        plan = self.stage_plan()
        output_path = Path(path) if path is not None else self.dataset_root / "benchmark_stage_plan.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(output_path, index=False)
        return plan

    def plot(self, metric: str = "positives_in_top_k"):
        """Plot a simple per-stage line chart for the selected metric."""
        if self.summary_.empty:
            print("No benchmark results yet. Run benchmark.run() first.")
            return None
        if metric not in self.summary_.columns:
            raise ValueError(f"Unknown metric {metric!r}. Available columns: {list(self.summary_.columns)}")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(self.summary_["feature"], self.summary_[metric], marker="o")
        if metric == "positives_in_top_k":
            ax.axhline(self.n, color="black", linestyle="--", linewidth=1, label="perfect")
            ax.set_ylim(0, self.n + 0.5)
            ax.legend()
        ax.set_xlabel("Feature stage")
        ax.set_ylabel(metric)
        ax.set_title(f"Filler selector benchmark ({self.mode})")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()

        try:
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except ImportError:
            plt.show()
        return fig

    def print(self) -> None:
        print(self.summary_text())

    def summary_text(self) -> str:
        lines = [
            "FillerSelectionBenchmark",
            f"  description: {self.verbal_description}",
            f"  mode: {self.mode}",
            f"  n per label: {self.n}",
            f"  dataset_root: {self.dataset_root}",
            f"  images_dir: {self.dataset.images_dir}",
            f"  manifest: {self.dataset.manifest_path}",
            f"  dataset_match: {self.dataset_match}",
            f"  attribute_schema: {self.attribute_schema.name}",
            f"  case_slug: {self.case_slug}",
            f"  selector: {self.selector}",
            f"  selector_model: {_selector_model_label(self.selector, self.selector_model)}",
            "  stages:",
        ]
        for stage in self.stages:
            lines.append(
                f"    {stage.stage_name}: {stage.positive_label} vs {stage.negative_label} | {stage.query}"
            )
        return "\n".join(lines)

    def _build_stages(self) -> list[BenchmarkStage]:
        active_positive = {}
        stages = []
        step = 0
        schema_values = self.schema.attribute_values()

        for feature in self.attribute_schema.feature_order:
            positive_value = schema_values.get(feature)
            if not positive_value:
                continue
            negative_value = self._contrast_for(feature, positive_value)
            if not negative_value:
                print(f"[FillerSelectionBenchmark] Skipping {feature}: no contrast label for {positive_value!r}.")
                continue

            step += 1
            if self.mode == "single":
                positive_fields = {feature: positive_value}
                negative_fields = {feature: negative_value}
                stage_dir = self.mode_dir / f"{step:02d}_{feature}"
            else:
                positive_fields = {**active_positive, feature: positive_value}
                negative_fields = {**active_positive, feature: negative_value}
                active_positive[feature] = positive_value
                stage_dir = self.mode_dir / f"{step:02d}_{feature}"

            positive_schema = _schema_from_fields(
                original_description=self.verbal_description,
                fields=positive_fields,
            )
            negative_schema = _schema_from_fields(
                original_description=self.verbal_description,
                fields=negative_fields,
            )
            stages.append(
                BenchmarkStage(
                    mode=self.mode,
                    step=step,
                    feature=feature,
                    positive_label=positive_value,
                    negative_label=negative_value,
                    query=_schema_query(positive_schema),
                    positive_schema=positive_schema,
                    negative_schema=negative_schema,
                    stage_dir=stage_dir,
                )
            )

        if not stages:
            raise ValueError(
                "No benchmarkable schema features were parsed from the description. "
                "Try a more explicit description, e.g. 'a white male with beard and blue eyes'."
            )
        return stages

    def _contrast_for(self, feature: str, positive_value: str) -> str | None:
        override = self.contrast_overrides.get(feature)
        if override:
            return override
        if feature == "age":
            numeric_age = re.search(r"\b(\d{2})\b", str(positive_value))
            if numeric_age:
                return "older adult" if int(numeric_age.group(1)) < 50 else "young adult"
        return self.attribute_schema.contrast_for(
            feature,
            positive_value,
            override=self.contrast_overrides.get(feature),
        )

    def _ensure_stage_dataset(self, stage: BenchmarkStage, generate_missing: bool) -> pd.DataFrame:
        positive = self.dataset.select(
            stage.positive_schema,
            n=self.n,
            match=self.dataset_match,
            newest_first=self.overwrite_generated,
        )
        negative = self.dataset.select(
            stage.negative_schema,
            n=self.n,
            exclude_image_ids=set(positive["image_id"].astype(str)) if not positive.empty else set(),
            match=self.dataset_match,
            newest_first=self.overwrite_generated,
        )

        if len(positive) < self.n or len(negative) < self.n:
            self._import_legacy_stage_folders(stage, copy_to_images=True)
            positive = self.dataset.select(
                stage.positive_schema,
                n=self.n,
                match=self.dataset_match,
                newest_first=self.overwrite_generated,
            )
            negative = self.dataset.select(
                stage.negative_schema,
                n=self.n,
                exclude_image_ids=set(positive["image_id"].astype(str)) if not positive.empty else set(),
                match=self.dataset_match,
                newest_first=self.overwrite_generated,
            )

        if generate_missing:
            if self.overwrite_generated:
                positive_missing = self.n
                negative_missing = self.n
            else:
                positive_missing = self.n - len(positive)
                negative_missing = self.n - len(negative)
            self._generate_label_images(stage, "positive", stage.positive_schema, positive_missing)
            self._generate_label_images(stage, "negative", stage.negative_schema, negative_missing)
            positive = self.dataset.select(
                stage.positive_schema,
                n=self.n,
                match=self.dataset_match,
                newest_first=self.overwrite_generated,
            )
            negative = self.dataset.select(
                stage.negative_schema,
                n=self.n,
                exclude_image_ids=set(positive["image_id"].astype(str)) if not positive.empty else set(),
                match=self.dataset_match,
                newest_first=self.overwrite_generated,
            )

        rows = []
        rows.extend(self._stage_rows(stage, "positive", stage.positive_label, positive))
        rows.extend(self._stage_rows(stage, "negative", stage.negative_label, negative))
        return pd.DataFrame(rows)

    def _generate_label_images(
        self,
        stage: BenchmarkStage,
        label_role: str,
        schema: FaceDescriptionSchema,
        missing: int,
    ) -> None:
        if missing <= 0:
            return

        print(f"Generating {missing} {label_role} images for {stage.stage_name} into {self.dataset.images_dir}")
        self.dataset.generate_fillers(
            schema,
            verbal_description=_schema_query(schema),
            n=missing,
            model=self.generator_model,
            provider=self.generator_provider,
            backend=self.generator_backend,
            backend_kwargs=self.generator_backend_kwargs,
            size=self.image_size,
            quality=self.image_quality,
            output_format=self.image_format,
            overwrite=False,
            source_mode=self.mode,
            source_stage=stage.stage_name,
            source_label_role=label_role,
        )

    def _import_legacy_stage_folders(self, stage: BenchmarkStage, *, copy_to_images: bool = True) -> list[dict[str, object]]:
        positive_dir = stage.stage_dir / _slug(stage.positive_label)
        negative_dir = stage.stage_dir / _slug(stage.negative_label)
        positive_paths = _image_files(positive_dir)
        negative_paths = _image_files(negative_dir)
        imported_rows: list[dict[str, object]] = []
        if positive_paths:
            imported = self.dataset.import_images(
                positive_paths,
                schema=stage.positive_schema,
                verbal_description=_schema_query(stage.positive_schema),
                source="stage_folder",
                source_mode=self.mode,
                source_stage=stage.stage_name,
                source_label_role="positive",
                copy_to_images=copy_to_images,
            )
            imported_rows.extend(imported.to_dict("records"))
        if negative_paths:
            imported = self.dataset.import_images(
                negative_paths,
                schema=stage.negative_schema,
                verbal_description=_schema_query(stage.negative_schema),
                source="stage_folder",
                source_mode=self.mode,
                source_stage=stage.stage_name,
                source_label_role="negative",
                copy_to_images=copy_to_images,
            )
            imported_rows.extend(imported.to_dict("records"))
        return imported_rows

    def _stage_rows(
        self,
        stage: BenchmarkStage,
        label_role: str,
        label: str,
        selected: pd.DataFrame,
    ) -> list[dict[str, object]]:
        rows = []
        for index, (_, source_row) in enumerate(selected.iterrows(), start=1):
            row = source_row.to_dict()
            row.update(
                {
                    "target_id": f"{label_role}_{index:03d}",
                    "benchmark_mode": self.mode,
                    "case_slug": self.case_slug,
                    "stage": stage.stage_name,
                    "step": stage.step,
                    "feature": stage.feature,
                    "label_role": label_role,
                    "label": label,
                    "is_positive": label_role == "positive",
                    "query": stage.query,
                }
            )
            rows.append(row)
        return rows

    def export_stage_folders(self, *, copy: bool = False) -> pd.DataFrame:
        """Create or refresh human-friendly stage folders from manifest selections.

        The benchmark itself is manifest-first. This method is only a view for
        manual inspection or external tools that expect folders.
        """
        rows = []
        for stage in self.stages:
            stage_manifest = self._ensure_stage_dataset(stage, generate_missing=False)
            for _, row in stage_manifest.iterrows():
                target_dir = stage.stage_dir / _slug(row["label"])
                target_dir.mkdir(parents=True, exist_ok=True)
                source_path = Path(str(row["image_path"]))
                target_path = target_dir / source_path.name
                if copy and source_path.exists() and not target_path.exists():
                    import shutil

                    shutil.copy2(source_path, target_path)
                rows.append(
                    {
                        "mode": self.mode,
                        "stage": stage.stage_name,
                        "label": row["label"],
                        "source_path": str(source_path),
                        "view_path": str(target_path),
                    }
                )
        return pd.DataFrame(rows)

    def import_stage_folders(self, *, copy_to_images: bool = True) -> pd.DataFrame:
        """Import optional stage-folder images into the manifest-first dataset."""
        rows = []
        for stage in self.stages:
            rows.extend(self._import_legacy_stage_folders(stage, copy_to_images=copy_to_images))
        return pd.DataFrame(rows)

    def _run_selector(
        self,
        stage: BenchmarkStage,
        manifest: pd.DataFrame,
        rebuild_index: bool,
    ) -> pd.DataFrame:
        if isinstance(self.selector, SelectorBackend):
            raw = self.selector.score(manifest.copy(), SelectorQuery.from_text(stage.query))
            return normalise_selector_scores(
                manifest,
                raw,
                backend_name=self.selector.name,
                model_name=self.selector.model_name,
            )
        if callable(self.selector):
            raw = self.selector(manifest.copy(), stage.query, len(manifest))
            raw = _adapt_legacy_selector_results(raw)
            return normalise_selector_scores(
                manifest,
                raw,
                backend_name="callable",
                model_name=self.selector_model,
            )
        if self.selector != "clip":
            raise ValueError(
                "selector must be 'clip', a SelectorBackend, or a "
                "callable(manifest, query, top_k)."
            )
        return self._run_clip_selector(stage, manifest, rebuild_index=rebuild_index)

    def _run_clip_selector(
        self,
        stage: BenchmarkStage,
        manifest: pd.DataFrame,
        rebuild_index: bool,
    ) -> pd.DataFrame:
        index_dir = self.results_root / "indices" / self.selector_model.replace("/", "_").replace("-", "_") / self.mode / self.case_slug / stage.stage_name
        backend = ClipSelectorBackend(
            model_name=self.selector_model,
            device=self.device,
            index_dir=index_dir,
            show_progress=True,
            rebuild_index=rebuild_index,
            preprocess="whole_image",
        )
        raw = backend.score(manifest.copy(), SelectorQuery.from_text(stage.query))
        return normalise_selector_scores(
            manifest,
            raw,
            backend_name=backend.name,
            model_name=backend.model_name,
        )

    def _summarise_stage(self, stage: BenchmarkStage, ranked: pd.DataFrame) -> dict[str, object]:
        top_k = min(self.n, len(ranked))
        top = ranked.head(top_k)
        positives_in_top_k = int(top["is_positive"].sum())
        positive_ranks = ranked.loc[ranked["is_positive"], "rank"]
        negative_ranks = ranked.loc[~ranked["is_positive"], "rank"]
        return {
            "mode": self.mode,
            "case_slug": self.case_slug,
            "stage": stage.stage_name,
            "step": stage.step,
            "feature": stage.feature,
            "positive_label": stage.positive_label,
            "negative_label": stage.negative_label,
            "query": stage.query,
            "n_positive": int(ranked["is_positive"].sum()),
            "n_negative": int((~ranked["is_positive"]).sum()),
            "top_k": top_k,
            "positives_in_top_k": positives_in_top_k,
            "precision_at_k": float(positives_in_top_k / top_k) if top_k else 0.0,
            "recall_at_k": float(positives_in_top_k / self.n),
            "perfect_top_k": bool(positives_in_top_k == self.n),
            "mean_positive_rank": float(positive_ranks.mean()) if len(positive_ranks) else None,
            "worst_positive_rank": float(positive_ranks.max()) if len(positive_ranks) else None,
            "first_negative_rank": float(negative_ranks.min()) if len(negative_ranks) else None,
        }

    def _display_stage(self, stage: BenchmarkStage, ranked: pd.DataFrame, summary: dict[str, object], enabled: bool) -> None:
        if not enabled:
            return
        try:
            from IPython.display import Markdown, display
            from .cfd_clip_pilot.visualization import show_clip_lineup
        except ImportError:
            print(f"{stage.stage_name}: {summary['positives_in_top_k']} / {summary['top_k']} positives in top-k")
            return

        display(
            Markdown(
                f"**{self.mode} | {stage.stage_name}: {stage.feature}**  <br>"
                f"Positive: `{stage.positive_label}` | Negative: `{stage.negative_label}`  <br>"
                f"Query: {stage.query}  <br>"
                f"Top-{summary['top_k']} positives: **{summary['positives_in_top_k']} / {summary['n_positive']}**"
            )
        )
        fig = show_clip_lineup(
            ranked,
            top_k=int(summary["top_k"]),
            cols=3,
            target_position="none",
            score_col="selector_score",
            id_col="label",
            suptitle=None,
        )
        display(fig)
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            pass
        display(ranked.head(int(summary["top_k"]))[["rank", "label_role", "label", "selector_score", "image_path"]])

    def _write_results(self) -> None:
        if self.results_.empty:
            return
        output_dir = self.results_root / self.mode / self.case_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        self.results_.to_csv(output_dir / "ranked_results.csv", index=False)
        self.summary_.to_csv(output_dir / "summary.csv", index=False)
        metadata = {
            "verbal_description": self.verbal_description,
            "n": self.n,
            "mode": self.mode,
            "case_slug": self.case_slug,
            "dataset_root": str(self.dataset_root),
            "selector": _selector_label(self.selector),
            "selector_model": _selector_model_label(self.selector, self.selector_model),
            "generator_provider": self.generator_provider,
            "generator_model": self.generator_model,
            "schema": self.schema.to_dict(),
            "attribute_schema": self.attribute_schema.to_records(),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return (
            f"FillerSelectionBenchmark(description={self.verbal_description!r}, n={self.n}, "
            f"mode={self.mode!r}, dataset_root={str(self.dataset_root)!r})"
        )


FillerSelectorBenchmark = FillerSelectionBenchmark


def _schema_from_fields(original_description: str, fields: dict[str, str]) -> FaceDescriptionSchema:
    return FaceDescriptionSchema.from_attributes(original_description, fields)


def _selector_label(selector: object) -> str:
    if isinstance(selector, SelectorBackend):
        return selector.name
    return selector if isinstance(selector, str) else "callable"


def _selector_model_label(selector: object, fallback: str) -> str:
    return selector.model_name if isinstance(selector, SelectorBackend) else fallback


def _adapt_legacy_selector_results(results: pd.DataFrame) -> pd.DataFrame:
    """Preserve rank-only legacy callables while the shared backend stays strict."""
    if not isinstance(results, pd.DataFrame):
        return results
    score_columns = {"selector_score", "decision_score", "score", "clip_score"}
    if score_columns.intersection(results.columns):
        return results

    output = results.copy()
    if "rank" in output.columns:
        ranks = pd.to_numeric(output["rank"], errors="coerce")
        if ranks.isna().any():
            raise ValueError("Legacy selector rank values must be numeric when no score is returned.")
        output["selector_score"] = -ranks.astype(float)
    else:
        output["selector_score"] = [-float(index) for index in range(len(output))]
    return output


def _schema_query(schema: FaceDescriptionSchema) -> str:
    subject = schema.subject_phrase()
    details = schema.detail_phrases()
    article = "an" if subject[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    sentence = f"A frontal face photograph of {article} {subject}"
    if details:
        sentence += " with " + ", ".join(details)
    return sentence + "."


def _schema_filter_text(schema: FaceDescriptionSchema) -> str:
    parts = [f"{key}={value}" for key, value in schema.attribute_values().items()]
    other_details = schema.other_details
    if other_details:
        parts.append("other_details=" + " | ".join(other_details))
    return "; ".join(parts)


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return cleaned or "unknown"


def _case_slug(text: str) -> str:
    base = _slug(text)[:48]
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def _image_files(folder: Path) -> list[Path]:
    return discover_images(folder, recursive=False, extensions=IMAGE_EXTENSIONS)
