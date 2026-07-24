from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd

from .FaceAttributeSchema import DEFAULT_FACE_ATTRIBUTE_SCHEMA, FaceAttributeSchema
from .FaceSearch import (
    ClipSelectorBackend,
    SelectorBackend,
    SelectorQuery,
    normalise_selector_scores,
)
from .ImageCatalog import discover_images

ScoreMode = Literal["auto", "positive", "contrastive"]
AttributeSelector = Callable[[pd.DataFrame, "AttributeProbeSpec"], pd.DataFrame]

DEFAULT_RESULTS_ROOT = Path("./build/attribute_retrieval_benchmark")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

__all__ = [
    "AttributeProbeSpec",
    "AttributeRetrievalBenchmark",
    "AttributeSelector",
    "ScoreMode",
]


@dataclass(frozen=True)
class AttributeProbeSpec:
    """Labels and text prompts for one controlled binary attribute probe."""

    attribute: str
    positive_label: str
    negative_label: str
    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        attribute = _slug(self.attribute)
        positive_prompts = _prompt_tuple(self.positive_prompts)
        negative_prompts = _prompt_tuple(self.negative_prompts)
        if not attribute:
            raise ValueError("attribute must not be empty.")
        if not str(self.positive_label).strip() or not str(self.negative_label).strip():
            raise ValueError("positive_label and negative_label must not be empty.")
        if not positive_prompts:
            raise ValueError("At least one positive prompt is required.")
        object.__setattr__(self, "attribute", attribute)
        object.__setattr__(self, "positive_label", str(self.positive_label).strip())
        object.__setattr__(self, "negative_label", str(self.negative_label).strip())
        object.__setattr__(self, "positive_prompts", positive_prompts)
        object.__setattr__(self, "negative_prompts", negative_prompts)

    @classmethod
    def from_schema(
        cls,
        attribute: str,
        positive_label: str,
        *,
        negative_label: str | None = None,
        attribute_schema: FaceAttributeSchema = DEFAULT_FACE_ATTRIBUTE_SCHEMA,
        positive_prompts: str | Sequence[str] | None = None,
        negative_prompts: str | Sequence[str] | None = None,
    ) -> "AttributeProbeSpec":
        definition = attribute_schema.require(attribute)
        negative = negative_label or definition.contrast_for(positive_label)
        if not negative:
            raise ValueError(
                f"No contrast is defined for {attribute!r}={positive_label!r}; pass negative_label explicitly."
            )
        positive_texts = positive_prompts or (_default_prompt(definition.prompt_role, positive_label),)
        negative_texts = negative_prompts or (_default_prompt(definition.prompt_role, negative),)
        return cls(
            attribute=definition.name,
            positive_label=positive_label,
            negative_label=negative,
            positive_prompts=_prompt_tuple(positive_texts),
            negative_prompts=_prompt_tuple(negative_texts),
        )

    @property
    def slug(self) -> str:
        payload = "|".join(
            (
                self.attribute,
                self.positive_label,
                self.negative_label,
                *self.positive_prompts,
                *self.negative_prompts,
            )
        )
        return f"{self.attribute}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:8]}"


class AttributeRetrievalBenchmark:
    """Benchmark any binary visual attribute with CLIP or a custom selector."""

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        probe: AttributeProbeSpec,
        *,
        label_column: str = "label",
        image_path_column: str = "image_path",
        image_id_column: str = "image_id",
        selector: Literal["clip"] | AttributeSelector | SelectorBackend = "clip",
        selector_model: str = "clip-ViT-B-32",
        device: str | None = None,
        score_mode: ScoreMode = "auto",
        top_k: int | None = None,
        decision_threshold: float = 0.0,
        results_root: str | Path = DEFAULT_RESULTS_ROOT,
    ) -> None:
        if score_mode not in {"auto", "positive", "contrastive"}:
            raise ValueError("score_mode must be 'auto', 'positive', or 'contrastive'.")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1.")

        self.probe = probe
        self.label_column = label_column
        self.image_path_column = image_path_column
        self.image_id_column = image_id_column
        self.selector = selector
        self.selector_model = selector_model
        self.device = device
        self.score_mode = score_mode
        self.top_k = top_k
        self.decision_threshold = float(decision_threshold)
        self.results_root = Path(results_root)
        self.manifest_source = manifest
        self.manifest = self._prepare_manifest(manifest)
        self.results_: pd.DataFrame = pd.DataFrame()
        self.summary_: pd.DataFrame = pd.DataFrame()

    @classmethod
    def from_folders(
        cls,
        positive_dir: str | Path,
        negative_dir: str | Path,
        *,
        probe: AttributeProbeSpec | None = None,
        attribute: str | None = None,
        positive_label: str | None = None,
        negative_label: str | None = None,
        positive_prompts: str | Sequence[str] = (),
        negative_prompts: str | Sequence[str] = (),
        recursive: bool = True,
        **kwargs,
    ) -> "AttributeRetrievalBenchmark":
        positive_paths = _image_files(Path(positive_dir), recursive=recursive)
        negative_paths = _image_files(Path(negative_dir), recursive=recursive)
        if not positive_paths:
            raise FileNotFoundError(f"No images found in positive_dir: {positive_dir}")
        if not negative_paths:
            raise FileNotFoundError(f"No images found in negative_dir: {negative_dir}")

        if probe is None:
            if not attribute or not positive_label or not negative_label:
                raise ValueError(
                    "Pass probe=AttributeProbeSpec(...) or provide attribute, positive_label, and negative_label."
                )
            probe = AttributeProbeSpec(
                attribute=attribute,
                positive_label=positive_label,
                negative_label=negative_label,
                positive_prompts=_prompt_tuple(positive_prompts),
                negative_prompts=_prompt_tuple(negative_prompts),
            )

        records = []
        for label, paths in (
            (probe.positive_label, positive_paths),
            (probe.negative_label, negative_paths),
        ):
            for path in paths:
                resolved = path.resolve()
                image_id = f"image_{hashlib.sha1(str(resolved).encode('utf-8')).hexdigest()[:12]}"
                records.append(
                    {
                        "image_id": image_id,
                        "target_id": image_id,
                        "image_path": str(resolved),
                        "label": label,
                    }
                )
        return cls(pd.DataFrame.from_records(records), probe, **kwargs)

    def run(
        self,
        *,
        display: bool = True,
        show_top_k: int | None = None,
        rebuild_index: bool = False,
    ) -> pd.DataFrame:
        """Score all images, compute metrics, save outputs, and optionally show top results."""
        print(
            f"Running attribute probe: {self.probe.attribute} | "
            f"{self.probe.positive_label} vs {self.probe.negative_label}"
        )
        if isinstance(self.selector, SelectorBackend):
            scored = self._run_selector_backend()
        elif callable(self.selector):
            scored = self._run_custom_selector()
        elif self.selector == "clip":
            scored = self._run_clip(rebuild_index=rebuild_index)
        else:
            raise ValueError(
                "selector must be 'clip', a SelectorBackend, or a callable(manifest, probe)."
            )

        self.results_ = self._rank(scored)
        self.summary_ = pd.DataFrame([self._summarise()])
        self._write_results()
        if display:
            self.show(top_k=show_top_k)
            self.statistics(display=True)
        return self.results_

    def statistics(self, display: bool = True) -> pd.DataFrame:
        if self.summary_.empty:
            print("No benchmark results yet. Run benchmark.run() first.")
            return self.summary_
        if display:
            try:
                from IPython.display import display as ipy_display

                ipy_display(self.summary_)
            except ImportError:
                print(self.summary_.to_string(index=False))
        return self.summary_.copy()

    def print_statistics(self) -> pd.DataFrame:
        return self.statistics(display=True)

    def show(self, top_k: int | None = None, *, cols: int = 5):
        if self.results_.empty:
            print("No benchmark results yet. Run benchmark.run() first.")
            return None
        from .cfd_clip_pilot.visualization import show_clip_lineup

        n_show = min(top_k or self._evaluation_k(), len(self.results_))
        display_rows = self.results_.copy()
        display_rows["display_label"] = display_rows.apply(
            lambda row: f"{row['label']} | {'positive' if row['is_positive'] else 'negative'}",
            axis=1,
        )
        fig = show_clip_lineup(
            display_rows,
            top_k=n_show,
            cols=cols,
            target_position="none",
            score_col="selector_score",
            id_col="display_label",
            suptitle=None,
        )
        try:
            from IPython.display import Markdown, display

            display(
                Markdown(
                    f"**{self.probe.attribute}: {self.probe.positive_label} vs "
                    f"{self.probe.negative_label}**  <br>"
                    f"Positive prompts: {' | '.join(self.probe.positive_prompts)}  <br>"
                    f"Negative prompts: {' | '.join(self.probe.negative_prompts) or '(none)'}"
                )
            )
            display(fig)
            import matplotlib.pyplot as plt

            plt.close(fig)
        except ImportError:
            import matplotlib.pyplot as plt

            plt.show()
        return fig

    def plot(self):
        if self.results_.empty:
            print("No benchmark results yet. Run benchmark.run() first.")
            return None
        import matplotlib.pyplot as plt

        ranked = self.results_.sort_values("rank")
        ideal = np.minimum(np.arange(1, len(ranked) + 1), int(ranked["is_positive"].sum()))
        cumulative = ranked["is_positive"].astype(int).cumsum().to_numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        colours = np.where(ranked["is_positive"], "#18794e", "#c43c35")
        axes[0].scatter(ranked["rank"], ranked["selector_score"], c=colours, s=28)
        axes[0].axvline(self._evaluation_k(), color="black", linestyle="--", linewidth=1)
        axes[0].set(xlabel="Rank", ylabel="Selector score", title="Ranked scores")

        axes[1].plot(ranked["rank"], cumulative, label="observed", color="#2563a6")
        axes[1].plot(ranked["rank"], ideal, label="perfect", color="black", linestyle="--")
        axes[1].set(xlabel="Retrieved images", ylabel="Positive images found", title="Cumulative retrieval")
        axes[1].legend()
        fig.suptitle(self.probe.attribute)
        fig.tight_layout()
        try:
            from IPython.display import display

            display(fig)
            plt.close(fig)
        except ImportError:
            plt.show()
        return fig

    def summary_text(self) -> str:
        score_mode = self._resolved_score_mode()
        return "\n".join(
            (
                "AttributeRetrievalBenchmark",
                f"  attribute: {self.probe.attribute}",
                f"  positive: {self.probe.positive_label}",
                f"  negative: {self.probe.negative_label}",
                f"  positive prompts: {' | '.join(self.probe.positive_prompts)}",
                f"  negative prompts: {' | '.join(self.probe.negative_prompts) or '(none)'}",
                f"  images: {len(self.manifest)}",
                f"  selector: {_selector_label(self.selector)}",
                f"  selector_model: {_selector_model_label(self.selector, self.selector_model)}",
                f"  score_mode: {score_mode}",
                f"  top_k: {self._evaluation_k()}",
            )
        )

    def print(self) -> None:
        print(self.summary_text())

    def _prepare_manifest(self, source: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            manifest = source.copy()
            base_dir = Path.cwd()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Manifest not found: {path}")
            manifest = pd.read_csv(path)
            base_dir = path.resolve().parent

        required = {self.image_path_column, self.label_column}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        output = manifest.copy()
        output["image_path"] = output[self.image_path_column].map(
            lambda value: str(_resolve_path(value, base_dir))
        )
        missing_files = [path for path in output["image_path"] if not Path(path).exists()]
        if missing_files:
            preview = missing_files[:3]
            raise FileNotFoundError(f"Manifest contains missing image files: {preview}")

        output["label"] = output[self.label_column].astype(str).str.strip()
        allowed = {self.probe.positive_label, self.probe.negative_label}
        unknown = sorted(set(output["label"]) - allowed)
        if unknown:
            raise ValueError(
                f"Manifest contains labels outside the probe's positive/negative labels: {unknown}"
            )
        output["is_positive"] = output["label"] == self.probe.positive_label

        id_source = (
            self.image_id_column
            if self.image_id_column in output.columns
            else "image_id" if "image_id" in output.columns else None
        )
        if id_source is not None:
            output["image_id"] = _normalise_identifiers(output[id_source], name="image_id")
        else:
            output["image_id"] = output["image_path"].map(
                lambda value: f"image_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12]}"
            )
        if output["image_id"].duplicated().any():
            duplicates = output.loc[output["image_id"].duplicated(), "image_id"].tolist()[:3]
            raise ValueError(f"image_id values must be unique; duplicates include {duplicates}")
        output["target_id"] = (
            _normalise_identifiers(output["target_id"], name="target_id")
            if "target_id" in output.columns
            else output["image_id"]
        )

        if not output["is_positive"].any() or output["is_positive"].all():
            raise ValueError("Manifest must contain at least one positive and one negative image.")
        return output.reset_index(drop=True)

    def _run_clip(self, *, rebuild_index: bool) -> pd.DataFrame:
        backend = ClipSelectorBackend(
            model_name=self.selector_model,
            device=self.device,
            index_dir=self._index_dir(),
            show_progress=True,
            rebuild_index=rebuild_index,
            preprocess="whole_image",
        )
        raw = backend.score(self.manifest.copy(), self._selector_query())
        return normalise_selector_scores(
            self.manifest,
            raw,
            backend_name=backend.name,
            model_name=backend.model_name,
        )

    def _run_selector_backend(self) -> pd.DataFrame:
        backend = self.selector
        raw = backend.score(self.manifest.copy(), self._selector_query())
        return normalise_selector_scores(
            self.manifest,
            raw,
            backend_name=backend.name,
            model_name=backend.model_name,
        )

    def _run_custom_selector(self) -> pd.DataFrame:
        raw = self.selector(self.manifest.copy(), self.probe)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError("Custom selector must return a pandas DataFrame.")
        return normalise_selector_scores(
            self.manifest,
            raw,
            backend_name="callable",
            model_name=self.selector_model,
        )

    def _selector_query(self) -> SelectorQuery:
        return SelectorQuery(
            positive_texts=self.probe.positive_prompts,
            negative_texts=self.probe.negative_prompts,
            score_mode=self._resolved_score_mode(),
            description=self.probe.positive_prompts[0],
        )

    def _rank(self, scored: pd.DataFrame) -> pd.DataFrame:
        if len(scored) != len(self.manifest):
            raise ValueError(
                f"Selector returned {len(scored)} rows for a {len(self.manifest)}-image manifest."
            )
        output = scored.sort_values("selector_score", ascending=False, kind="mergesort").reset_index(drop=True)
        output["rank"] = np.arange(1, len(output) + 1)
        if (
            self.probe.negative_prompts
            or callable(self.selector)
            or isinstance(self.selector, SelectorBackend)
        ):
            output["predicted_positive"] = output["selector_score"] >= self.decision_threshold
        else:
            output["predicted_positive"] = pd.NA
        return output

    def _summarise(self) -> dict[str, object]:
        ranked = self.results_
        labels = ranked["is_positive"].astype(bool).to_numpy()
        scores = ranked["selector_score"].astype(float).to_numpy()
        k = self._evaluation_k()
        positives = int(labels.sum())
        negatives = int((~labels).sum())
        positives_at_k = int(labels[:k].sum())
        positive_ranks = ranked.loc[ranked["is_positive"], "rank"]
        negative_ranks = ranked.loc[~ranked["is_positive"], "rank"]

        accuracy = np.nan
        balanced_accuracy = np.nan
        if ranked["predicted_positive"].notna().all():
            predicted = ranked["predicted_positive"].astype(bool).to_numpy()
            true_positive_rate = float(predicted[labels].mean())
            true_negative_rate = float((~predicted[~labels]).mean())
            accuracy = float((predicted == labels).mean())
            balanced_accuracy = (true_positive_rate + true_negative_rate) / 2

        return {
            "attribute": self.probe.attribute,
            "positive_label": self.probe.positive_label,
            "negative_label": self.probe.negative_label,
            "selector": _selector_label(self.selector),
            "selector_model": _selector_model_label(self.selector, self.selector_model),
            "score_mode": self._resolved_score_mode(),
            "n_positive": positives,
            "n_negative": negatives,
            "top_k": k,
            "positives_in_top_k": positives_at_k,
            "precision_at_k": positives_at_k / k,
            "recall_at_k": positives_at_k / positives,
            "average_precision": _average_precision(labels),
            "roc_auc": _roc_auc(labels, scores),
            "threshold_accuracy": accuracy,
            "threshold_balanced_accuracy": balanced_accuracy,
            "mean_positive_rank": float(positive_ranks.mean()),
            "worst_positive_rank": int(positive_ranks.max()),
            "first_negative_rank": int(negative_ranks.min()),
            "mean_positive_score": float(scores[labels].mean()),
            "mean_negative_score": float(scores[~labels].mean()),
        }

    def _evaluation_k(self) -> int:
        positives = int(self.manifest["is_positive"].sum())
        return min(self.top_k or positives, len(self.manifest))

    def _resolved_score_mode(self) -> Literal["positive", "contrastive"]:
        if self.score_mode == "auto":
            return "contrastive" if self.probe.negative_prompts else "positive"
        if self.score_mode == "contrastive" and not self.probe.negative_prompts:
            raise ValueError("score_mode='contrastive' requires at least one negative prompt.")
        return self.score_mode

    def _index_dir(self) -> Path:
        model_slug = _slug(self.selector_model)
        return self.results_root / "indices" / model_slug / self.probe.slug

    def _write_results(self) -> None:
        output_dir = self.results_root / self.probe.slug
        output_dir.mkdir(parents=True, exist_ok=True)
        self.results_.to_csv(output_dir / "ranked_results.csv", index=False)
        self.summary_.to_csv(output_dir / "summary.csv", index=False)
        metadata = {
            "probe": {
                "attribute": self.probe.attribute,
                "positive_label": self.probe.positive_label,
                "negative_label": self.probe.negative_label,
                "positive_prompts": list(self.probe.positive_prompts),
                "negative_prompts": list(self.probe.negative_prompts),
            },
            "selector": _selector_label(self.selector),
            "selector_model": _selector_model_label(self.selector, self.selector_model),
            "score_mode": self._resolved_score_mode(),
            "decision_threshold": self.decision_threshold,
            "n_images": len(self.manifest),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def __repr__(self) -> str:
        return (
            f"AttributeRetrievalBenchmark(attribute={self.probe.attribute!r}, "
            f"n_images={len(self.manifest)}, "
            f"selector_model={_selector_model_label(self.selector, self.selector_model)!r})"
        )


def _prompt_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    else:
        values = tuple(value)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _selector_label(selector: object) -> str:
    if isinstance(selector, SelectorBackend):
        return selector.name
    return selector if isinstance(selector, str) else "callable"


def _selector_model_label(selector: object, fallback: str) -> str:
    return selector.model_name if isinstance(selector, SelectorBackend) else fallback


def _normalise_identifiers(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} values must not be empty.")
    return values.astype(str)


def _default_prompt(role: str, value: str) -> str:
    if role == "subject":
        subject = f"{value} person" if "adult" in value.lower() else f"{value} adult person"
        article = "an" if subject[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        return f"A frontal face photograph of {article} {subject}."
    return f"A frontal face photograph of an adult person with {value}."


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _image_files(folder: Path, *, recursive: bool) -> list[Path]:
    return discover_images(folder, recursive=recursive, extensions=IMAGE_EXTENSIONS)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _average_precision(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(labels) / np.arange(1, len(labels) + 1)
    return float(precision[labels].sum() / positives)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average", ascending=True).to_numpy()
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
