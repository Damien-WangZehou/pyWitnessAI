from __future__ import annotations

import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .ImageCatalog import (
    CatalogReport,
    ImageCatalog,
    MetadataJoinSpec,
    MetadataSource,
)


ScoreMode = Literal["auto", "positive", "contrastive"]
SelectorFactory = Callable[..., "SelectorBackend"]

__all__ = [
    "AllRankedPolicy",
    "CallableSelectorBackend",
    "ClipSelectorBackend",
    "FaceSearch",
    "GroundTruthSpec",
    "MatchPolicy",
    "MinScorePolicy",
    "SelectionResult",
    "SelectorBackend",
    "SelectorQuery",
    "TopKAndMinScorePolicy",
    "TopKPolicy",
    "available_selector_backends",
    "create_selector_backend",
    "normalise_selector_scores",
    "register_selector_backend",
]


@dataclass(frozen=True)
class SelectorQuery:
    """Text query understood by every selector backend."""

    positive_texts: tuple[str, ...] | str
    negative_texts: tuple[str, ...] | str = ()
    score_mode: ScoreMode = "auto"
    description: str = ""

    def __post_init__(self) -> None:
        positive = _text_tuple(self.positive_texts)
        negative = _text_tuple(self.negative_texts)
        if not positive:
            raise ValueError("SelectorQuery requires at least one positive text.")
        if self.score_mode not in {"auto", "positive", "contrastive"}:
            raise ValueError("score_mode must be 'auto', 'positive', or 'contrastive'.")
        if self.score_mode == "contrastive" and not negative:
            raise ValueError("score_mode='contrastive' requires negative_texts.")
        object.__setattr__(self, "positive_texts", positive)
        object.__setattr__(self, "negative_texts", negative)
        if not self.description:
            object.__setattr__(self, "description", positive[0])

    @classmethod
    def from_text(cls, description: str) -> "SelectorQuery":
        text = str(description).strip()
        if not text:
            raise ValueError("description must not be empty.")
        return cls(positive_texts=(text,), description=text)

    @property
    def resolved_score_mode(self) -> Literal["positive", "contrastive"]:
        if self.score_mode == "auto":
            return "contrastive" if self.negative_texts else "positive"
        return self.score_mode


class SelectorBackend(ABC):
    """Backend contract: score every candidate and identify rows by image_id."""

    name: str = "custom"
    model_name: str = ""
    preprocess: str = "whole_image"

    @abstractmethod
    def score(self, manifest: pd.DataFrame, query: SelectorQuery) -> pd.DataFrame:
        """Return one row per candidate with image_id and selector_score."""


class CallableSelectorBackend(SelectorBackend):
    """Adapt a callable(manifest, SelectorQuery) to the backend contract."""

    def __init__(
        self,
        selector: Callable[[pd.DataFrame, SelectorQuery], pd.DataFrame],
        *,
        name: str = "callable",
        model_name: str = "",
    ) -> None:
        self.selector = selector
        self.name = str(name).strip() or "callable"
        self.model_name = str(model_name)

    def score(self, manifest: pd.DataFrame, query: SelectorQuery) -> pd.DataFrame:
        output = self.selector(manifest.copy(), query)
        if not isinstance(output, pd.DataFrame):
            raise TypeError("Custom selector must return a pandas DataFrame.")
        return output


SelectorLike = str | SelectorBackend | Callable[[pd.DataFrame, SelectorQuery], pd.DataFrame]


class ClipSelectorBackend(SelectorBackend):
    """CLIP text-to-image scorer with validated, metadata-independent caching."""

    name = "clip"
    CACHE_VERSION = 1

    def __init__(
        self,
        *,
        model_name: str = "clip-ViT-B-32",
        device: str | None = None,
        cache_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        batch_size: int = 32,
        show_progress: bool = False,
        rebuild_index: bool = False,
        preprocess: str = "whole_image",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if preprocess != "whole_image":
            raise ValueError(
                "ClipSelectorBackend currently supports preprocess='whole_image'. "
                "The legacy CLIPFillerSelector retains preprocess='largest_face'."
            )
        self.model_name = str(model_name).strip()
        if not self.model_name:
            raise ValueError("model_name must not be empty.")
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.index_dir = Path(index_dir) if index_dir is not None else None
        self.batch_size = int(batch_size)
        self.show_progress = bool(show_progress)
        self.rebuild_index = bool(rebuild_index)
        self.preprocess = preprocess

    def score(self, manifest: pd.DataFrame, query: SelectorQuery) -> pd.DataFrame:
        prepared = _prepare_clip_manifest(manifest)
        if prepared.empty:
            return pd.DataFrame(
                columns=[
                    "image_id",
                    "selector_score",
                    "clip_score",
                    "positive_similarity",
                    "negative_similarity",
                ]
            )

        # Lazy imports keep pyWitnessAI importable without optional CLIP dependencies.
        from .cfd_clip_pilot.clip_backend import SentenceTransformerClipEncoder
        from .cfd_clip_pilot.index import ClipIndex

        encoder = SentenceTransformerClipEncoder(model_name=self.model_name, device=self.device)
        fingerprint = _embedding_fingerprint(
            prepared,
            model_name=self.model_name,
            preprocess=self.preprocess,
            cache_version=self.CACHE_VERSION,
        )
        index = self._load_or_build_index(
            prepared,
            encoder=encoder,
            index_class=ClipIndex,
            fingerprint=fingerprint,
        )

        positive_vectors = encoder.encode_texts(
            query.positive_texts,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )
        positive_similarity = (index.image_embeddings @ positive_vectors.T).mean(axis=1)

        negative_similarity = np.full(len(index.manifest), np.nan, dtype=np.float32)
        if query.negative_texts:
            negative_vectors = encoder.encode_texts(
                query.negative_texts,
                batch_size=self.batch_size,
                show_progress=self.show_progress,
            )
            negative_similarity = (index.image_embeddings @ negative_vectors.T).mean(axis=1)

        selector_score = positive_similarity
        if query.resolved_score_mode == "contrastive":
            selector_score = positive_similarity - negative_similarity

        return pd.DataFrame(
            {
                "image_id": index.manifest["image_id"].astype(str).to_numpy(),
                "selector_score": np.asarray(selector_score, dtype=np.float32),
                "clip_score": np.asarray(positive_similarity, dtype=np.float32),
                "positive_similarity": np.asarray(positive_similarity, dtype=np.float32),
                "negative_similarity": np.asarray(negative_similarity, dtype=np.float32),
            }
        )

    def _load_or_build_index(self, manifest, *, encoder, index_class, fingerprint: str):
        index_location = self._index_location(fingerprint)
        if index_location is not None and not self.rebuild_index:
            try:
                candidate = index_class.load(index_location)
                _validate_cached_index(
                    candidate,
                    manifest=manifest,
                    fingerprint=fingerprint,
                    model_name=self.model_name,
                )
                return candidate
            except (
                FileNotFoundError,
                ValueError,
                OSError,
                EOFError,
                KeyError,
                json.JSONDecodeError,
            ):
                pass

        index = index_class.build(
            manifest=manifest,
            encoder=encoder,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
        )
        index.metadata.update(
            {
                "dataset_fingerprint": fingerprint,
                "preprocess": self.preprocess,
                "cache_version": self.CACHE_VERSION,
            }
        )
        if index_location is not None:
            index.save(index_location)
        return index

    def _index_location(self, fingerprint: str) -> Path | None:
        if self.index_dir is not None:
            return self.index_dir
        if self.cache_dir is None:
            return None
        return self.cache_dir / _slug(self.model_name) / fingerprint


class MatchPolicy(ABC):
    """Convert a complete score ranking into a selected/not-selected mask."""

    @abstractmethod
    def apply(self, ranked: pd.DataFrame) -> pd.Series:
        pass


@dataclass(frozen=True)
class AllRankedPolicy(MatchPolicy):
    def apply(self, ranked: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=ranked.index, dtype=bool)


@dataclass(frozen=True)
class TopKPolicy(MatchPolicy):
    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1.")

    def apply(self, ranked: pd.DataFrame) -> pd.Series:
        return ranked["rank"].le(min(self.k, len(ranked)))


@dataclass(frozen=True)
class MinScorePolicy(MatchPolicy):
    threshold: float

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite.")

    def apply(self, ranked: pd.DataFrame) -> pd.Series:
        return ranked["selector_score"].ge(float(self.threshold))


@dataclass(frozen=True)
class TopKAndMinScorePolicy(MatchPolicy):
    k: int
    threshold: float

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1.")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("threshold must be finite.")

    def apply(self, ranked: pd.DataFrame) -> pd.Series:
        return ranked["rank"].le(min(self.k, len(ranked))) & ranked["selector_score"].ge(
            float(self.threshold)
        )


@dataclass(frozen=True)
class GroundTruthSpec:
    """Explicitly define how metadata determines whether a candidate is a hit."""

    column: str | None = None
    positive_values: tuple[object, ...] = (True, 1, "true", "yes", "positive", "hit")
    conditions: Mapping[str, object] = field(default_factory=dict)
    predicate: Callable[[Mapping[str, object]], bool | None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    target_column: str = "target_id"

    def __post_init__(self) -> None:
        modes = int(self.column is not None) + int(bool(self.conditions)) + int(self.predicate is not None)
        if modes != 1:
            raise ValueError("Provide exactly one of column, conditions, or predicate.")
        if self.column is not None and not str(self.column).strip():
            raise ValueError("column must not be empty.")

    def labels(self, manifest: pd.DataFrame) -> pd.Series:
        if self.column is not None:
            if self.column not in manifest.columns:
                raise ValueError(f"Ground-truth column is missing from metadata: {self.column}")
            values = [
                pd.NA if _missing(value) else _matches_expected(value, self.positive_values)
                for value in manifest[self.column]
            ]
            return pd.Series(pd.array(values, dtype="boolean"), index=manifest.index)

        if self.conditions:
            missing_columns = [column for column in self.conditions if column not in manifest.columns]
            if missing_columns:
                raise ValueError(f"Ground-truth columns are missing from metadata: {missing_columns}")
            labels = []
            for record in manifest.to_dict(orient="records"):
                if any(_missing(record.get(column)) for column in self.conditions):
                    labels.append(pd.NA)
                    continue
                labels.append(
                    all(
                        _matches_expected(record.get(column), expected)
                        for column, expected in self.conditions.items()
                    )
                )
            return pd.Series(pd.array(labels, dtype="boolean"), index=manifest.index)

        labels = []
        for record in manifest.to_dict(orient="records"):
            value = self.predicate(record) if self.predicate is not None else None
            labels.append(pd.NA if value is None or _missing(value) else bool(value))
        return pd.Series(pd.array(labels, dtype="boolean"), index=manifest.index)


@dataclass
class SelectionResult:
    query: SelectorQuery
    ranked: pd.DataFrame
    summary: dict[str, object]
    catalog_report: CatalogReport
    policy: MatchPolicy

    @property
    def matches(self) -> pd.DataFrame:
        return self.ranked.loc[self.ranked["selected"].fillna(False)].reset_index(drop=True).copy()

    def show(
        self,
        *,
        cols: int = 5,
        max_items: int | None = 50,
        display: bool = True,
    ):
        """Display selected faces in a grid and return the matplotlib figure."""
        if cols < 1:
            raise ValueError("cols must be >= 1.")
        if max_items is not None and max_items < 1:
            raise ValueError("max_items must be >= 1 or None.")

        matches = self.matches
        if matches.empty:
            print("No faces matched the selection policy.")
            return None
        shown = matches if max_items is None else matches.head(max_items)
        if len(shown) < len(matches):
            print(f"Showing {len(shown)} of {len(matches)} matched faces.")

        from .cfd_clip_pilot.visualization import show_clip_lineup

        id_column = "target_id" if "target_id" in shown.columns else "image_id"
        fig = show_clip_lineup(
            shown,
            top_k=len(shown),
            cols=cols,
            target_position="none",
            score_col="selector_score",
            id_col=id_column,
            suptitle=self.query.description,
        )
        if display:
            try:
                from IPython.display import display as ipy_display

                ipy_display(fig)
            except ImportError:
                import matplotlib.pyplot as plt

                plt.show()
        return fig

    def print_summary(self) -> None:
        for key, value in self.summary.items():
            print(f"{key}: {value}")

    def save_csv(self, path: str | Path, *, matches_only: bool = False) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        (self.matches if matches_only else self.ranked).to_csv(output, index=False)
        return output


class FaceSearch:
    """Search a manifest-backed image catalog with a pluggable selector backend."""

    def __init__(
        self,
        catalog: ImageCatalog | pd.DataFrame,
        *,
        selector: SelectorLike = "clip",
        selector_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        self.catalog = (
            catalog
            if isinstance(catalog, ImageCatalog)
            else ImageCatalog.from_manifest(catalog, validate_images=False)
        )
        self.selector = create_selector_backend(selector, **dict(selector_kwargs or {}))

    @classmethod
    def from_folder(
        cls,
        image_dir: str | Path,
        *,
        recursive: bool = True,
        metadata: MetadataSource | None = None,
        metadata_join: MetadataJoinSpec | None = None,
        validate_images: bool = True,
        selector: SelectorLike = "clip",
        selector_kwargs: Mapping[str, object] | None = None,
    ) -> "FaceSearch":
        catalog = ImageCatalog.from_folder(
            image_dir,
            recursive=recursive,
            metadata=metadata,
            metadata_join=metadata_join,
            validate_images=validate_images,
        )
        return cls(catalog, selector=selector, selector_kwargs=selector_kwargs)

    def score(self, description: str | SelectorQuery) -> pd.DataFrame:
        query = description if isinstance(description, SelectorQuery) else SelectorQuery.from_text(description)
        candidates = self.catalog.ready
        if candidates.empty:
            raise ValueError("No valid images are available to score.")
        raw = self.selector.score(candidates.copy(), query)
        return normalise_selector_scores(
            candidates,
            raw,
            backend_name=self.selector.name,
            model_name=self.selector.model_name,
            require_complete=True,
        )

    def select(
        self,
        description: str | SelectorQuery,
        *,
        policy: MatchPolicy | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        ground_truth: GroundTruthSpec | Mapping[str, object] | None = None,
    ) -> SelectionResult:
        query = description if isinstance(description, SelectorQuery) else SelectorQuery.from_text(description)
        match_policy = _resolve_policy(policy=policy, top_k=top_k, min_score=min_score)
        ranked = self.score(query)
        ranked["selected"] = match_policy.apply(ranked).astype(bool)

        truth_spec = (
            GroundTruthSpec(conditions=dict(ground_truth))
            if ground_truth is not None and not isinstance(ground_truth, GroundTruthSpec)
            else ground_truth
        )
        summary = _selection_summary(
            ranked,
            catalog_report=self.catalog.report,
            ground_truth=truth_spec,
        )
        return SelectionResult(
            query=query,
            ranked=ranked,
            summary=summary,
            catalog_report=self.catalog.report,
            policy=match_policy,
        )


def normalise_selector_scores(
    manifest: pd.DataFrame,
    raw_scores: pd.DataFrame,
    *,
    backend_name: str,
    model_name: str = "",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Validate backend output, restore current metadata, and rank stably."""
    if not isinstance(raw_scores, pd.DataFrame):
        raise TypeError("Selector backend must return a pandas DataFrame.")
    required_manifest = {"image_id", "image_path"}
    missing_manifest = sorted(required_manifest - set(manifest.columns))
    if missing_manifest:
        raise ValueError(f"Manifest is missing selector columns: {missing_manifest}")
    if manifest["image_id"].astype(str).duplicated().any():
        raise ValueError("Manifest image_id values must be unique.")

    raw = raw_scores.copy()
    if "selector_score" not in raw.columns:
        for alias in ("decision_score", "score", "clip_score"):
            if alias in raw.columns:
                raw = raw.rename(columns={alias: "selector_score"})
                break
    if "selector_score" not in raw.columns:
        raise ValueError(
            "Selector results must contain selector_score, decision_score, score, or clip_score."
        )

    if "image_id" not in raw.columns:
        if "image_path" not in raw.columns:
            raise ValueError("Selector results must contain image_id or image_path.")
        path_to_id = {
            _normalise_result_path(path): str(image_id)
            for path, image_id in zip(manifest["image_path"], manifest["image_id"])
        }
        raw["image_id"] = raw["image_path"].map(
            lambda value: path_to_id.get(_normalise_result_path(value))
        )
        if raw["image_id"].isna().any():
            unknown = raw.loc[raw["image_id"].isna(), "image_path"].astype(str).tolist()[:3]
            raise ValueError(f"Selector returned unknown image paths: {unknown}")
    raw["image_id"] = raw["image_id"].astype(str)

    if raw["image_id"].duplicated().any():
        duplicates = raw.loc[raw["image_id"].duplicated(), "image_id"].tolist()[:3]
        raise ValueError(f"Selector returned duplicate image_id values: {duplicates}")

    manifest_ids = set(manifest["image_id"].astype(str))
    raw_ids = set(raw["image_id"])
    unknown_ids = sorted(raw_ids - manifest_ids)
    if unknown_ids:
        raise ValueError(f"Selector returned unknown image_id values: {unknown_ids[:3]}")
    missing_ids = sorted(manifest_ids - raw_ids)
    if require_complete and missing_ids:
        raise ValueError(f"Selector did not score all candidates; missing image_id values: {missing_ids[:3]}")

    raw["selector_score"] = pd.to_numeric(raw["selector_score"], errors="coerce")
    if raw["selector_score"].isna().any() or not np.isfinite(raw["selector_score"].to_numpy()).all():
        raise ValueError("selector_score values must all be finite numbers.")

    reserved = {
        "rank",
        "backend_rank",
        "selector_score",
        "clip_score",
        "positive_similarity",
        "negative_similarity",
        "selector_backend",
        "selector_model",
        "selected",
        "ground_truth_match",
        "hit",
    }
    base = manifest.drop(columns=[column for column in reserved if column in manifest.columns]).copy()
    base["image_id"] = base["image_id"].astype(str)
    base["__catalog_order"] = range(len(base))

    backend_diagnostics = {
        "backend_rank",
        "clip_score",
        "positive_similarity",
        "negative_similarity",
        "selector_score",
    }
    extra_columns = [
        column for column in raw.columns if column == "image_id" or column not in base.columns
    ]
    extras = raw[extra_columns].copy()
    if "rank" in extras.columns:
        extras = extras.rename(columns={"rank": "backend_rank"})
    rename_backend_columns = {
        column: f"backend__{column}"
        for column in extras.columns
        if column != "image_id"
        and column not in backend_diagnostics
        and not column.startswith(("backend__", "selector__"))
    }
    extras = extras.rename(columns=rename_backend_columns)
    if extras.columns.duplicated().any():
        duplicates = extras.columns[extras.columns.duplicated()].tolist()
        raise ValueError(f"Selector returned conflicting diagnostic columns: {duplicates}")

    output = base.merge(extras, on="image_id", how="inner", validate="one_to_one", sort=False)
    output["selector_backend"] = str(backend_name)
    output["selector_model"] = str(model_name)
    sort_columns = ["selector_score"]
    ascending = [False]
    if "backend_rank" in output.columns:
        sort_columns.append("backend_rank")
        ascending.append(True)
    sort_columns.append("__catalog_order")
    ascending.append(True)
    output = output.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
    output["rank"] = range(1, len(output) + 1)
    return output.drop(columns=["__catalog_order"])


_SELECTOR_REGISTRY: dict[str, SelectorFactory] = {}


def register_selector_backend(
    name: str,
    factory: SelectorFactory,
    *,
    overwrite: bool = False,
) -> None:
    backend_name = _normalise_backend_name(name)
    if backend_name in _SELECTOR_REGISTRY and not overwrite:
        raise ValueError(f"Selector backend {backend_name!r} is already registered.")
    if not callable(factory):
        raise TypeError("factory must be callable.")
    _SELECTOR_REGISTRY[backend_name] = factory


def create_selector_backend(selector: SelectorLike = "clip", **kwargs) -> SelectorBackend:
    if isinstance(selector, SelectorBackend):
        if kwargs:
            raise ValueError("selector_kwargs cannot be used with an existing SelectorBackend instance.")
        return selector
    if isinstance(selector, str):
        name = _normalise_backend_name(selector)
        if name not in _SELECTOR_REGISTRY:
            known = ", ".join(available_selector_backends())
            raise ValueError(f"Unsupported selector backend {selector!r}. Registered backends: {known}.")
        backend = _SELECTOR_REGISTRY[name](**kwargs)
        if not isinstance(backend, SelectorBackend):
            raise TypeError(f"Selector factory {name!r} did not return a SelectorBackend.")
        return backend
    if callable(selector):
        name = str(kwargs.pop("name", "callable"))
        model_name = str(kwargs.pop("model_name", ""))
        if kwargs:
            raise ValueError(f"Unsupported callable selector options: {sorted(kwargs)}")
        return CallableSelectorBackend(selector, name=name, model_name=model_name)
    raise TypeError("selector must be a registered name, SelectorBackend instance, or callable.")


def available_selector_backends() -> tuple[str, ...]:
    return tuple(sorted(_SELECTOR_REGISTRY))


def _resolve_policy(
    *,
    policy: MatchPolicy | None,
    top_k: int | None,
    min_score: float | None,
) -> MatchPolicy:
    if policy is not None and (top_k is not None or min_score is not None):
        raise ValueError("Pass policy or top_k/min_score, not both.")
    if policy is not None:
        if not isinstance(policy, MatchPolicy):
            raise TypeError("policy must be a MatchPolicy.")
        return policy
    if top_k is not None and min_score is not None:
        return TopKAndMinScorePolicy(top_k, min_score)
    if top_k is not None:
        return TopKPolicy(top_k)
    if min_score is not None:
        return MinScorePolicy(min_score)
    raise ValueError(
        "Define what 'matched' means with policy=..., top_k=..., or min_score=.... "
        "Use AllRankedPolicy() explicitly to select every ranked image."
    )


def _selection_summary(
    ranked: pd.DataFrame,
    *,
    catalog_report: CatalogReport,
    ground_truth: GroundTruthSpec | None,
) -> dict[str, object]:
    selected = ranked["selected"].fillna(False).astype(bool)
    summary: dict[str, object] = {
        "candidate_count": catalog_report.image_count,
        "ready_count": catalog_report.ready_count,
        "skipped_count": catalog_report.invalid_count,
        "scored_count": len(ranked),
        "selected_count": int(selected.sum()),
        "metadata_provided": catalog_report.metadata_provided,
        "metadata_matched_image_count": catalog_report.metadata_matched_image_count,
        "metadata_missing_image_count": catalog_report.metadata_missing_image_count,
        "hit_image_count": None,
        "hit_target_count": None,
        "selected_labeled_count": None,
        "selected_unlabeled_count": None,
        "precision": None,
        "recall": None,
    }
    ranked["ground_truth_match"] = pd.Series(pd.array([pd.NA] * len(ranked), dtype="boolean"))
    ranked["hit"] = pd.Series(pd.array([pd.NA] * len(ranked), dtype="boolean"))
    if ground_truth is None:
        return summary

    labels = ground_truth.labels(ranked)
    ranked["ground_truth_match"] = labels
    labeled = labels.notna()
    positive = labels.fillna(False).astype(bool)
    hits = selected & positive
    ranked["hit"] = pd.Series(pd.array(hits.where(labeled, pd.NA), dtype="boolean"))

    selected_labeled = selected & labeled
    hit_count = int(hits.sum())
    positive_count = int(positive.sum())
    selected_labeled_count = int(selected_labeled.sum())
    target_column = ground_truth.target_column
    if target_column in ranked.columns:
        hit_targets = int(ranked.loc[hits, target_column].dropna().astype(str).nunique())
    else:
        hit_targets = hit_count

    summary.update(
        {
            "hit_image_count": hit_count,
            "hit_target_count": hit_targets,
            "selected_labeled_count": selected_labeled_count,
            "selected_unlabeled_count": int((selected & ~labeled).sum()),
            "precision": (
                hit_count / selected_labeled_count if selected_labeled_count else None
            ),
            "recall": hit_count / positive_count if positive_count else None,
        }
    )
    return summary


def _prepare_clip_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"CLIP manifest is missing required columns: {missing}")
    output = manifest[["image_id", "image_path"]].copy()
    output["image_id"] = output["image_id"].astype(str)
    if output["image_id"].duplicated().any():
        raise ValueError("CLIP manifest image_id values must be unique.")
    missing_paths = [path for path in output["image_path"].astype(str) if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"CLIP manifest contains missing image files: {missing_paths[:3]}")
    output["target_id"] = output["image_id"]
    if "target_id" in manifest.columns:
        target_ids = manifest["target_id"]
        present = target_ids.notna() & target_ids.astype(str).str.strip().ne("")
        output.loc[present.to_numpy(), "target_id"] = target_ids.loc[present].astype(str).to_numpy()
    return output[["image_id", "target_id", "image_path"]].reset_index(drop=True)


def _embedding_fingerprint(
    manifest: pd.DataFrame,
    *,
    model_name: str,
    preprocess: str,
    cache_version: int,
) -> str:
    images = []
    for record in manifest[["image_id", "image_path"]].to_dict(orient="records"):
        path = Path(str(record["image_path"])).resolve()
        stat = path.stat()
        images.append(
            {
                "image_id": str(record["image_id"]),
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "cache_version": cache_version,
        "model_name": model_name,
        "preprocess": preprocess,
        "images": sorted(images, key=lambda item: item["image_id"]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_cached_index(index, *, manifest: pd.DataFrame, fingerprint: str, model_name: str) -> None:
    if index.metadata.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Cached index fingerprint does not match the current images.")
    if index.metadata.get("model_name") != model_name:
        raise ValueError("Cached index model does not match the selector model.")
    if len(index.manifest) != len(index.image_embeddings):
        raise ValueError("Cached index embedding count does not match its manifest.")
    expected = set(manifest["image_id"].astype(str))
    actual = set(index.manifest["image_id"].astype(str))
    if expected != actual:
        raise ValueError("Cached index image IDs do not match the current manifest.")


def _matches_expected(value: object, expected: object) -> bool:
    choices = expected if _is_value_collection(expected) else (expected,)
    return any(_normalise_comparable(value) == _normalise_comparable(choice) for choice in choices)


def _is_value_collection(value: object) -> bool:
    return isinstance(value, (set, frozenset, list, tuple)) and not isinstance(value, (str, bytes))


def _normalise_comparable(value: object) -> object:
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _text_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    values = (value,) if isinstance(value, str) else tuple(value)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _normalise_result_path(value: object) -> str:
    return str(Path(str(value)).resolve()).casefold()


def _normalise_backend_name(value: str) -> str:
    name = str(value).strip().lower()
    if not name:
        raise ValueError("Selector backend name must not be empty.")
    return name


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "model"


register_selector_backend("clip", ClipSelectorBackend)
