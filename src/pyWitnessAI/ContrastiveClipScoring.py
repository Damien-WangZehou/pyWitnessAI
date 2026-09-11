from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .EyewitnessContrastPrompts import ContrastPlan
    from .clip import ClipIndex


ContrastMethod = Literal[
    "raw_positive",
    "structured_positive",
    "positive_minus_neutral",
    "attribute_contrastive",
    "attribute_contrastive_family_mean",
]

DEFAULT_NEUTRAL_PROMPT = "A face photograph of a person."
SCORER_VERSION = "contrastive_v1.2.0"
DEFAULT_METHODS: tuple[ContrastMethod, ...] = (
    "structured_positive",
    "positive_minus_neutral",
    "attribute_contrastive",
    "attribute_contrastive_family_mean",
)
_ALL_METHODS: tuple[ContrastMethod, ...] = ("raw_positive", *DEFAULT_METHODS)

__all__ = [
    "ContrastMethod",
    "ContrastiveClipScorer",
    "ContrastiveScoreResult",
    "DEFAULT_METHODS",
    "DEFAULT_NEUTRAL_PROMPT",
    "SCORER_VERSION",
    "score_contrast_plans",
]


@dataclass(frozen=True)
class ContrastiveScoreResult:
    """Long-form rankings and auditable per-attribute score components."""

    ranked: pd.DataFrame
    components: pd.DataFrame
    prompt_audit: pd.DataFrame
    provenance: Mapping[str, object]


class ContrastiveClipScorer:
    """Batch CLIP scorer for pre-parsed eyewitness contrast plans.

    The scorer reuses one :class:`ClipIndex`, encodes each unique prompt once,
    and gives each parsed attribute dimension equal weight.  Component scores
    are z-standardised within the fixed candidate pool by default so dimensions
    with different CLIP score ranges cannot dominate merely because of scale.
    A component without reference prompts contributes its positive similarity.
    The presence of uncertainty markers never forces its weight to zero.

    Raw descriptions are used only in memory and are never returned in the
    score result.  ``prompt_audit`` collapses all raw chunks to one count-only
    row and stores neither their text, identifier, ordering, nor text hash.
    """

    def __init__(
        self,
        index: "ClipIndex",
        encoder,
        *,
        neutral_prompt: str = DEFAULT_NEUTRAL_PROMPT,
        batch_size: int = 64,
        show_progress: bool = False,
        standardise_components: bool = True,
    ) -> None:
        if len(index.manifest) < 2:
            raise ValueError("Contrastive scoring requires at least two candidates.")
        image_embeddings = np.asarray(index.image_embeddings, dtype=np.float32)
        if not np.isfinite(image_embeddings).all():
            raise ValueError("Image-index embeddings must all be finite.")
        image_norms = np.linalg.norm(image_embeddings, axis=1)
        if (image_norms <= 1e-12).any() or not np.isfinite(image_norms).all():
            raise ValueError("Image-index embeddings must have non-zero finite norms.")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        neutral = str(neutral_prompt).strip()
        if not neutral:
            raise ValueError("neutral_prompt must not be empty.")
        index_model = str(index.metadata.get("model_name", "")).strip()
        encoder_model = str(getattr(encoder, "model_name", "")).strip()
        if index_model and encoder_model and index_model != encoder_model:
            raise ValueError(
                "Encoder model does not match the image index model: "
                f"{encoder_model!r} != {index_model!r}."
            )
        index_revision = str(index.metadata.get("model_revision", "") or "").strip()
        encoder_revision = str(getattr(encoder, "model_revision", "") or "").strip()
        if index_revision and encoder_revision and index_revision != encoder_revision:
            raise ValueError(
                "Encoder revision does not match the image index revision: "
                f"{encoder_revision!r} != {index_revision!r}."
            )
        self.index = index
        self.encoder = encoder
        self.neutral_prompt = neutral
        self.batch_size = int(batch_size)
        self.show_progress = bool(show_progress)
        self.standardise_components = bool(standardise_components)

    def score(
        self,
        plans: Sequence["ContrastPlan"],
        *,
        raw_descriptions: Sequence[str] | None = None,
        methods: Sequence[ContrastMethod] | None = None,
    ) -> ContrastiveScoreResult:
        plans = list(plans)
        raw_texts = _validate_raw_descriptions(raw_descriptions, len(plans))
        selected_methods = _validate_methods(
            _ALL_METHODS if methods is None and raw_texts is not None else DEFAULT_METHODS
            if methods is None
            else methods
        )
        if not plans:
            return ContrastiveScoreResult(
                ranked=_empty_ranked(),
                components=_empty_components(),
                prompt_audit=_empty_prompt_audit(),
                provenance=self._provenance(selected_methods),
            )
        if "raw_positive" in selected_methods and raw_texts is None:
            raise ValueError("raw_descriptions are required for method='raw_positive'.")

        prompt_roles: dict[str, set[str]] = defaultdict(set)
        private_texts: set[str] = set()
        raw_chunks: dict[str, tuple[tuple[str, ...], tuple[float, ...]]] = {}
        needs_positive = any(method != "raw_positive" for method in selected_methods)
        needs_negative = any(
            method in {"attribute_contrastive", "attribute_contrastive_family_mean"}
            for method in selected_methods
        )
        has_pairs = any(plan.pairs for plan in plans)
        if needs_positive:
            for plan in plans:
                for pair in plan.pairs:
                    for text in pair.positive_texts:
                        prompt_roles[text].add("attribute_positive")
                        if getattr(pair, "source_text", False):
                            private_texts.add(text)
                    if needs_negative:
                        for text in pair.negative_texts:
                            prompt_roles[text].add("attribute_reference")
        if has_pairs and any(method == "positive_minus_neutral" for method in selected_methods):
            prompt_roles[self.neutral_prompt].add("neutral_reference")
        if raw_texts is not None and "raw_positive" in selected_methods:
            for text in raw_texts:
                chunks, weights = self._prepare_raw_chunks(text)
                raw_chunks[text] = (chunks, weights)
                for chunk in chunks:
                    prompt_roles[chunk].add("raw_description_chunk")
                    private_texts.add(chunk)

        if not prompt_roles:
            return ContrastiveScoreResult(
                ranked=_empty_ranked(),
                components=_empty_components(),
                prompt_audit=_empty_prompt_audit(),
                provenance=self._provenance(selected_methods),
            )

        ordered_texts = sorted(prompt_roles, key=lambda text: (_sha256(text), text))
        encoded = np.asarray(
            self.encoder.encode_texts(
                ordered_texts,
                batch_size=self.batch_size,
                show_progress=self.show_progress,
            ),
            dtype=np.float32,
        )
        if encoded.ndim != 2 or encoded.shape[0] != len(ordered_texts):
            raise ValueError("Encoder returned an invalid number or shape of text embeddings.")
        if encoded.shape[1] != self.index.image_embeddings.shape[1]:
            raise ValueError(
                "Text embedding dimension does not match the image index: "
                f"{encoded.shape[1]} != {self.index.image_embeddings.shape[1]}."
            )
        encoded = _l2_normalise_rows(encoded)
        text_vectors = dict(zip(ordered_texts, encoded, strict=True))
        raw_vectors = {
            raw_text: _normalise_vector(
                np.average(
                    np.vstack([text_vectors[chunk] for chunk in chunks]),
                    axis=0,
                    weights=weights,
                )
            )
            for raw_text, (chunks, weights) in raw_chunks.items()
        }
        prompt_audit = _prompt_audit(prompt_roles, private_texts)

        ranked_parts: list[pd.DataFrame] = []
        component_records: list[dict[str, object]] = []
        neutral_similarity = None
        if self.neutral_prompt in text_vectors:
            neutral_similarity = self.index.image_embeddings @ text_vectors[self.neutral_prompt]

        for query_index, plan in enumerate(plans):
            dimension_arrays: dict[str, dict[str, object]] = {}
            grouped: dict[tuple[str, str], list] = defaultdict(list)
            if needs_positive:
                for pair in plan.pairs:
                    grouped[(str(pair.family), str(pair.dimension))].append(pair)

            for (family, dimension), pairs in sorted(grouped.items()):
                positive_parts = []
                negative_parts = []
                weights = []
                pair_ids = []
                n_references = 0
                for pair in pairs:
                    pos = _mean_similarity(self.index.image_embeddings, text_vectors, pair.positive_texts)
                    neg = (
                        _mean_similarity(
                            self.index.image_embeddings,
                            text_vectors,
                            pair.negative_texts,
                        )
                        if needs_negative and pair.negative_texts
                        else None
                    )
                    weight = float(getattr(pair, "weight", 1.0))
                    if not np.isfinite(weight) or weight <= 0:
                        raise ValueError(f"Pair {pair.pair_id!r} has a non-positive or invalid weight.")
                    positive_parts.append(pos)
                    if neg is not None:
                        n_references += 1
                    # Missing references contribute no subtraction, while the
                    # positive score retains the pair's full configured weight.
                    negative_parts.append(neg if neg is not None else np.zeros_like(pos))
                    weights.append(weight)
                    pair_ids.append(str(pair.pair_id))
                positive = np.average(np.vstack(positive_parts), axis=0, weights=weights)
                negative = (
                    np.average(np.vstack(negative_parts), axis=0, weights=weights)
                    if n_references
                    else None
                )
                contrast = (
                    positive - negative if negative is not None else positive
                ) if needs_negative else None
                positive_minus_neutral = (
                    positive - neutral_similarity if neutral_similarity is not None else None
                )
                dimension_key = f"{family}.{dimension}"
                dimension_arrays[dimension_key] = {
                    "family": family,
                    "dimension": dimension,
                    "pair_ids": "|".join(pair_ids),
                    "positive": positive,
                    "negative": negative,
                    "contrast": contrast,
                    "positive_minus_neutral": positive_minus_neutral,
                }
                contrast_standard = (
                    self._component_scale(contrast)
                    if contrast is not None
                    else np.full(len(positive), np.nan)
                )
                positive_standard = self._component_scale(positive)
                neutral_standard = (
                    self._component_scale(positive_minus_neutral)
                    if positive_minus_neutral is not None
                    else np.full(len(positive), np.nan)
                )
                for image_index, image_id in enumerate(
                    self.index.manifest["image_id"].astype(str).tolist()
                ):
                    component_records.append(
                        {
                            "query_index": query_index,
                            "plan_status": str(plan.status),
                            "dimension_key": dimension_key,
                            "family": family,
                            "dimension": dimension,
                            "pair_ids": "|".join(pair_ids),
                            "n_positive_only_pairs": sum(not pair.negative_texts for pair in pairs),
                            "image_index": image_index,
                            "image_id": image_id,
                            "target_id": str(self.index.manifest.iloc[image_index]["target_id"]),
                            "positive_similarity": float(positive[image_index]),
                            "negative_similarity": (
                                float(negative[image_index]) if negative is not None else np.nan
                            ),
                            "contrast_score": (
                                float(contrast[image_index]) if contrast is not None else np.nan
                            ),
                            "positive_component": float(positive_standard[image_index]),
                            "positive_minus_neutral_component": float(neutral_standard[image_index]),
                            "contrast_component": float(contrast_standard[image_index]),
                        }
                    )

            for method in selected_methods:
                score = self._method_score(
                    method,
                    dimension_arrays,
                    raw_text=raw_texts[query_index] if raw_texts is not None else None,
                    text_vectors=text_vectors,
                    raw_vectors=raw_vectors,
                )
                if score is None:
                    continue
                ranked_parts.append(
                    _rank_scores(
                        self.index.manifest,
                        score,
                        query_index=query_index,
                        method=method,
                        plan_status=str(plan.status),
                        n_dimensions=len(dimension_arrays),
                    )
                )

        ranked = pd.concat(ranked_parts, ignore_index=True) if ranked_parts else _empty_ranked()
        components = (
            pd.DataFrame.from_records(component_records)
            if component_records
            else _empty_components()
        )
        return ContrastiveScoreResult(
            ranked=ranked,
            components=components,
            prompt_audit=prompt_audit,
            provenance=self._provenance(selected_methods),
        )

    def _method_score(
        self,
        method: ContrastMethod,
        dimensions: dict[str, dict[str, object]],
        *,
        raw_text: str | None,
        text_vectors: dict[str, np.ndarray],
        raw_vectors: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        if method == "raw_positive":
            if raw_text is None:
                return None
            raw = self.index.image_embeddings @ raw_vectors[raw_text]
            return self._component_scale(raw)
        if not dimensions:
            return None

        if method == "structured_positive":
            return self._aggregate_dimensions(dimensions, "positive")
        if method == "positive_minus_neutral":
            return self._aggregate_dimensions(dimensions, "positive_minus_neutral")
        if method == "attribute_contrastive":
            return self._aggregate_dimensions(dimensions, "contrast")
        if method == "attribute_contrastive_family_mean":
            values = {
                (str(item["family"]), str(item["dimension"])):
                self._component_scale(np.asarray(item["contrast"], dtype=np.float32))
                for item in dimensions.values()
            }
            return _aggregate_dimension_scores(values, by_family=True)
        raise AssertionError(f"Unhandled method: {method}")

    def _aggregate_dimensions(
        self,
        dimensions: dict[str, dict[str, object]],
        key: str,
    ) -> np.ndarray:
        arrays = {}
        for values in dimensions.values():
            value = values[key]
            if value is None:
                raise ValueError(f"Missing score component {key!r}.")
            arrays[(str(values["family"]), str(values["dimension"]))] = self._component_scale(
                np.asarray(value, dtype=np.float32)
            )
        return _aggregate_dimension_scores(arrays)

    def _component_scale(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return _zscore(array) if self.standardise_components else array

    def _prepare_raw_chunks(self, text: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
        prepare = getattr(self.encoder, "prepare_text_chunks", None)
        if not callable(prepare):
            return (text,), (1.0,)
        chunks, token_counts, _, _ = prepare(text)
        if not chunks or len(chunks) != len(token_counts):
            raise ValueError("Encoder returned an invalid raw-description chunk plan.")
        weights = tuple(float(max(1, count)) for count in token_counts)
        return tuple(str(chunk) for chunk in chunks), weights

    def _provenance(self, methods: Sequence[ContrastMethod]) -> dict[str, object]:
        return {
            "scorer_version": SCORER_VERSION,
            "model_name": str(
                getattr(self.encoder, "model_name", "")
                or self.index.metadata.get("model_name", "")
            ),
            "model_revision": getattr(self.encoder, "model_revision", None),
            "model_load_source": getattr(self.encoder, "loaded_model_source", None),
            "source_index_embedding_sha256": self.index.metadata.get("embedding_sha256"),
            "candidate_embedding_sha256": _array_sha256(self.index.image_embeddings),
            "index_manifest_sha256": _manifest_sha256(self.index.manifest),
            "image_encoder_revision": self.index.metadata.get("model_revision"),
            "image_text_revision_match_verified": bool(
                self.index.metadata.get("model_revision")
                and getattr(self.encoder, "model_revision", None)
                and self.index.metadata.get("model_revision")
                == getattr(self.encoder, "model_revision", None)
            ),
            "candidate_count": int(len(self.index.manifest)),
            "neutral_prompt": self.neutral_prompt,
            "standardise_components": self.standardise_components,
            "raw_chunk_policy": "token-weighted mean, then L2 normalise",
            "missing_reference_policy": "use_positive_similarity",
            "uncertainty_policy": "retain_configured_weight",
            "methods": list(methods),
        }


def score_contrast_plans(
    index: ClipIndex,
    encoder,
    plans: Sequence["ContrastPlan"],
    **kwargs,
) -> ContrastiveScoreResult:
    """Convenience wrapper around :class:`ContrastiveClipScorer`."""
    score_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key in {"raw_descriptions", "methods"}
    }
    return ContrastiveClipScorer(index, encoder, **kwargs).score(plans, **score_kwargs)


def _mean_similarity(
    image_embeddings: np.ndarray,
    text_vectors: dict[str, np.ndarray],
    texts: Sequence[str],
) -> np.ndarray:
    vectors = np.vstack([text_vectors[str(text)] for text in texts])
    return (image_embeddings @ vectors.T).mean(axis=1)


def _zscore(values: np.ndarray) -> np.ndarray:
    return _standardise_scores(np.asarray(values, dtype=np.float32))


def _standardise_scores(values: np.ndarray) -> np.ndarray:
    """Standardise one dimension while preserving its computation precision."""
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Score components must all be finite before standardisation.")
    std = float(values.std(ddof=0))
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _aggregate_dimension_scores(dimensions, *, by_family: bool = False) -> np.ndarray:
    """Give dimensions equal weight, optionally averaging within families first."""
    values = list(dimensions.values())
    if by_family:
        families: dict[str, list[np.ndarray]] = defaultdict(list)
        for (family, _), scores in dimensions.items():
            families[family].append(scores)
        values = [np.vstack(scores).mean(axis=0) for scores in families.values()]
    return np.vstack(values).mean(axis=0)


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalise an empty or non-finite text vector.")
    return vector / norm


def _l2_normalise_rows(values: np.ndarray) -> np.ndarray:
    """Return finite, non-zero vectors with unit L2 norm row by row."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Text embeddings must be a two-dimensional matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Text embeddings must all be finite.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 1e-12).any() or not np.isfinite(norms).all():
        raise ValueError("Text embeddings must have a non-zero finite norm.")
    return (matrix / norms).astype(np.float32)


def _manifest_sha256(manifest: pd.DataFrame) -> str:
    """Fingerprint the ordered embedding-to-candidate mapping.

    Machine-specific path columns are deliberately excluded.  Row order is
    retained because row ``i`` must continue to identify embedding ``i``.
    """
    columns = sorted(
        column for column in manifest.columns if "path" not in str(column).lower()
    )
    canonical = pd.DataFrame(
        {
            column: manifest[column].astype(object).where(manifest[column].notna(), "<NA>").astype(str)
            for column in columns
        }
    )
    payload = canonical.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _rank_scores(
    manifest: pd.DataFrame,
    scores: np.ndarray,
    *,
    query_index: int,
    method: str,
    plan_status: str,
    n_dimensions: int,
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(manifest),):
        raise ValueError("Candidate score vector has the wrong shape.")
    if not np.isfinite(values).all():
        raise ValueError("Candidate scores must all be finite.")
    output = manifest.reset_index(drop=True).copy()
    output.insert(0, "query_index", int(query_index))
    output.insert(1, "method", method)
    output["plan_status"] = plan_status
    output["n_dimensions"] = int(n_dimensions)
    output["selector_score"] = values
    output = output.sort_values(
        ["selector_score", "image_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["rank"] = np.arange(1, len(output) + 1, dtype=int)
    return output


def _validate_methods(methods: Sequence[ContrastMethod]) -> tuple[ContrastMethod, ...]:
    allowed = set(_ALL_METHODS)
    output = tuple(dict.fromkeys(methods))
    invalid = [method for method in output if method not in allowed]
    if invalid:
        raise ValueError(f"Unknown contrastive scoring methods: {invalid}")
    if not output:
        raise ValueError("At least one scoring method is required.")
    return output


def _validate_raw_descriptions(
    values: Sequence[str] | None,
    expected: int,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if len(values) != expected:
        raise ValueError("raw_descriptions length must match plans length.")
    output = tuple(str(value).strip() for value in values)
    if any(not value for value in output):
        raise ValueError("raw_descriptions must not contain empty text.")
    return output


def _prompt_audit(
    roles: dict[str, set[str]],
    private_texts: set[str],
) -> pd.DataFrame:
    records = []
    public_items = [item for item in roles.items() if item[0] not in private_texts]
    for prompt_number, (text, role_values) in enumerate(
        sorted(public_items, key=lambda item: _sha256(item[0])),
        start=1,
    ):
        records.append(
            {
                "prompt_id": f"generated_prompt_{prompt_number:04d}",
                "prompt_sha256": _sha256(text),
                "roles": "|".join(sorted(role_values)),
                "is_raw_description": False,
                "n_prompts": 1,
                "prompt_text": text,
            }
        )
    if private_texts:
        records.append(
            {
                "prompt_id": pd.NA,
                "prompt_sha256": pd.NA,
                "roles": "|".join(sorted({role for text in private_texts for role in roles[text]})),
                "is_raw_description": True,
                "n_prompts": len(private_texts),
                "prompt_text": pd.NA,
            }
        )
    return pd.DataFrame.from_records(records, columns=_empty_prompt_audit().columns)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _empty_ranked() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query_index",
            "method",
            "image_id",
            "target_id",
            "image_path",
            "plan_status",
            "n_dimensions",
            "selector_score",
            "rank",
        ]
    )


def _empty_components() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query_index",
            "plan_status",
            "dimension_key",
            "family",
            "dimension",
            "pair_ids",
            "n_positive_only_pairs",
            "image_index",
            "image_id",
            "target_id",
            "positive_similarity",
            "negative_similarity",
            "contrast_score",
            "positive_component",
            "positive_minus_neutral_component",
            "contrast_component",
        ]
    )


def _empty_prompt_audit() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "prompt_id",
            "prompt_sha256",
            "roles",
            "is_raw_description",
            "n_prompts",
            "prompt_text",
        ]
    )
