"""Model-independent scoring strategies for description-based face retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TYPE_CHECKING

import numpy as np
import pandas as pd

from .Description import ParsedDescription

if TYPE_CHECKING:
    from .FaceSearch import SelectorBackend

__all__ = [
    "ScoringResult", "ScoringStrategy", "CosineScoring", "ContrastScoring",
    "create_scoring_strategy",
]


@dataclass(frozen=True)
class ScoringResult:
    scores: pd.DataFrame
    components: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, object] = field(default_factory=dict)


class ScoringStrategy(Protocol):
    """Return one finite selector_score per candidate, identified by image_id."""

    def score(
        self, manifest: pd.DataFrame, description: ParsedDescription, backend: SelectorBackend
    ) -> ScoringResult: ...


class CosineScoring:
    """Send the parser's text unchanged to the backend; do not rescale scores.

    With ClipSelectorBackend this is the original CLIP cosine similarity.
    Other backends retain their own native positive similarity score.
    """

    def score(self, manifest, description, backend) -> ScoringResult:
        from .FaceSearch import SelectorQuery

        scores = backend.score(manifest.copy(), SelectorQuery.from_text(description.model_text))
        return ScoringResult(scores, metadata={"scoring_strategy": "cosine"})


class ContrastScoring:
    """Average attribute margins, retaining unpaired attributes as positive scores.

    Parsing policy belongs to the description parser. This strategy never rejects
    an attribute because it is uncertain. Standardisation and equal family
    weighting are explicit experimental options, disabled by default.

    Builders implement ``build(ParsedDescription)`` and use its accepted
    attributes directly. Raw/text-only parsers supply no attributes, so their
    whole model text is scored as one positive-only component.
    """

    def __init__(
        self, *, builder=None, standardise_components: bool = False,
        aggregation: str = "dimension_mean",
    ) -> None:
        if aggregation not in {"dimension_mean", "family_mean"}:
            raise ValueError("aggregation must be 'dimension_mean' or 'family_mean'.")
        if builder is None:
            from .EyewitnessContrastPrompts import (
                EyewitnessContrastPromptBuilder, GENERIC_CONTRAST_SCHEMA,
            )
            builder = EyewitnessContrastPromptBuilder(schema=GENERIC_CONTRAST_SCHEMA, policy="lenient")
        self.builder = builder
        self.standardise_components = bool(standardise_components)
        self.aggregation = aggregation

    def score(self, manifest, description, backend) -> ScoringResult:
        from .FaceSearch import SelectorQuery, normalise_selector_scores
        from .ContrastiveClipScoring import _aggregate_dimension_scores, _standardise_scores

        if description.attributes and not description.accepted_attributes:
            raise ValueError("The parser accepted no description attributes to score.")
        plan = self.builder.build(description)
        metadata = {
            "scoring_strategy": "contrast",
            "aggregation": self.aggregation,
            "standardise_components": self.standardise_components,
            "contrast_plan": plan,
        }
        # Only text-only input can fall back to the whole description. Falling
        # back with structured attributes could reintroduce rejected evidence.
        if not plan.pairs:
            if description.attributes:
                raise ValueError("The contrast builder produced no components for accepted attributes.")
            output = CosineScoring().score(manifest, description, backend)
            metadata["fallback_reason"] = "no_components_use_description"
            return ScoringResult(output.scores, metadata=metadata)

        image_ids = manifest["image_id"].astype(str).tolist()
        records = []
        dimension_values: dict[tuple[str, str], list[tuple[np.ndarray, float]]] = {}
        for pair in plan.pairs:
            query = SelectorQuery(
                positive_texts=pair.positive_texts,
                negative_texts=pair.negative_texts,
                description=description.model_text,
            )
            raw = backend.score(manifest.copy(), query)
            ranked = normalise_selector_scores(
                manifest, raw, backend_name=backend.name, model_name=backend.model_name,
            ).set_index("image_id").loc[image_ids]
            values = ranked["selector_score"].to_numpy(dtype=float)
            weight = float(pair.weight)
            if not np.isfinite(weight) or weight <= 0:
                raise ValueError("Component weights must be finite and greater than zero.")
            key = (pair.family, pair.dimension)
            dimension_values.setdefault(key, []).append((values, weight))
            uncertain = bool(getattr(pair, "uncertain", False)) or any(
                getattr(item, "uncertain", False)
                for item in plan.evidence if item.evidence_id in pair.evidence_ids
            )
            for image_index, (image_id, score) in enumerate(zip(image_ids, values)):
                records.append({
                    "image_id": image_id, "image_index": image_index, "component_id": pair.pair_id,
                    "family": pair.family, "dimension": pair.dimension,
                    "component_method": "contrast" if pair.negative_texts else "positive_only",
                    "uncertain": uncertain, "weight": weight,
                    "raw_score": float(score),
                })
        dimensions = {
            key: np.average(np.stack([v for v, _ in parts]), axis=0, weights=[w for _, w in parts])
            for key, parts in dimension_values.items()
        }
        if self.standardise_components:
            dimensions = {key: _standardise_scores(values) for key, values in dimensions.items()}
        for record in records:
            # Each prompt's raw score remains visible; its dimension contributes
            # one (optionally standardised) value to the final aggregation.
            record["dimension_score"] = float(
                dimensions[(record["family"], record["dimension"])][record["image_index"]]
            )
        scores = pd.DataFrame({
            "image_id": image_ids,
            "selector_score": _aggregate_dimension_scores(
                dimensions, by_family=self.aggregation == "family_mean",
            ),
        })
        return ScoringResult(scores, pd.DataFrame.from_records(records), metadata)


def create_scoring_strategy(scoring: str | ScoringStrategy = "cosine") -> ScoringStrategy:
    if isinstance(scoring, str):
        factories = {"cosine": CosineScoring, "contrast": ContrastScoring}
        try:
            return factories[scoring.strip().lower()]()
        except KeyError as exc:
            raise ValueError("scoring must be 'cosine', 'contrast', or a ScoringStrategy object.") from exc
    if callable(getattr(scoring, "score", None)):
        return scoring
    raise TypeError("scoring must be a strategy name or an object with score().")
