"""Frozen v1.1 output contract over the shared contrastive scoring engine.

For complete positive/reference pairs, the v1.2 engine has the same arithmetic
as v1.1. This adapter rejects the new positive-only input case and removes the
new output fields. Exact oracle tests cover rankings, components, prompt audit
and provenance, including raw-only and mixed-method runs.
"""

from __future__ import annotations

from pyWitnessAI.ContrastiveClipScoring import (
    ContrastiveClipScorer as _Scorer,
    ContrastiveScoreResult,
    DEFAULT_METHODS,
    SCORER_VERSION as ENGINE_VERSION,
)


SCORER_VERSION = "contrastive_v1.1.0"
DEFAULT_NEUTRAL_PROMPT = (
    "A standardized frontal neutral-expression headshot of a White adult man."
)
_VERIFIED_ENGINE_VERSION = "contrastive_v1.2.0"


class ContrastiveClipScorer(_Scorer):
    """Retain the preregistered contract only on its validated input domain."""

    def __init__(self, index, encoder, *, neutral_prompt=DEFAULT_NEUTRAL_PROMPT, **kwargs):
        if ENGINE_VERSION != _VERIFIED_ENGINE_VERSION:
            raise RuntimeError("The shared scorer changed; revalidate the frozen CFD contract before use.")
        super().__init__(index, encoder, neutral_prompt=neutral_prompt, **kwargs)

    def score(self, plans, *, raw_descriptions=None, methods=None):
        plans = list(plans)
        for plan in plans:
            for pair in plan.pairs:
                if not pair.positive_texts or not pair.negative_texts or getattr(pair, "source_text", False):
                    raise ValueError("The frozen v1.1 scorer requires complete positive/reference pairs.")
        result = super().score(plans, raw_descriptions=raw_descriptions, methods=methods)
        components = result.components.drop(columns="n_positive_only_pairs")
        audit = result.prompt_audit.copy()
        # v1.1 records private chunks under their raw role even if the same
        # literal text is also present as a generated attribute prompt.
        audit.loc[audit["is_raw_description"], "roles"] = "raw_description_chunk"
        return ContrastiveScoreResult(result.ranked, components, audit, result.provenance)

    def _provenance(self, methods):
        provenance = super()._provenance(methods)
        provenance["scorer_version"] = SCORER_VERSION
        provenance.pop("missing_reference_policy")
        provenance.pop("uncertainty_policy")
        return provenance


def score_contrast_plans(index, encoder, plans, **kwargs):
    score_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in {"raw_descriptions", "methods"}}
    return ContrastiveClipScorer(index, encoder, **kwargs).score(plans, **score_kwargs)
