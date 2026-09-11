"""CFD templates and the frozen strict-v1 plan contract.

Parsing and semantic rules are supplied by pyWitnessAI. This adapter fixes the
study's templates, conservative policy and audit serialization; it contains no
second parser. Oracle fixtures in tests cover the former standalone engine.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

from pyWitnessAI.EyewitnessContrastPrompts import (
    Abstention,
    ContrastPlan as _ContrastPlan,
    ContrastPromptPair as _ContrastPromptPair,
    ContrastPromptSchema,
    EyewitnessContrastPromptBuilder as _PromptBuilder,
    GENERIC_CONTRAST_SCHEMA,
    PARSER_VERSION,
)


PROMPT_TEMPLATE_VERSION = "cfd_wm_neutral_v1"
PROMPT_TEMPLATE = (
    "A standardized frontal neutral-expression headshot of a White adult man with {descriptor}."
)
AGE_PROMPT_TEMPLATE = (
    "A standardized frontal neutral-expression headshot of a White man who appears {descriptor}."
)
POOL_PROFILE = "CFD-3.0|White|Male|Neutral"
STRICT_V1_REGISTRY = ContrastPromptSchema(
    name="CFD White Male Neutral eyewitness contrast registry",
    version="1.0.0",
    pool_profile=POOL_PROFILE,
    rules=tuple(
        replace(rule, template=AGE_PROMPT_TEMPLATE if rule.family == "age" else PROMPT_TEMPLATE)
        for rule in GENERIC_CONTRAST_SCHEMA.rules
    ),
)
CFD_WHITE_MALE_NEUTRAL_CONTRAST_V1 = STRICT_V1_REGISTRY


class ContrastPromptPair(_ContrastPromptPair):
    """The study accepts only matched positive/reference components."""

    def __post_init__(self):
        if not self.negative_texts:
            raise ValueError("A contrast prompt pair requires positive and reference prompts.")
        super().__post_init__()


class ContrastPlan(_ContrastPlan):
    """Serialize the original plan fields, excluding newer parser annotations."""

    def to_dict(self, *, include_text=False):
        payload = super().to_dict(include_text=include_text)
        payload.pop("original_text", None)
        payload.pop("normalized_text", None)
        for pair in payload["pairs"]:
            pair.pop("uncertain", None)
            pair.pop("source_text", None)
        for item in payload["evidence"]:
            for field in ("uncertain", "accepted", "reason"):
                item.pop(field, None)
        return payload


class EyewitnessContrastPromptBuilder(_PromptBuilder):
    """Inject the CFD protocol into the shared strict-v1 parser."""

    def __init__(self, schema=STRICT_V1_REGISTRY, *, policy="strict_v1"):
        if policy != "strict_v1":
            raise ValueError("Only policy='strict_v1' is implemented by this frozen study.")
        if any(not rule.alternatives for rule in schema.rules):
            raise ValueError("The frozen study requires a reference for every rule.")
        super().__init__(schema, policy=policy)

    def build(self, description):
        if description is not None and not isinstance(description, str):
            raise TypeError("The frozen parser accepts description text, not pre-parsed attributes.")
        return super().build(description)

    def _plan(self, pairs, abstentions, evidence, description=""):
        abstentions = tuple(
            replace(item, reason_code="pool_invariant")
            if item.reason_code == "unsupported_v1" and item.feature in {"race", "gender", "expression"}
            else item
            for item in abstentions
        )
        return ContrastPlan(
            status="abstain" if not pairs else "partial" if abstentions else "ready",
            pairs=tuple(pairs),
            abstentions=abstentions,
            evidence=tuple(evidence),
            provenance={
                "schema_name": self.schema.name,
                "schema_version": self.schema.version,
                "schema_hash": self.schema.hash,
                "parser_version": PARSER_VERSION,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
                "pool_profile": self.schema.pool_profile,
                "policy": self.policy,
                "full_raw_description_field_present": False,
                "in_memory_evidence_spans_present": True,
                "default_serialization_includes_spans": False,
            },
        )
