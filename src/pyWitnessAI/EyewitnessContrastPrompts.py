from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal, TYPE_CHECKING

from .Description import DescriptionAttribute, LenientDescriptionParser, ParsedDescription
from ._description_rules import (
    AGE_PROMPT_TEMPLATE, PROMPT_TEMPLATE, GENERIC_CONTRAST_SCHEMA,
    Abstention, ContrastKind, ContrastPromptRule, ContrastPromptSchema,
    _HEAD_ALIASES, _LANDMARK_RE, _LOGICAL_NEGATION_RE, _UNCERTAINTY_RE,
    _age_contexts, _family_contexts, _find_alias, _midpoint_near_alias,
    _normalise, _scope_abstentions, _split_feature_conjunctions,
)

if TYPE_CHECKING:
    from .FaceSearch import SelectorQuery


PARSER_VERSION = "strict_v1.3.3"
LENIENT_PARSER_VERSION = "lenient_v1.0.0"
PROMPT_TEMPLATE_VERSION = "generic_face_v1"

PlanStatus = Literal["ready", "partial", "abstain"]

__all__ = [
    "AGE_PROMPT_TEMPLATE",
    "Abstention",
    "AttributeEvidence",
    "ContrastPlan",
    "ContrastPromptPair",
    "ContrastPromptRule",
    "ContrastPromptSchema",
    "EyewitnessContrastPromptBuilder",
    "GENERIC_CONTRAST_SCHEMA",
    "LENIENT_PARSER_VERSION",
    "PARSER_VERSION",
    "PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_VERSION",
]


@dataclass(frozen=True)
class AttributeEvidence:
    evidence_id: str
    feature: str
    dimension: str
    canonical_value: str
    polarity: str
    normalised_span: str
    rule_id: str
    uncertain: bool = False
    accepted: bool = True
    reason: str | None = None

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        output = asdict(self)
        if not include_text:
            output.pop("normalised_span", None)
        return output


@dataclass(frozen=True)
class ContrastPromptPair:
    pair_id: str
    family: str
    dimension: str
    positive_value: str
    negative_values: tuple[str, ...]
    positive_texts: tuple[str, ...]
    negative_texts: tuple[str, ...]
    contrast_kind: ContrastKind
    weight: float
    evidence_ids: tuple[str, ...]
    uncertain: bool = False
    source_text: bool = False

    def __post_init__(self) -> None:
        if not self.positive_texts:
            raise ValueError("A prompt component requires at least one positive prompt.")
        if not self.negative_texts:
            object.__setattr__(self, "contrast_kind", "positive_only")
        if self.positive_value in self.negative_values:
            raise ValueError("negative_values must not contain positive_value.")
        forbidden = re.compile(r"\b(?:no|not|without|never|neither)\b", re.IGNORECASE)
        if self.negative_texts and any(
            forbidden.search(text) for text in (*self.positive_texts, *self.negative_texts)
        ):
            raise ValueError("Generated contrast prompts must not use logical-negation words.")

    def to_selector_query(self) -> SelectorQuery:
        from .FaceSearch import SelectorQuery

        return SelectorQuery(
            positive_texts=self.positive_texts,
            negative_texts=self.negative_texts,
            score_mode="contrastive" if self.negative_texts else "positive",
            description=f"{self.family}.{self.dimension}={self.positive_value}",
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        output = asdict(self)
        if self.source_text and not include_text:
            output.pop("positive_texts", None)
        return output


@dataclass(frozen=True)
class ContrastPlan:
    status: PlanStatus
    pairs: tuple[ContrastPromptPair, ...]
    abstentions: tuple[Abstention, ...]
    evidence: tuple[AttributeEvidence, ...]
    provenance: Mapping[str, object]
    original_text: str = ""
    normalized_text: str = ""

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        output = {
            "status": self.status,
            "pairs": [pair.to_dict(include_text=include_text) for pair in self.pairs],
            "abstentions": [item.to_dict(include_text=include_text) for item in self.abstentions],
            "evidence": [item.to_dict(include_text=include_text) for item in self.evidence],
            "provenance": dict(self.provenance),
        }
        if include_text:
            output.update(original_text=self.original_text, normalized_text=self.normalized_text)
        return output

    def to_json(self, *, include_text: bool = False, **kwargs) -> str:
        options = {"sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(include_text=include_text), **options)


class EyewitnessContrastPromptBuilder:
    """Build attribute prompts without discarding the witness's description.

    A ``ParsedDescription`` supplies authoritative attributes and decisions;
    building prompts never parses its text again. With no attributes, its model
    text becomes one positive-only component. ``policy`` applies only to string
    input, which is a convenience for standalone use.

    The default ``lenient`` policy retains uncertain and unregistered evidence.
    Registered attributes use positive/reference prompts; descriptions without
    a defensible contrast use the original clause as a positive-only component.
    ``strict_v1`` reproduces the earlier conservative selection rules. Original
    text and evidence remain in memory; serialise with ``include_text=True`` to
    include witness text in the output.
    """

    def __init__(
        self,
        schema: ContrastPromptSchema = GENERIC_CONTRAST_SCHEMA,
        *,
        policy: str = "lenient",
    ) -> None:
        if policy not in {"lenient", "strict_v1"}:
            raise ValueError("policy must be 'lenient' or 'strict_v1'.")
        self.schema = schema
        self.policy = policy
        self._rules_by_family: dict[str, list[ContrastPromptRule]] = defaultdict(list)
        for rule in schema.rules:
            self._rules_by_family[rule.family].append(rule)

    def build(self, description: str | ParsedDescription) -> ContrastPlan:
        if isinstance(description, ParsedDescription):
            return self._build_parsed(description)
        if self.policy == "lenient":
            parsed = LenientDescriptionParser(schema=self.schema).parse(description)
            plan = self._build_parsed(parsed)
            return replace(plan, provenance={
                **plan.provenance,
                "parser_version": LENIENT_PARSER_VERSION,
                "policy": self.policy,
                "input_kind": "text",
            })
        text = _normalise(description)
        if not text:
            return self._plan((), (Abstention("empty_description"),), (), description)

        evidence: list[AttributeEvidence] = []
        abstentions: list[Abstention] = []
        if _LANDMARK_RE.search(text):
            abstentions.append(
                Abstention("landmark_reference", "hair", "length", normalised_span="hair landmark")
            )

        abstentions.extend(_scope_abstentions(text))
        hard_clauses = [
            segment.strip()
            for part in re.split(r"[.!?;:\n]+", text)
            for segment in re.split(r"\b(?:but|however)\b", part)
            if segment.strip()
        ]
        hard_clauses = [
            segment
            for clause in hard_clauses
            for segment in _split_feature_conjunctions(clause)
        ]
        for clause in hard_clauses:
            uncertain = bool(_UNCERTAINTY_RE.search(clause))
            negated = bool(_LOGICAL_NEGATION_RE.search(clause))
            for family, heads in _HEAD_ALIASES.items():
                # The generic ``hair`` head must not match the nested phrase
                # ``facial hair``; those words describe a different family.
                family_clause = (
                    re.sub(r"\bfacial hair\b", " ", clause)
                    if family == "hair"
                    else clause
                )
                if family == "age":
                    numeric_age = re.search(r"\b\d{1,2}\s*(?:years?\s*old|year-old)\b", family_clause)
                    if numeric_age:
                        abstentions.append(
                            Abstention(
                                "numeric_age_requires_preregistered_binning",
                                "age",
                                "band",
                                normalised_span=numeric_age.group(0),
                            )
                        )
                        contexts = []
                    else:
                        contexts = _age_contexts(family_clause)
                else:
                    contexts = _family_contexts(family_clause, heads)
                if not contexts:
                    continue
                if uncertain:
                    abstentions.append(
                        Abstention("epistemic_uncertainty", family, normalised_span=clause)
                    )
                    continue
                if family == "facial_hair":
                    special, special_abstentions = self._facial_hair_specials(
                        clause,
                        clause_negated=negated,
                    )
                    evidence.extend(special)
                    abstentions.extend(special_abstentions)
                for context in contexts:
                    matched = self._match_context(
                        family,
                        context,
                        clause,
                        abstentions,
                        clause_negated=negated,
                    )
                    # Ear/jaw references are unsafe evidence about hair length,
                    # but other explicit hair dimensions in the clause remain
                    # usable (for example, "dark curly hair down to his ears").
                    if family == "hair" and _LANDMARK_RE.search(clause):
                        matched = [item for item in matched if item.dimension != "length"]
                    evidence.extend(matched)

        evidence = _unique_evidence(evidence)
        by_dimension: dict[tuple[str, str], list[AttributeEvidence]] = defaultdict(list)
        for item in evidence:
            by_dimension[(item.feature, item.dimension)].append(item)

        accepted: list[AttributeEvidence] = []
        for (family, dimension), items in sorted(by_dimension.items()):
            values = sorted({item.canonical_value for item in items})
            if len(values) > 1:
                abstentions.append(
                    Abstention(
                        "conflicting_values",
                        family,
                        dimension,
                        normalised_span=" | ".join(values),
                    )
                )
            else:
                accepted.extend(items[:1])

        # A named hair hue is more specific than the vague dark/light tone.
        if any(item.feature == "hair" and item.dimension == "hue" for item in accepted):
            accepted = [
                item
                for item in accepted
                if not (item.feature == "hair" and item.dimension == "tone")
            ]

        contrasted = []
        for item in accepted:
            rule = self.schema.require(item.feature, item.dimension, item.canonical_value)
            if rule.alternatives:
                contrasted.append(item)
            else:
                abstentions.append(Abstention(
                    "missing_contrast", item.feature, item.dimension, item.rule_id,
                    item.normalised_span,
                ))
        pairs = tuple(self._pair_from_evidence(item) for item in sorted(
            contrasted, key=lambda item: (item.feature, item.dimension, item.canonical_value)
        ))
        return self._plan(
            pairs, tuple(_unique_abstentions(abstentions)), tuple(contrasted), description
        )

    def _build_parsed(self, parsed: ParsedDescription) -> ContrastPlan:
        # An empty annotation set is deliberate (e.g. a raw parser), not a
        # request to discover attributes using this builder's parsing rules.
        attributes = parsed.attributes
        if not attributes:
            if not parsed.model_text.strip():
                return self._plan((), (Abstention("empty_description"),), (), parsed)
            attributes = (DescriptionAttribute(text=parsed.model_text),)
        evidence = []
        pairs = []
        abstentions = []
        values_by_dimension: dict[tuple[str, str], set[str]] = defaultdict(set)
        for attribute in attributes:
            if attribute.accepted and attribute.dimension and attribute.value:
                values_by_dimension[(attribute.family, attribute.dimension)].add(attribute.value)
        fallback_pairs: dict[str, int] = {}
        for number, attribute in enumerate(attributes, start=1):
            try:
                rule = self.schema.require(attribute.family, attribute.dimension, attribute.value)
            except KeyError:
                rule = None
            item = AttributeEvidence(
                evidence_id=f"evidence:{number:04d}",
                feature=attribute.family or "description",
                dimension=attribute.dimension or "description",
                canonical_value=attribute.value or "unregistered",
                polarity=attribute.polarity,
                normalised_span=attribute.text,
                rule_id=rule.rule_id if rule is not None else "",
                uncertain=attribute.uncertain,
                accepted=attribute.accepted,
                reason=attribute.reason,
            )
            evidence.append(item)
            if not attribute.accepted:
                abstentions.append(Abstention(
                    attribute.reason or "rejected_by_parser", item.feature,
                    item.dimension, item.rule_id, item.normalised_span,
                ))
                continue
            conflict = len(values_by_dimension[(attribute.family, attribute.dimension)]) > 1
            if rule is not None and attribute.polarity in {"affirmed", "lexical_absence"} and not conflict:
                pairs.append(self._pair_from_evidence(item))
                continue
            # A negation, unresolved alternative, or unknown attribute is kept
            # verbatim. Inventing an opposite would change the reported evidence.
            if attribute.text in fallback_pairs:
                index = fallback_pairs[attribute.text]
                prior = pairs[index]
                pairs[index] = replace(
                    prior,
                    evidence_ids=(*prior.evidence_ids, item.evidence_id),
                    uncertain=prior.uncertain or item.uncertain,
                )
                continue
            fallback_pairs[attribute.text] = len(pairs)
            pairs.append(ContrastPromptPair(
                pair_id=f"positive_only:{number:04d}",
                family=item.feature,
                dimension=item.dimension,
                positive_value=item.canonical_value,
                negative_values=(),
                positive_texts=(attribute.text,),
                negative_texts=(),
                contrast_kind="positive_only",
                weight=1.0,
                evidence_ids=(item.evidence_id,),
                uncertain=item.uncertain,
                source_text=True,
            ))
        return self._plan(pairs, abstentions, evidence, parsed)

    def _match_context(
        self,
        family: str,
        context: str,
        clause: str,
        abstentions: list[Abstention],
        *,
        clause_negated: bool,
    ) -> list[AttributeEvidence]:
        matches: list[AttributeEvidence] = []
        if family == "facial_hair" and re.search(
            r"\bno facial hair\b|\bno (?:real |visible )?(?:beard|moustache|mustache)\b",
            context,
        ):
            return matches
        rules = self._rules_by_family.get(family, [])
        for rule in rules:
            alias = _find_alias(context, rule.aliases)
            if not alias:
                continue
            local_prefix = context[: context.find(alias)]
            if clause_negated or _LOGICAL_NEGATION_RE.search(local_prefix):
                abstentions.append(
                    Abstention("negated_scalar", family, rule.dimension, rule.rule_id, context)
                )
                continue
            if _midpoint_near_alias(context, alias) and rule.dimension in {
                "size", "length", "width", "height", "fullness", "shape"
            }:
                abstentions.append(
                    Abstention(
                        "ambiguous_midpoint_with_extreme",
                        family,
                        rule.dimension,
                        rule.rule_id,
                        context,
                    )
                )
                continue
            evidence_id = f"evidence:{rule.rule_id}"
            matches.append(
                AttributeEvidence(
                    evidence_id=evidence_id,
                    feature=family,
                    dimension=rule.dimension,
                    canonical_value=rule.value,
                    polarity="affirmed",
                    normalised_span=context,
                    rule_id=rule.rule_id,
                )
            )
        if not matches and re.search(r"\b(?:medium|average|normal|ordinary)\b", context):
            abstentions.append(
                Abstention("uninformative_midpoint", family, normalised_span=context)
            )
        return matches

    def _facial_hair_specials(
        self,
        clause: str,
        *,
        clause_negated: bool,
    ) -> tuple[list[AttributeEvidence], list[Abstention]]:
        output: list[AttributeEvidence] = []
        abstentions: list[Abstention] = []
        lexical_no_hair = re.search(r"\bno facial hair\b", clause)
        lexical_clean = re.search(r"\bclean[- ]shaven\b", clause)
        if lexical_no_hair or (lexical_clean and not clause_negated):
            rule = self.schema.require("facial_hair", "state", "clean_shaven")
            output.append(
                AttributeEvidence(
                    evidence_id=f"evidence:{rule.rule_id}",
                    feature="facial_hair",
                    dimension="state",
                    canonical_value="clean_shaven",
                    polarity="lexical_absence",
                    normalised_span=clause,
                    rule_id=rule.rule_id,
                )
            )
        if re.search(r"\bno (?:real |visible )?(?:beard|moustache|mustache)\b", clause):
            abstentions.append(
                Abstention("negated_specific_presence", "facial_hair", "state", normalised_span=clause)
            )
        return output, abstentions

    def _pair_from_evidence(self, item: AttributeEvidence) -> ContrastPromptPair:
        rule = self.schema.require(item.feature, item.dimension, item.canonical_value)
        positive = (rule.template.format(descriptor=rule.descriptor),)
        negative = tuple(rule.template.format(descriptor=value) for value in rule.alternatives)
        return ContrastPromptPair(
            pair_id=rule.rule_id,
            family=rule.family,
            dimension=rule.dimension,
            positive_value=rule.value,
            negative_values=rule.alternatives,
            positive_texts=positive,
            negative_texts=negative,
            contrast_kind=rule.contrast_kind,
            weight=rule.weight,
            evidence_ids=(item.evidence_id,),
            uncertain=item.uncertain,
        )

    def _plan(
        self,
        pairs: Sequence[ContrastPromptPair],
        abstentions: Sequence[Abstention],
        evidence: Sequence[AttributeEvidence],
        description: str | ParsedDescription = "",
    ) -> ContrastPlan:
        status: PlanStatus = "abstain" if not pairs else "partial" if abstentions else "ready"
        parsed = description if isinstance(description, ParsedDescription) else None
        return ContrastPlan(
            status=status,
            pairs=tuple(pairs),
            abstentions=tuple(abstentions),
            evidence=tuple(evidence),
            provenance={
                "schema_name": self.schema.name,
                "schema_version": self.schema.version,
                "schema_hash": self.schema.hash,
                "parser_version": (
                    None if parsed is not None else
                    PARSER_VERSION if self.policy == "strict_v1" else LENIENT_PARSER_VERSION
                ),
                "prompt_template_version": (
                    PROMPT_TEMPLATE_VERSION if self.schema is GENERIC_CONTRAST_SCHEMA else self.schema.version
                ),
                "prompt_template_sha256": hashlib.sha256(
                    "\n".join(dict.fromkeys(rule.template for rule in self.schema.rules)).encode()
                ).hexdigest(),
                "pool_profile": self.schema.pool_profile,
                "policy": "provided" if parsed is not None else self.policy,
                "input_kind": "parsed_description" if parsed is not None else "text",
                "full_raw_description_field_present": True,
                "in_memory_evidence_spans_present": True,
                "default_serialization_includes_spans": False,
            },
            original_text=(
                parsed.original_text if parsed is not None else
                "" if description is None else str(description)
            ),
            normalized_text=parsed.normalized_text if parsed is not None else _normalise(description),
        )


def _unique_evidence(values: Sequence[AttributeEvidence]) -> list[AttributeEvidence]:
    output = []
    seen = set()
    for item in values:
        key = (item.feature, item.dimension, item.canonical_value, item.polarity)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _unique_abstentions(values: Sequence[Abstention]) -> list[Abstention]:
    output = []
    seen = set()
    for item in values:
        key = (item.reason_code, item.feature, item.dimension, item.rule_id)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
