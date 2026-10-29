"""Small, interchangeable preprocessing policies for witness descriptions.

Parsing and scoring are separate choices. The raw policy only normalizes text;
the rule policies additionally annotate English attribute mentions. The lenient
policy keeps every mention, including attributes absent from a contrast schema.
The strict policy excludes uncertain clauses from model input while retaining
their evidence. None of these policies invents an opposite for a negated trait.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, TYPE_CHECKING, runtime_checkable

from ._description_rules import (
    GENERIC_CONTRAST_SCHEMA, _HEAD_ALIASES, _LOGICAL_NEGATION_RE,
    _UNCERTAINTY_RE, _age_contexts, _contains_feature_head, _family_contexts,
    _find_alias, _safe_polarity_scope_split, _scope_abstentions,
)

if TYPE_CHECKING:
    from ._description_rules import ContrastPromptSchema
    from .FaceAttributeSchema import FaceAttributeSchema


__all__ = [
    "DescriptionAttribute",
    "DescriptionParser",
    "ParsedDescription",
    "RawDescriptionParser",
    "RuleBasedDescriptionParser",
    "LenientDescriptionParser",
    "StrictDescriptionParser",
    "SchemaDescriptionParser",
    "create_description_parser",
    "normalize_description",
]


@dataclass(frozen=True)
class DescriptionAttribute:
    """A mentioned attribute and its policy decision, independent of scoring.

    ``text`` retains its normalized clause, including uncertainty and negation.
    Empty ``dimension``/``value`` indicate a mention without a registered rule.
    ``accepted`` controls model input, not whether the evidence is retained.
    """

    text: str
    family: str = ""
    dimension: str = ""
    value: str = ""
    uncertain: bool = False
    accepted: bool = True
    reason: str | None = None
    polarity: str = "affirmed"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedDescription:
    """Authoritative parser output shared by prompt builders and scorers.

    Cosine scoring uses ``model_text``. Attribute scoring uses ``attributes``
    and their acceptance decisions without parsing text again. An empty tuple
    means text-only input, so scoring uses the whole ``model_text`` instead.
    """

    original_text: str
    normalized_text: str
    model_text: str
    attributes: tuple[DescriptionAttribute, ...] = ()

    @classmethod
    def from_attributes(
        cls,
        attributes: Mapping[str, str],
        *,
        original_text: str | None = None,
        lowercase: bool = True,
    ) -> ParsedDescription:
        """Use explicit schema values without parsing or guessing their meaning."""
        model_text = "; ".join(attributes.values())
        original_text = model_text if original_text is None else original_text
        return cls(
            original_text,
            normalize_description(original_text, lowercase=lowercase),
            model_text,
            tuple(
                DescriptionAttribute(text=value, family=name, dimension=name, value=value)
                for name, value in attributes.items()
            ),
        )

    @property
    def schema_attributes(self) -> dict[str, str]:
        """Unambiguous, accepted schema values suitable for recording or matching."""
        values: dict[str, set[str]] = {}
        for item in self.accepted_attributes:
            if (item.family and item.family == item.dimension and item.value
                    and not item.uncertain and item.polarity in {"affirmed", "lexical_absence"}):
                values.setdefault(item.family, set()).add(item.value)
        return {name: next(iter(options)) for name, options in values.items() if len(options) == 1}

    @property
    def accepted_attributes(self) -> tuple[DescriptionAttribute, ...]:
        return tuple(item for item in self.attributes if item.accepted)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class DescriptionParser(Protocol):
    """Implement ``parse`` to plug an experiment's own policy into a model."""

    def parse(self, text: str) -> ParsedDescription: ...


def normalize_description(text: str, *, lowercase: bool = True) -> str:
    """Collapse whitespace and optionally lowercase, without rewriting words."""
    if not isinstance(text, str):
        raise TypeError("Description must be a string.")
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized.lower() if lowercase else normalized


class RawDescriptionParser:
    """Pass the whole description through after whitespace/case normalization."""

    def __init__(self, *, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def parse(self, text: str) -> ParsedDescription:
        normalized = normalize_description(text, lowercase=self.lowercase)
        return ParsedDescription(text, normalized, normalized)


class SchemaDescriptionParser:
    """Return canonical face attributes using an explicit face schema.

    Only attributes recognised by the schema reach the model. The schema owns
    uncertainty and negation handling; unknown descriptions produce empty model
    text. The same parser instance can be used by generation and face search.
    """

    name = "schema"

    def __init__(self, schema: FaceAttributeSchema | None = None, *, lowercase: bool = True) -> None:
        if schema is None:
            from .FaceAttributeSchema import DEFAULT_FACE_ATTRIBUTE_SCHEMA
            schema = DEFAULT_FACE_ATTRIBUTE_SCHEMA
        self.schema = schema
        self.lowercase = lowercase

    def parse(self, text: str) -> ParsedDescription:
        normalized = normalize_description(text, lowercase=self.lowercase)
        values = self.schema.parse(normalized)
        return ParsedDescription.from_attributes(values, original_text=text, lowercase=self.lowercase)


class RuleBasedDescriptionParser(RawDescriptionParser):
    """Annotate English attributes with a lenient or strict uncertainty policy.

    ``schema`` optionally supplies alternative attribute aliases. It affects
    annotation, never whether an unsupported description survives preprocessing.
    Strict filtering is deliberately clause based: an uncertainty marker scopes
    over coordinated attributes until a sentence, ``but``/``however``, or a fresh
    explicit assertion. This is a documented rule, not a general language model.
    Set ``lowercase=False`` to preserve case in text delivered to the model.
    """

    def __init__(
        self,
        policy: str = "lenient",
        *,
        lowercase: bool = True,
        schema: ContrastPromptSchema | None = None,
    ) -> None:
        super().__init__(lowercase=lowercase)
        if policy not in {"lenient", "strict"}:
            raise ValueError("Parser policy must be 'lenient' or 'strict'.")
        self.policy = policy
        self.schema = schema

    def parse(self, text: str) -> ParsedDescription:
        normalized = normalize_description(text, lowercase=self.lowercase)

        schema = self.schema if self.schema is not None else GENERIC_CONTRAST_SCHEMA
        attributes: list[DescriptionAttribute] = []
        accepted_clauses: list[str] = []
        excluded = False
        # Preserve newline boundaries for rule scope even though model text
        # uses single spaces in the no-filtering case.
        clauses = _description_clauses(text, lowercase=self.lowercase)
        for clause in clauses:
            lookup = _rule_text(clause)
            uncertain = bool(_UNCERTAINTY_RE.search(lookup))
            accepted = self.policy != "strict" or not uncertain
            reason = "epistemic_uncertainty" if not accepted else None
            if accepted:
                accepted_clauses.append(clause)
            else:
                excluded = True
            # A negator inside "not sure"/"don't think" expresses uncertainty;
            # a separate "not blue" remains explicit negation.
            negated = bool(_LOGICAL_NEGATION_RE.search(_UNCERTAINTY_RE.sub("", lookup)))
            polarity = "negated" if negated else "affirmed"
            clause_attributes: list[DescriptionAttribute] = []
            contexts_by_family: dict[str, list[str]] = {}
            matched_words: set[str] = set()
            for rule in schema.rules:
                if rule.family not in contexts_by_family:
                    family_clause = (
                        re.sub(r"\bfacial hair\b", " ", lookup)
                        if rule.family == "hair" else lookup
                    )
                    heads = _HEAD_ALIASES.get(rule.family, (rule.family.replace("_", " "),))
                    contexts_by_family[rule.family] = (
                        _age_contexts(family_clause) if rule.family == "age"
                        else _family_contexts(family_clause, heads)
                    )
                contexts = contexts_by_family[rule.family]
                if not any(_find_alias(context, rule.aliases) for context in contexts):
                    continue
                matched_words.update(rule.aliases)
                matched_words.update(_HEAD_ALIASES.get(rule.family, (rule.family.replace("_", " "),)))
                item_polarity = polarity
                if rule.family == "facial_hair" and rule.value == "clean_shaven":
                    if re.search(r"\bno facial hair\b", lookup):
                        item_polarity = "lexical_absence"
                clause_attributes.append(DescriptionAttribute(
                    text=clause,
                    family=rule.family,
                    dimension=rule.dimension,
                    value=rule.value,
                    uncertain=uncertain,
                    accepted=accepted,
                    reason=reason,
                    polarity=item_polarity,
                ))

            known_families = {item.family for item in clause_attributes}
            for mention in _scope_abstentions(lookup):
                if mention.feature not in known_families:
                    clause_attributes.append(DescriptionAttribute(
                        text=clause,
                        family=mention.feature,
                        uncertain=uncertain,
                        accepted=accepted,
                        reason=reason or "unmapped_attribute",
                        polarity=polarity,
                    ))
                    known_families.add(mention.feature)
            # Preserve unregistered modifiers even alongside a registered
            # attribute ("large red nose"). A contrast strategy can then score
            # the original clause as an additional positive-only description.
            unmatched = _UNCERTAINTY_RE.sub(" ", lookup)
            for word in sorted(matched_words, key=len, reverse=True):
                unmatched = re.sub(rf"\b{re.escape(word)}\b", " ", unmatched)
            unmatched = re.sub(
                r"\b(?:a|an|the|he|she|they|his|her|their|it|its|i|we|you|my|"
                r"had|has|have|having|was|were|is|are|be|been|looked|looks|appeared|"
                r"and|or|with|of|to|that|there|no|not|without)\b", " ", unmatched,
            )
            unmapped = bool(re.search(r"\w", unmatched)) and all(
                item.dimension for item in clause_attributes
            )
            if not clause_attributes or unmapped:
                clause_attributes.append(DescriptionAttribute(
                    text=clause,
                    uncertain=uncertain,
                    accepted=accepted,
                    reason=reason or "unmapped_attribute",
                    polarity=polarity,
                ))
            attributes.extend(clause_attributes)

        model_text = "; ".join(accepted_clauses) if excluded else normalized
        return ParsedDescription(text, normalized, model_text, tuple(attributes))


class LenientDescriptionParser(RuleBasedDescriptionParser):
    """Keep all mentions and their uncertainty markers in the model input."""

    def __init__(self, *, lowercase: bool = True, schema: ContrastPromptSchema | None = None) -> None:
        super().__init__("lenient", lowercase=lowercase, schema=schema)


class StrictDescriptionParser(RuleBasedDescriptionParser):
    """Exclude uncertain clauses from model input while retaining their evidence."""

    def __init__(self, *, lowercase: bool = True, schema: ContrastPromptSchema | None = None) -> None:
        super().__init__("strict", lowercase=lowercase, schema=schema)


def create_description_parser(
    parser: str | DescriptionParser = "raw", **kwargs,
) -> DescriptionParser:
    """Resolve a built-in name or return an experiment's parser instance."""
    if not isinstance(parser, str):
        if not callable(getattr(parser, "parse", None)):
            raise TypeError("parser must be 'raw', 'lenient', 'strict', 'schema', or implement parse(text).")
        if kwargs:
            raise TypeError("Parser options must be configured on the supplied parser instance.")
        return parser
    parsers = {
        "raw": RawDescriptionParser,
        "lenient": LenientDescriptionParser,
        "strict": StrictDescriptionParser,
        "schema": SchemaDescriptionParser,
    }
    try:
        parser_class = parsers[parser]
    except KeyError:
        raise ValueError("Unknown parser; choose 'raw', 'lenient', 'strict', or 'schema'.") from None
    return parser_class(**kwargs)


def _description_clauses(text: str, *, lowercase: bool) -> list[str]:
    """Keep modifier lists together and respect the declared uncertainty scope."""


    def has_head(value: str) -> bool:
        return _contains_feature_head(_rule_text(value)) or bool(re.search(
            r"\b(?:freckles?|tattoos?|dimples?|birthmarks?|wrinkles?)\b", value, re.IGNORECASE,
        ))

    def split(clause: str) -> list[str]:
        for boundary in re.finditer(r",|\b(?:and|with)\b", clause, re.IGNORECASE):
            left = clause[:boundary.start()].strip(" ,")
            right = clause[boundary.end():].strip(" ,")
            if not (left and right and has_head(left) and has_head(right)):
                continue
            explicit_assertion = bool(re.match(
                r"(?:(?:he|she|they|the (?:man|woman|person))\s+"
                r"(?:had|has|was|were|is|are)|(?:\w+\s+){1,3}(?:is|are|was|were))\b",
                right, re.IGNORECASE,
            ))
            if _UNCERTAINTY_RE.search(_rule_text(left)) and not explicit_assertion:
                continue
            if _LOGICAL_NEGATION_RE.search(_rule_text(left)) and not _safe_polarity_scope_split(_rule_text(left), _rule_text(right)):
                continue
            return [*split(left), *split(right)]
        # A new hedge after an already stated feature starts a new scope:
        # "short hair, maybe blue eyes" or "short hair maybe blue eyes".
        for marker in _UNCERTAINTY_RE.finditer(_rule_text(clause)):
            left = clause[:marker.start()].strip(" ,")
            right = clause[marker.start():].strip(" ,")
            if left and has_head(left) and has_head(right) and not _UNCERTAINTY_RE.search(_rule_text(left)):
                return [*split(left), right]
        return [clause]

    output: list[str] = []
    for part in re.split(r"[.!?;:\r\n]+|\b(?:but|however)\b", text, flags=re.IGNORECASE):
        clause = normalize_description(part, lowercase=lowercase).strip(" ,")
        if clause:
            output.extend(split(clause))
    return output


def _rule_text(text: str) -> str:
    """Normalize only the rule lookup view; evidence and model text stay intact."""
    return text.lower().replace("\u2019", "'").replace("\u2018", "'")
