from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Sequence

AttributeParser = Callable[[str], str | None]
PromptRole = Literal["subject", "detail"]

_PORTRAIT_INSTRUCTIONS = (
    "Make it an image portrait, only from shoulders above. "
    "Make the background gray. "
    "The people should be looking at the camera with frontal face. "
    "Do not emphasize or depict body actions, hand placement, or clothing interactions."
)

__all__ = [
    "DEFAULT_FACE_ATTRIBUTE_SCHEMA",
    "FaceAttributeDefinition",
    "FaceAttributeSchema",
    "FaceDescriptionSchema",
    "LEGACY_FACE_ATTRIBUTE_SCHEMA_V1",
]


@dataclass(frozen=True)
class FaceDescriptionSchema:
    """Structured facial-description fields used to build generation prompts."""

    original_description: str
    gender: str | None = None
    race: str | None = None
    age: str | None = None
    hair: str | None = None
    facial_hair: str | None = None
    eyes: str | None = None
    eyebrows: str | None = None
    nose: str | None = None
    build: str | None = None
    face_shape: str | None = None
    forehead: str | None = None
    mouth: str | None = None
    ears: str | None = None
    jaw: str | None = None
    teeth: str | None = None
    expression: str | None = None
    clothing: str | None = None
    accessories: str | None = None
    other_details: tuple[str, ...] = field(default_factory=tuple)
    hair_color: str | None = None
    hair_texture: str | None = None
    eyebrow_color: str | None = None
    custom_attributes: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @classmethod
    def from_attributes(
        cls,
        original_description: str,
        attributes: Mapping[str, str],
        *,
        other_details: tuple[str, ...] = (),
    ) -> "FaceDescriptionSchema":
        field_names = {item.name for item in fields(cls)}
        known = {
            key: value
            for key, value in attributes.items()
            if key in field_names and key not in {"original_description", "custom_attributes", "other_details"}
        }
        custom = tuple(
            sorted(
                (str(key), str(value))
                for key, value in attributes.items()
                if value and key not in known
            )
        )
        return cls(
            original_description=original_description,
            custom_attributes=custom,
            other_details=tuple(other_details),
            **known,
        )

    def attribute_values(self) -> dict[str, str]:
        excluded = {"original_description", "custom_attributes", "other_details"}
        values = {
            item.name: str(getattr(self, item.name))
            for item in fields(self)
            if item.name not in excluded and getattr(self, item.name)
        }
        values.update({str(key): str(value) for key, value in self.custom_attributes if value})
        return values

    def with_attributes(self, attributes: Mapping[str, str]) -> "FaceDescriptionSchema":
        values = self.attribute_values()
        values.update({str(key): str(value) for key, value in attributes.items() if value})
        return self.from_attributes(
            self.original_description,
            values,
            other_details=self.other_details,
        )

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "original_description": self.original_description,
            **self.attribute_values(),
        }
        if self.other_details:
            values["other_details"] = list(self.other_details)
        return values

    def subject_phrase(self) -> str:
        subject_terms = []
        if self.age:
            subject_terms.append(self.age)
        if self.race:
            subject_terms.append(self.race)
        if self.gender:
            subject_terms.append(self.gender)
        subject_terms.append("person" if self.age and "adult" in self.age.lower() else "adult person")
        return " ".join(subject_terms)

    def detail_phrases(self) -> list[str]:
        hair = _combine_attribute_labels(
            (self.hair, self.hair_texture, self.hair_color),
            noun="hair",
        )
        eyebrows = _combine_attribute_labels(
            (self.eyebrow_color, self.eyebrows),
            noun="eyebrows",
        )
        detail_fields = [
            hair,
            self.facial_hair,
            self.eyes,
            eyebrows,
            self.nose,
            self.build,
            self.face_shape,
            self.forehead,
            self.mouth,
            self.ears,
            self.jaw,
            self.teeth,
            self.expression,
            self.clothing,
            self.accessories,
        ]
        custom = [value for _, value in self.custom_attributes]
        return [text for text in detail_fields if text] + custom + list(self.other_details)

    def to_generation_prompt(
        self, variation: str | None = None, *, description: str | None = None,
    ) -> str:
        """Build one portrait prompt, optionally using authoritative parser text."""
        if description is None:
            details = self.detail_phrases()
            if self.attribute_values():
                description = self.subject_phrase()
                if details:
                    description += " with " + ", ".join(details)
            else:
                description = "; ".join(filter(None, (self.original_description, *details)))
        if variation:
            description = "; ".join(filter(None, (description, variation)))

        subject = {"female": "woman", "non-binary": "person"}.get(self.gender, "man")
        prompt = (
            f"Make a photo of an under average-looking {subject}, with facial features "
            "and hair style completely different from each other"
        )
        if description:
            prompt += f", but matching the following description: {description}"
        prompt = prompt.rstrip(". ") + ". "
        return prompt + _PORTRAIT_INSTRUCTIONS


def _combine_attribute_labels(values: tuple[str | None, ...], *, noun: str) -> str | None:
    labels = [value for value in values if value]
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    if "bald head" in labels:
        return "bald head"

    stems = []
    for label in labels:
        stem = re.sub(rf"\s+{re.escape(noun)}$", "", label, flags=re.IGNORECASE)
        if stem == "visible" and len(labels) > 1:
            continue
        if stem and stem not in stems:
            stems.append(stem)
    return f"{' '.join(stems)} {noun}" if stems else labels[0]


@dataclass(frozen=True)
class FaceAttributeDefinition:
    """Parsing, contrast, and prompt rules for one face attribute."""

    name: str
    order: int
    patterns: tuple[tuple[str, str], ...] = ()
    contrasts: Mapping[str, str] = field(default_factory=dict)
    parser: AttributeParser | None = field(default=None, repr=False, compare=False)
    prompt_role: PromptRole = "detail"
    sample_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _attribute_name(self.name)
        if not name:
            raise ValueError("Attribute name must not be empty.")
        if self.prompt_role not in {"subject", "detail"}:
            raise ValueError("prompt_role must be 'subject' or 'detail'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "contrasts", MappingProxyType(dict(self.contrasts)))
        choices = tuple(self.sample_values)
        if any(not isinstance(value, str) or not value.strip() for value in choices):
            raise ValueError("sample_values must contain non-empty strings.")
        if len(choices) != len(set(choices)):
            raise ValueError("sample_values must be unique.")
        object.__setattr__(self, "sample_values", choices)

    def parse(self, normalised_description: str) -> str | None:
        if self.parser is not None:
            return self.parser(normalised_description)
        return _first_match(normalised_description, self.patterns)

    def contrast_for(self, value: str) -> str | None:
        return self.contrasts.get(value)


class FaceAttributeSchema:
    """Immutable, extensible collection of ordered face-attribute rules."""

    def __init__(
        self,
        definitions: Iterable[FaceAttributeDefinition],
        *,
        name: str = "face_attributes",
        version: str = "1.0.0",
    ) -> None:
        ordered = tuple(sorted(definitions, key=lambda item: (item.order, item.name)))
        names = [definition.name for definition in ordered]
        duplicates = sorted({item for item in names if names.count(item) > 1})
        if duplicates:
            raise ValueError(f"Duplicate attribute definitions: {duplicates}")
        self.name = str(name).strip() or "face_attributes"
        self.version = str(version).strip() or "1.0.0"
        self._definitions = ordered
        self._by_name = MappingProxyType({definition.name: definition for definition in ordered})

    @property
    def definitions(self) -> tuple[FaceAttributeDefinition, ...]:
        return self._definitions

    @property
    def feature_order(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self._definitions)

    def get(self, name: str) -> FaceAttributeDefinition | None:
        return self._by_name.get(_attribute_name(name))

    def require(self, name: str) -> FaceAttributeDefinition:
        definition = self.get(name)
        if definition is None:
            raise KeyError(f"Unknown face attribute {name!r}. Available: {self.feature_order}")
        return definition

    def parse(self, description: str) -> dict[str, str]:
        text = _normalise_text(description)
        values = {}
        for definition in self._definitions:
            value = definition.parse(text)
            if value:
                values[definition.name] = value
        return values

    def sample(self, *, rng: random.Random | None = None) -> dict[str, str]:
        """Sample only declared values; omitted fields receive no invented value.

        Reuse a seeded Random instance to produce a reproducible batch. Hair
        colour and texture are omitted when the sampled head is bald.
        """
        rng = rng if rng is not None else random.Random()
        values = {
            definition.name: rng.choice(definition.sample_values)
            for definition in self._definitions if definition.sample_values
        }
        if values.get("hair") == "bald head":
            values.pop("hair_color", None)
            values.pop("hair_texture", None)
        return values

    def contrast_for(
        self,
        name: str,
        value: str,
        *,
        override: str | None = None,
    ) -> str | None:
        if override:
            return override
        definition = self.get(name)
        return definition.contrast_for(value) if definition is not None else None

    def extend(
        self,
        definitions: FaceAttributeDefinition | Iterable[FaceAttributeDefinition],
        *,
        replace: bool = False,
        name: str | None = None,
        version: str | None = None,
    ) -> "FaceAttributeSchema":
        if isinstance(definitions, FaceAttributeDefinition):
            additions = (definitions,)
        else:
            additions = tuple(definitions)

        merged = {definition.name: definition for definition in self._definitions}
        for definition in additions:
            if definition.name in merged and not replace:
                raise ValueError(
                    f"Attribute {definition.name!r} already exists; pass replace=True to replace it."
                )
            merged[definition.name] = definition
        return FaceAttributeSchema(
            merged.values(),
            name=name or self.name,
            version=version or self.version,
        )

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "name": definition.name,
                "order": definition.order,
                "prompt_role": definition.prompt_role,
                "patterns": list(definition.patterns),
                "parser": (
                    getattr(definition.parser, "__qualname__", None)
                    if definition.parser is not None
                    else None
                ),
                "contrasts": dict(definition.contrasts),
                **({"sample_values": list(definition.sample_values)} if definition.sample_values else {}),
            }
            for definition in self._definitions
        ]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _attribute_name(name) in self._by_name

    def __repr__(self) -> str:
        return (
            f"FaceAttributeSchema(name={self.name!r}, version={self.version!r}, "
            f"features={self.feature_order!r})"
        )


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _attribute_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


_UNCERTAINTY_CUE_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|may|might|not sure|unsure|uncertain|unclear|"
    r"i (?:think|believe|guess)|(?:hard|difficult|unable) to (?:tell|see|remember|recall)|"
    r"could (?:be|have)|"
    r"can(?:not|'t) (?:tell|see|say|remember|recall|make out|be (?:sure|certain))|"
    r"could(?: not|n't) (?:tell|see|say|remember|recall|make out))\b",
    re.IGNORECASE,
)
_NEGATION_CUE_RE = re.compile(
    r"\b(?:no|not|never|neither|without|cannot|"
    r"(?:do|does|did|is|are|was|were|has|have|had|can|could|would|should)n['’]?t)\b",
    re.IGNORECASE,
)
_NON_PREFIX_RE = re.compile(r"\bnon\s*[-–— ]\s*$", re.IGNORECASE)
_SCOPE_BOUNDARY_RE = re.compile(r"[,;:.!?]|\b(?:but|however)\b", re.IGNORECASE)
_POSTPOSED_UNCERTAINTY_RE = re.compile(
    r"^\s*,?\s*(?:maybe|perhaps|possibly|probably|i (?:think|believe|guess))"
    r"(?=\s*(?:[,;:.!?]|$))",
    re.IGNORECASE,
)
_DISJUNCTION_CUE_RE = re.compile(
    r"\b(?:either|or)\b|(?<=[a-z])\s*/\s*(?=[a-z])",
    re.IGNORECASE,
)


def _scope_prefix(text: str, start: int) -> str:
    prefix = text[:start]
    boundary = 0
    for match in _SCOPE_BOUNDARY_RE.finditer(prefix):
        boundary = match.end()
    return prefix[boundary:]


def _scope_segment(text: str, start: int, end: int) -> str:
    """Return the punctuation/contrast-delimited segment containing a match."""
    left = start - len(_scope_prefix(text, start))
    following = _SCOPE_BOUNDARY_RE.search(text, end)
    right = following.start() if following is not None else len(text)
    return text[left:right]


def _match_has_asserted_polarity(text: str, match: re.Match[str]) -> bool:
    """Check local polarity without treating alternatives as assertions."""
    prefix = _scope_prefix(text, match.start())
    return not (
        _UNCERTAINTY_CUE_RE.search(prefix)
        or _NEGATION_CUE_RE.search(prefix)
        or _NON_PREFIX_RE.search(prefix)
        or _POSTPOSED_UNCERTAINTY_RE.search(text[match.end() :])
    )


def _match_is_disjunctive(text: str, match: re.Match[str]) -> bool:
    segment = _scope_segment(text, match.start(), match.end())
    # ``y/o`` is an age-unit abbreviation, not an alternative construction.
    segment = re.sub(r"\by\s*/\s*o\b", "", segment, flags=re.IGNORECASE)
    return bool(_DISJUNCTION_CUE_RE.search(segment))


def _match_is_asserted(text: str, match: re.Match[str]) -> bool:
    """Reject locally negated, uncertain, or alternative matches."""
    return _match_has_asserted_polarity(text, match) and not _match_is_disjunctive(
        text, match
    )


def _matched_values(
    text: str,
    patterns: Sequence[tuple[str, str]],
) -> list[str]:
    matches: list[tuple[int, str]] = []
    for pattern, value in patterns:
        for match in re.finditer(pattern, text):
            if _match_is_asserted(text, match):
                matches.append((match.start(), value))
    matches.sort(key=lambda item: item[0])
    return list(dict.fromkeys(value for _, value in matches))


def _first_match(text: str, patterns: Sequence[tuple[str, str]]) -> str | None:
    """Return one unambiguous asserted value; conflicting values abstain."""
    values = _matched_values(text, patterns)
    return values[0] if len(values) == 1 else None


def _parse_age(text: str) -> str | None:
    numeric_matches: list[tuple[int, int]] = []
    coordinated = False

    # Capture a shared unit in constructions such as ``16 or 30 years old``
    # so that the first value cannot evade the adult-only guard.
    age_separator = r"(?:or|and|to|[-–—/])"
    shared_unit = re.compile(
        rf"\b(\d{{1,3}})\s*{age_separator}\s*(\d{{1,3}})"
        r"(?:\s+years?\s+old|[- ]year[- ]old|\s+y/?o)\b"
    )
    for match in shared_unit.finditer(text):
        if not _match_has_asserted_polarity(text, match):
            continue
        coordinated = True
        numeric_matches.extend(
            (match.start(index), int(match.group(index))) for index in (1, 2)
        )

    shared_aged_prefix = re.compile(
        rf"\b(?:age|aged)\s+(?:between\s+)?(\d{{1,3}})\s*"
        rf"{age_separator}\s*(\d{{1,3}})\b"
    )
    for match in shared_aged_prefix.finditer(text):
        if not _match_has_asserted_polarity(text, match):
            continue
        coordinated = True
        numeric_matches.extend(
            (match.start(index), int(match.group(index))) for index in (1, 2)
        )

    for pattern in (
        r"\b(?:age|aged)\s+(\d{1,3})\b",
        r"\b(?:around|about|approximately|roughly)?\s*(\d{1,3})"
        r"(?:\s+years?\s+old|[- ]year[- ]old|\s+y/?o)\b",
    ):
        for match in re.finditer(pattern, text):
            if not _match_has_asserted_polarity(text, match):
                continue
            coordinated = coordinated or _match_is_disjunctive(text, match)
            numeric_matches.append((match.start(1), int(match.group(1))))

    # Validate every asserted explicit age before conflict or coordination can
    # cause abstention.  An underage value must never be hidden by an adult one.
    for _, age in numeric_matches:
        if not 18 <= age <= 120:
            raise ValueError(
                "The default synthetic-face schema supports adult ages 18 through 120."
            )

    ages = list(dict.fromkeys(age for _, age in sorted(numeric_matches)))
    if coordinated or len(ages) > 1:
        return None
    if ages:
        age = ages[0]
        return f"around {age} years old"
    return _first_match(
        text,
        (
            (
                r"\byoung(?:\s+\w+){0,2}\s+"
                r"(?:adult|person|man|woman|male|female|guy|dude|gentleman|lady)\b",
                "young adult",
            ),
            (
                r"\bmiddle[- ]aged(?:\s+\w+){0,2}\s+"
                r"(?:adult|person|man|woman|male|female|guy|dude|gentleman|lady)\b",
                "middle-aged adult",
            ),
            (
                r"\b(?:old|older)(?:\s+\w+){0,2}\s+"
                r"(?:adult|person|man|woman|male|female|guy|dude|gentleman|lady)\b",
                "older adult",
            ),
        ),
    )


_HAIR_MODIFIERS = r"(?:curly|wavy|straight|black|brown|blond|blonde|gray|grey|red|dark|light)"


def _parse_hair_length(text: str) -> str | None:
    patterns: list[tuple[str, str]] = [(r"\bbald(?:\s+head)?\b", "bald head")]
    for length, label in (
        ("long", "long hair"),
        ("short", "short hair"),
        (r"medium[- ]length", "medium-length hair"),
    ):
        patterns.append((rf"\b{length}(?:\s+{_HAIR_MODIFIERS})*\s+hair\b", label))
    return _first_match(text, tuple(patterns))


def _parse_hair_color(text: str) -> str | None:
    colors = (
        ("black", "black hair"),
        ("brown", "brown hair"),
        (r"blond(?:e)?", "blond hair"),
        (r"gr[ae]y", "gray hair"),
        ("red", "red hair"),
        ("dark", "dark hair"),
        ("light", "light hair"),
    )
    patterns: list[tuple[str, str]] = []
    for color, label in colors:
        patterns.extend(
            (
                (
                    rf"\b{color}(?:\s+(?:long|short|medium[- ]length|curly|wavy|straight))*\s+hair\b",
                    label,
                ),
                (
                    rf"\b(?:long|short|medium[- ]length|curly|wavy|straight)"
                    rf"(?:\s+(?:long|short|medium[- ]length|curly|wavy|straight))*\s+{color}\s+hair\b",
                    label,
                ),
            )
        )
    values = _matched_values(text, tuple(patterns))
    named = [value for value in values if value not in {"dark hair", "light hair"}]
    selected = named or values
    return selected[0] if len(selected) == 1 else None


def _parse_hair_texture(text: str) -> str | None:
    bridge = r"(?:long|short|medium[- ]length|black|brown|blond|blonde|gr[ae]y|red|dark|light)"
    return _first_match(
        text,
        (
            (rf"\bcurly(?:\s+{bridge}){{0,3}}\s+hair\b", "curly hair"),
            (rf"\bwavy(?:\s+{bridge}){{0,3}}\s+hair\b", "wavy hair"),
            (rf"\bstraight(?:\s+{bridge}){{0,3}}\s+hair\b", "straight hair"),
        ),
    )


def _parse_facial_hair(text: str) -> str | None:
    return _first_match(
        text,
        (
            (r"\b(no facial hair|clean[- ]shaven|clean shaven)\b", "no facial hair"),
            (r"\bfull beard\b", "full beard"),
            (r"\bbig beard\b", "big beard"),
            (r"\bthick beard\b", "thick beard"),
            (r"(?<!full )(?<!big )(?<!thick )\bbeard(?:ed)?\b", "beard"),
            (r"\b(?:moustache|mustache)\b", "mustache"),
            (r"\bstubble\b", "stubble"),
        ),
    )


def _parse_eyebrow_color(text: str) -> str | None:
    modifiers = r"(?:bushy|thick|thin|fine|arched|straight)"
    patterns: list[tuple[str, str]] = []
    for color, label in (
        ("brown", "brown eyebrows"),
        ("black", "black eyebrows"),
        (r"blond(?:e)?", "blond eyebrows"),
        (r"gr[ae]y", "gray eyebrows"),
        ("red", "red eyebrows"),
    ):
        patterns.extend(
            (
                (rf"\b{color}(?:\s+{modifiers})*\s+eyebrows?\b", label),
                (rf"\b{modifiers}(?:\s+{modifiers})*\s+{color}\s+eyebrows?\b", label),
            )
        )
    return _first_match(text, tuple(patterns))


def _parse_modifier_sequence(
    text: str,
    *,
    head: str,
    aliases: Sequence[tuple[str, str]],
) -> str | None:
    alias_pattern = "(?:" + "|".join(alias for alias, _ in aliases) + ")"
    sequence = re.compile(
        rf"\b(?P<modifiers>{alias_pattern}"
        rf"(?:\s*(?:,|and)?\s*{alias_pattern})*)\s+{head}\b"
    )
    found: list[str] = []
    for match in sequence.finditer(text):
        if not _match_is_asserted(text, match):
            continue
        modifiers = match.group("modifiers")
        for alias, value in aliases:
            if re.search(rf"\b(?:{alias})\b", modifiers):
                found.append(value)
    values = list(dict.fromkeys(found))
    return values[0] if len(values) == 1 else None


def _parse_eyebrows(text: str) -> str | None:
    return _parse_modifier_sequence(
        text,
        head=r"eyebrows?",
        aliases=(
            (r"bushy", "bushy eyebrows"),
            (r"thick", "thick eyebrows"),
            (r"thin|fine", "thin eyebrows"),
            (r"arched", "arched eyebrows"),
            (r"straight", "straight eyebrows"),
        ),
    )


def _parse_nose(text: str) -> str | None:
    return _parse_modifier_sequence(
        text,
        head="nose",
        aliases=(
            (r"hawk", "hawk nose"),
            (r"aquiline", "aquiline nose"),
            (r"roman", "Roman nose"),
            (r"broad|wide", "broad nose"),
            (r"narrow|thin", "narrow nose"),
            (r"big|large", "large nose"),
            (r"small", "small nose"),
            (r"long", "long nose"),
            (r"short", "short nose"),
            (r"straight", "straight nose"),
            (r"pointed|pointy", "pointed nose"),
        ),
    )


def _parse_race(text: str) -> str | None:
    """Parse an explicitly person-bound demographic descriptor only."""
    subject = r"(?:adult|person|man|woman|male|female|dude|guy|gentleman|lady)"
    descriptors = (
        (r"east asian", "East Asian"),
        (r"south asian", "South Asian"),
        (r"(?<!east )(?<!south )asian", "Asian"),
        (r"white", "White"),
        (r"caucasian", "Caucasian"),
        (r"black", "Black"),
        (r"african american", "African American"),
        (r"latino|latina", "Latino"),
        (r"hispanic", "Hispanic"),
        (r"indian", "Indian"),
        (r"middle eastern", "Middle Eastern"),
        (r"arab", "Arab"),
    )
    patterns: list[tuple[str, str]] = []
    for descriptor, value in descriptors:
        patterns.extend(
            (
                (rf"\b(?:{descriptor})\s+(?:adult\s+)?{subject}\b", value),
                (
                    rf"\b{subject}\s+(?:who\s+)?(?:was|is|appeared)?\s*(?:{descriptor})\b",
                    value,
                ),
            )
        )
    return _first_match(text, tuple(patterns))


def _parse_expression(text: str) -> str | None:
    """Parse emotions only when wording describes a visible expression."""
    patterns: list[tuple[str, str]] = [
        (r"\bopen[- ]mouth\s+smile\b", "open-mouth smile"),
        (r"\bclosed[- ]mouth\s+smile\b", "closed-mouth smile"),
        (r"(?<!mouth )\bsmil(?:e|ing)\b", "smiling expression"),
        (r"\bneutral\s+expression\b", "neutral expression"),
    ]
    for emotion in ("angry", "sad", "surprised"):
        label = f"{emotion} expression"
        patterns.extend(
            (
                (rf"\b{emotion}(?:[- ]looking)?\s+(?:expression|look|face)\b", label),
                (rf"\b(?:looked|appeared|seemed)\s+{emotion}\b", label),
                (
                    rf"\b(?:expression|face|look)\s+(?:was|is|looked|appeared|seemed)\s+{emotion}\b",
                    label,
                ),
            )
        )
    return _first_match(text, tuple(patterns))


LEGACY_FACE_ATTRIBUTE_SCHEMA_V1 = FaceAttributeSchema(
    (
        FaceAttributeDefinition(
            "gender",
            10,
            patterns=(
                (r"\b(dude|guy|man|male|gentleman)\b", "male"),
                (r"\b(woman|female|lady)\b", "female"),
                (r"\b(nonbinary|non-binary)\b", "non-binary"),
            ),
            contrasts={"male": "female", "female": "male"},
            prompt_role="subject",
        ),
        FaceAttributeDefinition(
            "age",
            20,
            parser=_parse_age,
            contrasts={
                "young adult": "older adult",
                "middle-aged adult": "young adult",
                "older adult": "young adult",
            },
            prompt_role="subject",
        ),
        FaceAttributeDefinition(
            "hair",
            30,
            parser=_parse_hair_length,
            contrasts={
                "long hair": "short hair",
                "short hair": "long hair",
                "bald head": "visible hair",
                "visible hair": "bald head",
            },
        ),
        FaceAttributeDefinition(
            "hair_color",
            31,
            parser=_parse_hair_color,
            contrasts={
                "black hair": "blond hair",
                "brown hair": "blond hair",
                "blond hair": "brown hair",
                "gray hair": "dark hair",
                "red hair": "brown hair",
                "dark hair": "light hair",
                "light hair": "dark hair",
            },
        ),
        FaceAttributeDefinition(
            "hair_texture",
            32,
            parser=_parse_hair_texture,
            contrasts={
                "curly hair": "straight hair",
                "wavy hair": "straight hair",
                "straight hair": "curly hair",
            },
        ),
        FaceAttributeDefinition(
            "facial_hair",
            40,
            parser=_parse_facial_hair,
            contrasts={
                "beard": "no facial hair",
                "full beard": "no facial hair",
                "mustache": "no facial hair",
                "stubble": "no facial hair",
                "no facial hair": "full beard",
            },
        ),
        FaceAttributeDefinition(
            "eyes",
            50,
            patterns=(
                (r"\bblue\s+eyes?\b", "blue eyes"),
                (r"\bbrown\s+eyes?\b", "brown eyes"),
                (r"\bgreen\s+eyes?\b", "green eyes"),
                (r"\bdark\s+eyes?\b", "dark eyes"),
                (r"\blight\s+eyes?\b", "light eyes"),
            ),
            contrasts={
                "blue eyes": "brown eyes",
                "brown eyes": "blue eyes",
                "green eyes": "brown eyes",
                "dark eyes": "light eyes",
                "light eyes": "dark eyes",
            },
        ),
        FaceAttributeDefinition(
            "eyebrow_color",
            59,
            parser=_parse_eyebrow_color,
            contrasts={
                "brown eyebrows": "black eyebrows",
                "black eyebrows": "brown eyebrows",
                "blond eyebrows": "brown eyebrows",
                "gray eyebrows": "brown eyebrows",
                "red eyebrows": "brown eyebrows",
            },
        ),
        FaceAttributeDefinition(
            "eyebrows",
            60,
            parser=_parse_eyebrows,
            contrasts={
                "bushy eyebrows": "thin eyebrows",
                "thick eyebrows": "thin eyebrows",
                "thin eyebrows": "thick eyebrows",
                "arched eyebrows": "straight eyebrows",
                "straight eyebrows": "arched eyebrows",
            },
        ),
        FaceAttributeDefinition(
            "nose",
            70,
            parser=_parse_nose,
            contrasts={
                "hawk nose": "straight nose",
                "broad nose": "narrow nose",
                "narrow nose": "broad nose",
                "large nose": "small nose",
                "small nose": "large nose",
                "long nose": "short nose",
                "short nose": "long nose",
                "straight nose": "hawk nose",
            },
        ),
        FaceAttributeDefinition(
            "build",
            80,
            patterns=(
                (r"\bbroad\s+build\b", "broad build"),
                (r"\bheavy\s+build\b", "heavy build"),
                (r"\bstocky\s+build\b", "stocky build"),
                (r"\b(?:slim|thin|slender|skinny)\s+build\b", "slim build"),
            ),
            contrasts={"broad build": "slim build", "slim build": "broad build"},
        ),
        FaceAttributeDefinition(
            "face_shape",
            90,
            patterns=(
                (r"\bround\s+face\b", "round face"),
                (r"\boval\s+face\b", "oval face"),
                (r"\bnarrow\s+face\b", "narrow face"),
                (r"\blong\s+face\b", "long face"),
                (r"\bsquare\s+face\b", "square face"),
            ),
            contrasts={
                "round face": "narrow face",
                "narrow face": "round face",
                "oval face": "square face",
                "square face": "oval face",
            },
        ),
        FaceAttributeDefinition(
            "race",
            100,
            parser=_parse_race,
            contrasts={
                "White": "Asian",
                "Asian": "White",
                "Black": "White",
                "Latino": "White",
                "Indian": "White",
                "Middle Eastern": "White",
            },
            prompt_role="subject",
        ),
        FaceAttributeDefinition(
            "forehead",
            110,
            patterns=((r"\bhigh\s+forehead\b", "high forehead"), (r"\blow\s+forehead\b", "low forehead")),
            contrasts={"high forehead": "low forehead", "low forehead": "high forehead"},
        ),
        FaceAttributeDefinition(
            "mouth",
            120,
            patterns=(
                (r"\bthin\s+lips?\b", "thin lips"),
                (r"\bfull\s+lips?\b", "full lips"),
                (r"\bwide\s+mouth\b", "wide mouth"),
            ),
            contrasts={"thin lips": "full lips", "full lips": "thin lips"},
        ),
        FaceAttributeDefinition(
            "ears",
            130,
            patterns=(
                (r"\b(?:protrude|protruding)\s+ears?\b", "protruding ears"),
                (r"\bprominent\s+ears?\b", "prominent ears"),
                (r"\bvisible\s+ears?\b", "visible ears"),
                (r"\b(?:covered|hidden)\s+ears?\b", "covered ears"),
            ),
            contrasts={
                "protruding ears": "non-protruding ears",
                "visible ears": "covered ears",
                "covered ears": "visible ears",
            },
        ),
        FaceAttributeDefinition(
            "jaw",
            140,
            patterns=(
                (r"\bsquare\s+jaw(?:line)?\b", "square jaw"),
                (r"\bstrong\s+jaw(?:line)?\b", "strong jaw"),
                (r"\bsoft\s+jaw(?:line)?\b", "soft jawline"),
            ),
            contrasts={
                "square jaw": "soft jawline",
                "strong jaw": "soft jawline",
                "soft jawline": "strong jaw",
            },
        ),
        FaceAttributeDefinition(
            "teeth",
            150,
            patterns=(
                (r"\b(?:no|without)\s+visible\s+teeth\b", "no visible teeth"),
                (r"\bvisible\s+teeth\b", "visible teeth"),
            ),
            contrasts={"visible teeth": "no visible teeth", "no visible teeth": "visible teeth"},
        ),
        FaceAttributeDefinition(
            "expression",
            160,
            parser=_parse_expression,
            contrasts={
                "neutral expression": "smiling expression",
                "smiling expression": "neutral expression",
                "open-mouth smile": "neutral expression",
                "closed-mouth smile": "neutral expression",
            },
        ),
        FaceAttributeDefinition(
            "clothing",
            170,
            patterns=(
                (r"\b(?:gray|grey)\s+(?:shirt|t-shirt|tee|top)\b", "gray shirt"),
                (r"\b(?:wearing\s+)?(?:a\s+)?black\s+(?:shirt|t-shirt|tee|top)\b", "black shirt"),
                (r"\b(?:wearing\s+)?(?:a\s+)?white\s+(?:shirt|t-shirt|tee|top)\b", "white shirt"),
                (r"\b(?:wearing\s+)?(?:a\s+)?blue\s+(?:shirt|t-shirt|tee|top)\b", "blue shirt"),
                (r"\b(?:wearing\s+)?(?:a\s+)?red\s+(?:shirt|t-shirt|tee|top)\b", "red shirt"),
                (r"\b(?:wearing\s+)?(?:a\s+)?suit\b", "suit"),
                (r"\b(?:wearing\s+)?(?:a\s+)?hoodie\b", "hoodie"),
            ),
            contrasts={
                "gray shirt": "black shirt",
                "white shirt": "black shirt",
                "black shirt": "white shirt",
                "blue shirt": "red shirt",
                "red shirt": "blue shirt",
            },
        ),
        FaceAttributeDefinition(
            "accessories",
            180,
            patterns=(
                (r"\b(?:wearing\s+)?glasses\b", "glasses"),
                (r"\b(?:wearing\s+)?sunglasses\b", "sunglasses"),
                (r"\b(?:wearing\s+)?earrings?\b", "earrings"),
                (r"\b(?:wearing\s+)?(?:a\s+)?hat\b", "hat"),
                (r"\b(?:wearing\s+)?(?:a\s+)?cap\b", "cap"),
            ),
            contrasts={
                "glasses": "no glasses",
                "sunglasses": "no glasses",
                "earrings": "no earrings",
                "hat": "no hat",
                "cap": "no cap",
            },
        ),
    ),
    name="legacy_default_face_attributes_v1",
    version="1.0.0-legacy-single-negative",
)


# Sampling values are explicit canonical descriptions, independent of parsing
# patterns and scientific contrast choices. Neutral portraits keep the mouth
# closed and avoid accessories that obscure facial features.
_DEFAULT_SAMPLE_VALUES = {
    "gender": ("male", "female"),
    "age": ("around 20 years old", "around 30 years old", "around 40 years old", "around 50 years old", "around 60 years old", "around 70 years old"),
    "hair": ("short hair", "medium-length hair", "long hair", "bald head"),
    "hair_color": ("black hair", "brown hair", "blond hair", "gray hair", "red hair"),
    "hair_texture": ("straight hair", "wavy hair", "curly hair"),
    "facial_hair": ("no facial hair", "stubble", "beard", "mustache"),
    "eyes": ("blue eyes", "brown eyes", "green eyes"),
    "eyebrow_color": ("black eyebrows", "brown eyebrows", "blond eyebrows", "gray eyebrows", "red eyebrows"),
    "eyebrows": ("thick eyebrows", "thin eyebrows", "arched eyebrows", "straight eyebrows"),
    "nose": ("broad nose", "narrow nose", "large nose", "small nose", "straight nose", "aquiline nose"),
    "build": ("broad build", "slim build", "stocky build"),
    "face_shape": ("round face", "oval face", "long face", "square face"),
    "race": ("White", "Black", "East Asian", "South Asian", "Latino", "Middle Eastern"),
    "forehead": ("high forehead", "low forehead"),
    "mouth": ("thin lips", "full lips", "wide mouth"),
    "ears": ("protruding ears", "prominent ears", "visible ears"),
    "jaw": ("square jaw", "strong jaw", "soft jawline"),
    "teeth": ("no visible teeth",),
    "expression": ("neutral expression",),
    "clothing": ("gray shirt",),
}


# The project default deliberately supplies no contrast anchors. A generic face schema
# cannot choose scientifically valid negative anchors without a candidate-pool
# definition and a preregistered contrast design.  The legacy lookup table above
# is retained solely for auditing an old single-negative choice; new code must
# opt into it explicitly or supply a custom schema/contrast override.
DEFAULT_FACE_ATTRIBUTE_SCHEMA = FaceAttributeSchema(
    (
        replace(definition, contrasts={}, sample_values=_DEFAULT_SAMPLE_VALUES.get(definition.name, ()))
        for definition in LEGACY_FACE_ATTRIBUTE_SCHEMA_V1.definitions
    ),
    name="default_face_attributes",
    version="2.0.0-safe-parsing-only",
)
