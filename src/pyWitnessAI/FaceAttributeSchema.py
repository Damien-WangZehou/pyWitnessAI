from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Sequence

AttributeParser = Callable[[str], str | None]
PromptRole = Literal["subject", "detail"]

__all__ = [
    "DEFAULT_FACE_ATTRIBUTE_SCHEMA",
    "FaceAttributeDefinition",
    "FaceAttributeSchema",
]


@dataclass(frozen=True)
class FaceAttributeDefinition:
    """Parsing, contrast, and prompt rules for one face attribute."""

    name: str
    order: int
    patterns: tuple[tuple[str, str], ...] = ()
    contrasts: Mapping[str, str] = field(default_factory=dict)
    parser: AttributeParser | None = field(default=None, repr=False, compare=False)
    prompt_role: PromptRole = "detail"

    def __post_init__(self) -> None:
        name = _attribute_name(self.name)
        if not name:
            raise ValueError("Attribute name must not be empty.")
        if self.prompt_role not in {"subject", "detail"}:
            raise ValueError("prompt_role must be 'subject' or 'detail'.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "patterns", tuple(self.patterns))
        object.__setattr__(self, "contrasts", MappingProxyType(dict(self.contrasts)))

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
    ) -> None:
        ordered = tuple(sorted(definitions, key=lambda item: (item.order, item.name)))
        names = [definition.name for definition in ordered]
        duplicates = sorted({item for item in names if names.count(item) > 1})
        if duplicates:
            raise ValueError(f"Duplicate attribute definitions: {duplicates}")
        self.name = str(name).strip() or "face_attributes"
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
        return FaceAttributeSchema(merged.values(), name=name or self.name)

    def to_records(self) -> list[dict[str, object]]:
        return [
            {
                "name": definition.name,
                "order": definition.order,
                "prompt_role": definition.prompt_role,
                "contrasts": dict(definition.contrasts),
            }
            for definition in self._definitions
        ]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _attribute_name(name) in self._by_name

    def __repr__(self) -> str:
        return f"FaceAttributeSchema(name={self.name!r}, features={self.feature_order!r})"


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _attribute_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _first_match(text: str, patterns: Sequence[tuple[str, str]]) -> str | None:
    for pattern, value in patterns:
        if re.search(pattern, text):
            return value
    return None


def _parse_age(text: str) -> str | None:
    match = re.search(r"\b(?:around|about)?\s*(\d{2})\s*(?:years?\s*old|yo)?\b", text)
    if match:
        age = int(match.group(1))
        if age < 18:
            raise ValueError("FillerGenerator only supports synthetic adult faces; age must be 18 or older.")
        return f"around {age} years old"
    return _first_match(
        text,
        (
            (r"\byoung(?:\s+adult)?\b", "young adult"),
            (r"\bmiddle[- ]aged(?:\s+adult)?\b", "middle-aged adult"),
            (r"\b(?:old|older)(?:\s+adult)?\b", "older adult"),
        ),
    )


_HAIR_MODIFIERS = r"(?:curly|wavy|straight|black|brown|blond|blonde|gray|grey|red|dark|light)"


def _parse_hair_length(text: str) -> str | None:
    if re.search(r"\bbald(?:\s+head)?\b", text):
        return "bald head"
    for length, label in (
        ("long", "long hair"),
        ("short", "short hair"),
        (r"medium[- ]length", "medium-length hair"),
    ):
        if re.search(rf"\b{length}(?:\s+{_HAIR_MODIFIERS})*\s+hair\b", text):
            return label
    return "visible hair" if re.search(r"\bhair\b", text) else None


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
    for color, label in colors:
        if re.search(
            rf"\b{color}(?:\s+(?:long|short|medium[- ]length|curly|wavy|straight))*\s+hair\b",
            text,
        ):
            return label
        if re.search(
            rf"\b(?:long|short|medium[- ]length|curly|wavy|straight)(?:\s+(?:long|short|medium[- ]length|curly|wavy|straight))*\s+{color}\s+hair\b",
            text,
        ):
            return label
    return None


def _parse_hair_texture(text: str) -> str | None:
    return _first_match(
        text,
        (
            (r"\bcurly(?:\s+\w+){0,3}\s+hair\b", "curly hair"),
            (r"\bwavy(?:\s+\w+){0,3}\s+hair\b", "wavy hair"),
            (r"\bstraight(?:\s+\w+){0,3}\s+hair\b", "straight hair"),
        ),
    )


def _parse_facial_hair(text: str) -> str | None:
    return _first_match(
        text,
        (
            (r"\b(no beard|no facial hair|clean[- ]shaven|clean shaven)\b", "no facial hair"),
            (r"\b(full beard|big beard|thick beard)\b", "full beard"),
            (r"\bbeard(?:ed)?\b", "beard"),
            (r"\b(?:moustache|mustache)\b", "mustache"),
            (r"\bstubble\b", "stubble"),
        ),
    )


def _parse_eyebrow_color(text: str) -> str | None:
    modifiers = r"(?:bushy|thick|thin|fine|arched|straight)"
    for color, label in (
        ("brown", "brown eyebrows"),
        ("black", "black eyebrows"),
        (r"blond(?:e)?", "blond eyebrows"),
        (r"gr[ae]y", "gray eyebrows"),
        ("red", "red eyebrows"),
    ):
        if re.search(rf"\b{color}(?:\s+{modifiers})*\s+eyebrows?\b", text):
            return label
        if re.search(rf"\b{modifiers}(?:\s+{modifiers})*\s+{color}\s+eyebrows?\b", text):
            return label
    return None


DEFAULT_FACE_ATTRIBUTE_SCHEMA = FaceAttributeSchema(
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
            patterns=(
                (r"\bbushy\s+eyebrows?\b", "bushy eyebrows"),
                (r"\bthick\s+eyebrows?\b", "thick eyebrows"),
                (r"\b(?:thin|fine)\s+eyebrows?\b", "thin eyebrows"),
                (r"\barched\s+eyebrows?\b", "arched eyebrows"),
                (r"\bstraight\s+eyebrows?\b", "straight eyebrows"),
            ),
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
            patterns=(
                (r"\b(?:hawk|aquiline|roman)\s+nose\b", "hawk nose"),
                (r"\b(?:broad|wide)\s+nose\b", "broad nose"),
                (r"\b(?:narrow|thin)\s+nose\b", "narrow nose"),
                (r"\b(?:big|large)\s+nose\b", "large nose"),
                (r"\bsmall\s+nose\b", "small nose"),
                (r"\blong\s+nose\b", "long nose"),
                (r"\bshort\s+nose\b", "short nose"),
                (r"\bstraight\s+nose\b", "straight nose"),
            ),
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
                (r"\b(?:broad|heavy|stocky)\s+build\b", "broad build"),
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
                (r"\b(?:narrow|long)\s+face\b", "narrow face"),
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
            patterns=(
                (r"\b(?:white|caucasian)\b", "White"),
                (r"\b(?:asian|east asian|south asian)\b", "Asian"),
                (r"\b(?:black|african)\b", "Black"),
                (r"\b(?:latino|latina|hispanic)\b", "Latino"),
                (r"\bindian\b", "Indian"),
                (r"\b(?:middle eastern|arab)\b", "Middle Eastern"),
            ),
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
                (r"\b(?:protrude|protruding|prominent)\s+ears?\b", "protruding ears"),
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
            patterns=(
                (r"\bopen[- ]mouth\s+smile\b", "open-mouth smile"),
                (r"\bclosed[- ]mouth\s+smile\b", "closed-mouth smile"),
                (r"\bsmil(?:e|ing)\b", "smiling expression"),
                (r"\bneutral\s+expression\b", "neutral expression"),
                (r"\bangry\b", "angry expression"),
                (r"\bsad\b", "sad expression"),
                (r"\bsurprised\b", "surprised expression"),
            ),
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
    name="default_face_attributes",
)
