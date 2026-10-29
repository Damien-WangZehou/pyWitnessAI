"""Shared attribute registry and English scope rules for description parsing.

Both text annotation and contrast prompt construction use these definitions.
Study-specific templates and pool restrictions are supplied by callers.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

PROMPT_TEMPLATE = "A face photograph of a person with {descriptor}."


AGE_PROMPT_TEMPLATE = "A face photograph of a person who appears {descriptor}."


ContrastKind = Literal["binary_opposite", "one_vs_rest", "positive_only"]


@dataclass(frozen=True)
class ContrastPromptRule:
    rule_id: str
    family: str
    dimension: str
    value: str
    descriptor: str
    alternatives: tuple[str, ...]
    aliases: tuple[str, ...]
    contrast_kind: ContrastKind = "binary_opposite"
    weight: float = 1.0
    template: str = PROMPT_TEMPLATE

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in (self.rule_id, self.family, self.dimension, self.value)):
            raise ValueError("Contrast rule identifiers must not be empty.")
        if not self.aliases:
            raise ValueError(f"Rule {self.rule_id!r} requires at least one alias.")
        if self.value in self.alternatives:
            raise ValueError(f"Rule {self.rule_id!r} includes its positive value as an alternative.")
        if self.weight <= 0:
            raise ValueError(f"Rule {self.rule_id!r} requires weight > 0.")
        if "{descriptor}" not in self.template:
            raise ValueError(f"Rule {self.rule_id!r} template lacks {{descriptor}}.")


@dataclass(frozen=True)
class ContrastPromptSchema:
    name: str
    version: str
    rules: tuple[ContrastPromptRule, ...]
    pool_profile: str = "unrestricted"
    hash: str = field(init=False)

    def __post_init__(self) -> None:
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Contrast rule_id values must be unique.")
        keys = [(rule.family, rule.dimension, rule.value) for rule in self.rules]
        if len(keys) != len(set(keys)):
            raise ValueError("Contrast schema family/dimension/value keys must be unique.")
        payload = {
            "name": self.name,
            "version": self.version,
            "pool_profile": self.pool_profile,
            "rules": [asdict(rule) for rule in self.rules],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "hash", digest)

    def require(self, family: str, dimension: str, value: str) -> ContrastPromptRule:
        for rule in self.rules:
            if (rule.family, rule.dimension, rule.value) == (family, dimension, value):
                return rule
        raise KeyError(f"Unknown contrast rule: {family}.{dimension}={value}")

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    **asdict(rule),
                    "alternatives": "|".join(rule.alternatives),
                    "aliases": "|".join(rule.aliases),
                    "schema_name": self.name,
                    "schema_version": self.version,
                    "schema_hash": self.hash,
                    "pool_profile": self.pool_profile,
                }
                for rule in self.rules
            ]
        )


@dataclass(frozen=True)
class Abstention:
    reason_code: str
    feature: str = ""
    dimension: str = ""
    rule_id: str = ""
    normalised_span: str = ""

    def to_dict(self, *, include_text: bool = False) -> dict[str, object]:
        output = asdict(self)
        if not include_text:
            output.pop("normalised_span", None)
        return output


def _r(
    family: str,
    dimension: str,
    value: str,
    descriptor: str,
    alternatives: Sequence[str],
    aliases: Sequence[str],
    *,
    contrast_kind: ContrastKind = "binary_opposite",
    template: str = PROMPT_TEMPLATE,
) -> ContrastPromptRule:
    return ContrastPromptRule(
        rule_id=f"strict_v1.{family}.{dimension}.{value}",
        family=family,
        dimension=dimension,
        value=value,
        descriptor=descriptor,
        alternatives=tuple(alternatives),
        aliases=tuple(aliases),
        contrast_kind=contrast_kind,
        template=template,
    )


_RULES = (
    _r("hair", "length", "short", "short hair", ("medium-length hair", "long hair"), ("short", "cropped"), contrast_kind="one_vs_rest"),
    _r("hair", "length", "long", "long hair", ("short hair", "medium-length hair"), ("long",)),
    _r("hair", "tone", "dark", "dark hair", ("light hair",), ("dark",)),
    _r("hair", "tone", "light", "light hair", ("dark hair",), ("light", "fair")),
    _r("hair", "hue", "black", "black hair", ("brown hair", "blond hair", "red hair", "grey hair"), ("black",), contrast_kind="one_vs_rest"),
    _r("hair", "hue", "brown", "brown hair", ("black hair", "blond hair", "red hair", "grey hair"), ("brown",), contrast_kind="one_vs_rest"),
    _r("hair", "hue", "blond", "blond hair", ("black hair", "brown hair", "red hair", "grey hair"), ("blond", "blonde"), contrast_kind="one_vs_rest"),
    _r("hair", "hue", "red", "red hair", ("black hair", "brown hair", "blond hair", "grey hair"), ("red", "ginger", "auburn"), contrast_kind="one_vs_rest"),
    _r("hair", "hue", "grey", "grey hair", ("black hair", "brown hair", "blond hair", "red hair"), ("grey", "gray", "silver"), contrast_kind="one_vs_rest"),
    _r("hair", "texture", "curly", "curly hair", ("straight hair",), ("curly", "curled")),
    _r("hair", "texture", "straight", "straight hair", ("curly hair",), ("straight",)),
    _r("hair", "density", "bald", "a bald head", ("a full head of hair",), ("bald", "shaved")),
    _r("hair", "density", "balding", "a balding head", ("a full head of hair",), ("balding", "receding")),
    _r("hair", "density", "full", "a full head of hair", ("a bald head", "a balding head"), ("full", "thick"), contrast_kind="one_vs_rest"),
    _r("facial_hair", "state", "beard", "a beard", ("stubble", "a moustache", "a clean-shaven face"), ("beard", "bearded"), contrast_kind="one_vs_rest"),
    _r("facial_hair", "state", "stubble", "stubble", ("a beard", "a moustache", "a clean-shaven face"), ("stubble", "unshaven"), contrast_kind="one_vs_rest"),
    _r("facial_hair", "state", "moustache", "a moustache", ("a beard", "stubble", "a clean-shaven face"), ("moustache", "mustache"), contrast_kind="one_vs_rest"),
    _r("facial_hair", "state", "clean_shaven", "a clean-shaven face", ("a beard", "stubble", "a moustache"), ("clean-shaven", "clean shaven", "no facial hair"), contrast_kind="one_vs_rest"),
    _r("eyes", "hue", "blue", "blue eyes", ("brown eyes", "green eyes", "hazel eyes", "grey eyes"), ("blue",), contrast_kind="one_vs_rest"),
    _r("eyes", "hue", "brown", "brown eyes", ("blue eyes", "green eyes", "hazel eyes", "grey eyes"), ("brown",), contrast_kind="one_vs_rest"),
    _r("eyes", "hue", "green", "green eyes", ("blue eyes", "brown eyes", "hazel eyes", "grey eyes"), ("green",), contrast_kind="one_vs_rest"),
    _r("eyes", "hue", "hazel", "hazel eyes", ("blue eyes", "brown eyes", "green eyes", "grey eyes"), ("hazel",), contrast_kind="one_vs_rest"),
    _r("eyes", "hue", "grey", "grey eyes", ("blue eyes", "brown eyes", "green eyes", "hazel eyes"), ("grey", "gray"), contrast_kind="one_vs_rest"),
    _r("eyes", "size", "large", "large eyes", ("small eyes",), ("large", "big")),
    _r("eyes", "size", "small", "small eyes", ("large eyes",), ("small",)),
    _r("eyes", "spacing", "close_set", "close-set eyes", ("wide-set eyes",), ("close-set", "close set")),
    _r("eyes", "spacing", "wide_set", "wide-set eyes", ("close-set eyes",), ("wide-set", "wide set")),
    _r("eyes", "shape", "round", "round eyes", ("narrow eyes",), ("round",)),
    _r("eyes", "shape", "narrow", "narrow eyes", ("round eyes",), ("narrow",)),
    _r("brows", "thickness", "thick", "thick eyebrows", ("thin eyebrows",), ("thick", "bushy", "dense")),
    _r("brows", "thickness", "thin", "thin eyebrows", ("thick eyebrows",), ("thin", "fine", "sparse")),
    _r("brows", "shape", "arched", "arched eyebrows", ("straight eyebrows",), ("arched", "curved")),
    _r("brows", "shape", "straight", "straight eyebrows", ("arched eyebrows",), ("straight",)),
    _r("nose", "size", "large", "a large nose", ("a small nose",), ("large", "big")),
    _r("nose", "size", "small", "a small nose", ("a large nose",), ("small",)),
    _r("nose", "length", "long", "a long nose", ("a short nose",), ("long",)),
    _r("nose", "length", "short", "a short nose", ("a long nose",), ("short",)),
    _r("nose", "width", "broad", "a broad nose", ("a narrow nose",), ("broad", "wide")),
    _r("nose", "width", "narrow", "a narrow nose", ("a broad nose",), ("narrow", "thin")),
    _r("nose", "tip", "pointed", "a pointed nose", ("a round-tipped nose",), ("pointed", "pointy")),
    _r("nose", "tip", "round", "a round-tipped nose", ("a pointed nose",), ("round", "rounded")),
    _r("nose", "profile", "hooked", "a hooked nose", ("a straight nose",), ("hooked", "crooked", "hawk")),
    _r("nose", "profile", "straight", "a straight nose", ("a hooked nose",), ("straight",)),
    _r("lips", "fullness", "thin", "thin lips", ("full lips",), ("thin",)),
    _r("lips", "fullness", "full", "full lips", ("thin lips",), ("full", "thick")),
    _r("mouth", "width", "wide", "a wide mouth", ("a narrow mouth",), ("wide", "broad")),
    _r("mouth", "width", "narrow", "a narrow mouth", ("a wide mouth",), ("narrow", "small")),
    _r("chin", "shape", "pointed", "a pointed chin", ("a round chin",), ("pointed", "pointy")),
    _r("chin", "shape", "round", "a round chin", ("a pointed chin",), ("round", "rounded")),
    _r("chin", "length", "long", "a long chin", ("a short chin",), ("long",)),
    _r("chin", "length", "short", "a short chin", ("a long chin",), ("short",)),
    _r("ears", "size", "large", "large ears", ("small ears",), ("large", "big")),
    _r("ears", "size", "small", "small ears", ("large ears",), ("small",)),
    _r("ears", "protrusion", "protruding", "protruding ears", ("ears lying close to the head",), ("protruding", "prominent", "sticking out")),
    _r("ears", "protrusion", "close", "ears lying close to the head", ("protruding ears",), ("close", "flat")),
    _r("face", "shape", "round", "a round face", ("a narrow face",), ("round", "rounded")),
    _r("face", "shape", "narrow", "a narrow face", ("a round face",), ("narrow", "long")),
    _r("forehead", "height", "high", "a high forehead", ("a low forehead",), ("high", "tall")),
    _r("forehead", "height", "low", "a low forehead", ("a high forehead",), ("low", "short")),
    _r("age", "band", "young", "young", ("middle-aged", "older"), ("young", "youthful", "in his twenties"), contrast_kind="one_vs_rest", template=AGE_PROMPT_TEMPLATE),
    _r("age", "band", "middle_aged", "middle-aged", ("young", "older"), ("middle-aged", "middle aged", "in his forties", "in his fifties"), contrast_kind="one_vs_rest", template=AGE_PROMPT_TEMPLATE),
    _r("age", "band", "older", "older", ("young", "middle-aged"), ("older", "old", "elderly"), contrast_kind="one_vs_rest", template=AGE_PROMPT_TEMPLATE),
)


GENERIC_CONTRAST_SCHEMA = ContrastPromptSchema(
    name="Generic eyewitness attribute contrast registry",
    version="1.0.0",
    pool_profile="unrestricted",
    rules=_RULES,
)


_HEAD_ALIASES = {
    "hair": (
        "hair",
        "hairstyle",
        "bald",
        "balding",
        "shaved head",
        "receding hairline",
    ),
    "facial_hair": (
        "facial hair",
        "beard",
        "bearded",
        "moustache",
        "mustache",
        "stubble",
        "unshaven",
        "clean-shaven",
        "clean shaven",
    ),
    "eyes": ("eye", "eyes"),
    "brows": ("eyebrow", "eyebrows", "brow", "brows"),
    "nose": ("nose",),
    "lips": ("lip", "lips"),
    "mouth": ("mouth",),
    "chin": ("chin",),
    "ears": ("ear", "ears"),
    "face": ("face",),
    "forehead": ("forehead",),
    "age": ("aged", "age", "year-old", "years old", "young", "older", "elderly", "middle-aged", "middle aged"),
}


_ALL_HEADS = tuple(sorted({item for values in _HEAD_ALIASES.values() for item in values}, key=len, reverse=True))


_AUDIT_ONLY_HEADS = (
    "glasses", "spectacles", "hat", "cap", "earring", "scar", "scars", "mark", "marks",
    "shirt", "jacket", "coat", "hoodie", "clothes", "clothing", "tie",
    "voice", "accent", "build", "height", "teeth", "tooth", "skin", "complexion",
)


_BOUNDARY_WORDS = {
    "with", "had", "has", "have", "having", "but", "although", "while", "wearing",
    "shirt", "jacket", "coat", "clothes", "clothing", "man", "person", "male", "guy",
}


_UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|may|might|i think|i believe|"
    r"i (?:would )?guess|seem(?:s|ed)? to remember|"
    r"i (?:do not|don't|did not|didn't) think|not sure|unsure|"
    r"(?:i )?(?:can(?:not|'t)|could(?: not|n't)) say|"
    r"(?:hard|difficult) to (?:tell|see|remember|recall|make out)|"
    r"could(?:n['’]?t| not) (?:see|remember|recall|tell|make out)|"
    r"can(?:not|'t) (?:see|remember|recall|tell|make out|be (?:sure|certain))|"
    r"(?:(?:(?:i|he|she|they|we) )?(?:was|were|am|is|are) )?unable to "
    r"(?:tell|see|remember|recall|say)|"
    r"don['’]?t (?:know|remember|recall))\b",
    re.IGNORECASE,
)


_LOGICAL_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|without|"
    r"(?:do|does|did|is|are|was|were|has|have|had|can|could|would|should)n['’]?t)\b",
    re.IGNORECASE,
)


_LANDMARK_RE = re.compile(
    r"\bhair\b.{0,28}\b(?:down\s+)?to\s+(?:his\s+)?(?:ears?|jaw|neck)\b|"
    r"\b(?:ears?|jaw|neck)\b.{0,28}\bhair\b",
    re.IGNORECASE,
)


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def _midpoint_near_alias(context: str, alias: str) -> bool:
    """Detect a midpoint modifier immediately competing with an extreme."""
    midpoint = r"(?:medium|average|normal|ordinary)"
    escaped = re.escape(alias)
    bridge = r"(?:\W+[a-z]+){0,1}\W+"
    return bool(
        re.search(rf"\b{midpoint}\b{bridge}\b{escaped}\b", context)
        or re.search(rf"\b{escaped}\b{bridge}\b{midpoint}\b", context)
    )


def _split_feature_conjunctions(clause: str) -> list[str]:
    """Split independent feature-list items without losing polarity scope."""
    # Uncertainty can scope over an entire coordinated/list construction.  A
    # later definite assertion is already separated at hard "but/however"
    # boundaries in ``build``; otherwise strict-v1 abstains from the whole
    # clause instead of guessing where uncertainty ends.
    if _UNCERTAINTY_RE.search(clause):
        return [clause.strip()]
    scoped = bool(_LOGICAL_NEGATION_RE.search(clause))
    # Commas conventionally delimit separate feature-list items, and splitting
    # them also prevents a post-head adjective from leaking into the next item.
    for match in re.finditer(r",", clause):
        left = clause[: match.start()].strip(" ,")
        right = clause[match.end() :].strip(" ,")
        if (
            left
            and right
            and _contains_feature_head(left)
            and _contains_feature_head(right)
            and (not scoped or _safe_polarity_scope_split(left, right))
        ):
            return [
                *_split_feature_conjunctions(left),
                *_split_feature_conjunctions(right),
            ]

    # ``with`` normally joins separate list items rather than sharing a
    # negator ("clean-shaven with no glasses").
    for match in re.finditer(r"\bwith\b", clause):
        left = clause[: match.start()].strip(" ,")
        right = clause[match.end() :].strip(" ,")
        if (
            left
            and right
            and _contains_feature_head(left)
            and _contains_feature_head(right)
            and (not scoped or _safe_polarity_scope_split(left, right))
        ):
            return [
                *_split_feature_conjunctions(left),
                *_split_feature_conjunctions(right),
            ]

    # A fresh subject/predicate after an audit-only phrase starts a new
    # assertion: "without glasses he had a large nose".
    for match in re.finditer(
        r"\b(?:he|the man|the male|the person)\s+(?:had|has|was|is|looked|appeared|seemed)\b",
        clause,
    ):
        left = clause[: match.start()].strip(" ,")
        right = clause[match.start() :].strip(" ,")
        if left and right and _contains_feature_head(left) and _contains_feature_head(right):
            return [
                *_split_feature_conjunctions(left),
                *_split_feature_conjunctions(right),
            ]

    # A negator/uncertainty marker can govern coordinated phrases ("did not
    # have a beard and a moustache").  Split only the same pre-registered safe
    # cases used for comma/with; otherwise conservatively keep the whole scope.
    if scoped:
        for match in re.finditer(r"\band\b", clause):
            left = clause[: match.start()].strip(" ,")
            right = clause[match.end() :].strip(" ,")
            if (
                left
                and right
                and _contains_feature_head(left)
                and _contains_feature_head(right)
                and _safe_polarity_scope_split(left, right)
            ):
                return [
                    *_split_feature_conjunctions(left),
                    *_split_feature_conjunctions(right),
                ]
        return [clause.strip()]

    matches = list(re.finditer(r"\band\b", clause))
    for match in matches:
        left = clause[: match.start()].strip(" ,")
        right = clause[match.end() :].strip(" ,")
        if left and right and _contains_feature_head(left) and _contains_feature_head(right):
            return [
                *_split_feature_conjunctions(left),
                *_split_feature_conjunctions(right),
            ]
    return [clause.strip()]


def _contains_feature_head(text: str) -> bool:
    return any(
        re.search(rf"\b{re.escape(head)}\b", text)
        for head in (*_ALL_HEADS, *_AUDIT_ONLY_HEADS)
    )


def _safe_polarity_scope_split(left: str, right: str) -> bool:
    """Allow a scoped split only when independence is explicit and safe."""
    def audit_only(value: str) -> bool:
        has_audit_head = any(
            re.search(rf"\b{re.escape(head)}\b", value) for head in _AUDIT_ONLY_HEADS
        )
        has_scored_head = any(
            re.search(rf"\b{re.escape(head)}\b", value) for head in _ALL_HEADS
        )
        return has_audit_head and not has_scored_head

    lexical_absence = bool(re.search(r"\bno facial hair\b", left))
    explicit_subject = bool(
        re.match(
            r"(?:he|the man|the male|the person)\s+"
            r"(?:had|has|was|is|looked|appeared|seemed)\b",
            right,
        )
    )
    head_pattern = "|".join(re.escape(head) for head in _ALL_HEADS)
    explicit_feature_assertion = bool(
        re.match(rf"(?:{head_pattern})\s+(?:is|are|was|were)\b", right)
    )
    return (
        audit_only(left)
        or audit_only(right)
        or lexical_absence
        or explicit_subject
        or explicit_feature_assertion
    )


def _family_contexts(clause: str, heads: Sequence[str]) -> list[str]:
    head_pattern = "|".join(re.escape(head) for head in sorted(heads, key=len, reverse=True))
    contexts = []
    for match in re.finditer(rf"\b(?:{head_pattern})\b", clause):
        before = clause[: match.start()]
        after = clause[match.end() :]
        before_tokens = list(re.finditer(r"[a-z]+(?:-[a-z]+)?", before))
        start = before_tokens[-6].start() if len(before_tokens) >= 6 else 0
        previous_heads = [
            token for token in before_tokens if token.group(0) in _ALL_HEADS
        ]
        if previous_heads:
            previous_head = previous_heads[-1]
            start = previous_head.end()
            tail = before[start:]
            conjunctions = list(re.finditer(r"\b(?:and|but|with)\b", tail))
            if conjunctions:
                start += conjunctions[-1].end()
        for token in reversed(before_tokens):
            if token.start() < start:
                break
            if token.group(0) in _BOUNDARY_WORDS or token.group(0) in _ALL_HEADS:
                start = token.end()
                break
        # Post-head adjectives are accepted only after an explicit copula
        # ("eyes were large").  Otherwise the next adjective usually belongs
        # to the following comma-separated feature ("large eyes, round face").
        copula = re.match(r"\s*(?:is|are|was|were|looked|appeared|seemed)\s+", after)
        if copula is None:
            end = 0
        else:
            after_tokens = list(re.finditer(r"[a-z]+(?:-[a-z]+)?", after))
            end = after_tokens[5].start() if len(after_tokens) > 5 else len(after)
            for token in after_tokens[1:]:
                if token.group(0) in _BOUNDARY_WORDS or token.group(0) in _ALL_HEADS:
                    end = token.start()
                    break
        context = f"{before[start:]} {match.group(0)} {after[:end]}".strip(" ,")
        contexts.append(re.sub(r"\s+", " ", context))
    # Age aliases sometimes act as both value and head; use the clause once.
    if not contexts and any(re.search(rf"\b{re.escape(head)}\b", clause) for head in heads):
        contexts.append(clause)
    return list(dict.fromkeys(contexts))


def _age_contexts(clause: str) -> list[str]:
    pattern = re.compile(
        r"\b(?:young|youthful|middle[- ]aged|older|elderly)\b|"
        r"\bold\s+(?:man|male|person|guy|gentleman)\b|"
        r"\bin his (?:twenties|forties|fifties)\b"
    )
    return [clause] if pattern.search(clause) else []


def _find_alias(context: str, aliases: Sequence[str]) -> str:
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", context):
            return alias
    return ""


def _scope_abstentions(text: str) -> list[Abstention]:
    rules = (
        (
            "unsupported_v1",
            "race",
            r"\b(?:white|black|asian|latino|hispanic)\s+(?:man|male|person|guy)\b|\bcaucasian\b",
        ),
        ("unsupported_v1", "gender", r"\b(?:male|man|guy|gentleman)\b"),
        ("unsupported_v1", "expression", r"\b(?:neutral expression|smiling|angry|happy|sad)\b"),
        ("not_observable_or_standardised", "clothing", r"\b(?:shirt|jacket|coat|hoodie|clothes|clothing|tie)\b"),
        ("not_observable_or_standardised", "voice", r"\b(?:voice|accent|spoke|speech)\b"),
        ("not_observable_or_standardised", "build", r"\b(?:build|built|height|muscular|stocky|slim body|thin body|heavyset|tall man|short man)\b"),
        ("not_observable_or_standardised", "teeth", r"\b(?:tooth|teeth)\b"),
        ("unsupported_v1", "skin", r"\b(?:skin|complexion|pale|tan|tanned)\b"),
        ("unsupported_v1", "accessory", r"\b(?:glasses|spectacles|hat|cap|earring)\b"),
        ("unsupported_v1", "distinctive_mark", r"\b(?:scar|scars|mark|marks)\b"),
    )
    output = []
    for reason, feature, pattern in rules:
        match = re.search(pattern, text)
        if match:
            output.append(Abstention(reason, feature, normalised_span=match.group(0)))
    return output
