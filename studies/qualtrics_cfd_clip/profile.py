from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scoring import SCORER_VERSION
from .description import (
    PARSER_VERSION,
    POOL_PROFILE,
    PROMPT_TEMPLATE_VERSION,
    STRICT_V1_REGISTRY,
)


SCORING_CONTRACT_VERSION = "cfd_clip_attribute_contrastive_v3"
DEFAULT_STUDY_VERSION = "cfd_clip_contrastive_l14_v3"
DEFAULT_PROFILE_PATH = Path(__file__).with_name("contrast_profile_live_v1.json")


@dataclass(frozen=True)
class LiveSelectionProfile:
    """Frozen, preregistration-facing contract for live filler selection."""

    profile_id: str
    profile_version: str
    profile_status: str
    study_version: str
    scoring_method: str
    score_space: str
    aggregation_version: str
    standardise_components: bool
    zero_pair_fallback: str
    schema_name: str
    schema_version: str
    schema_sha256: str
    parser_version: str
    prompt_template_version: str
    scorer_version: str
    pool_profile: str
    pool_version: str
    model_name: str
    model_revision: str
    model_snapshot_sha256: str
    index_sha256: str
    candidate_embedding_sha256: str
    pool_sha256: str
    media_manifest_sha256: str
    approved_rule_ids: tuple[str, ...]
    profile_sha256: str

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PROFILE_PATH) -> "LiveSelectionProfile":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The live selection profile must be a JSON object.")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        values = dict(payload)
        values["approved_rule_ids"] = tuple(values.get("approved_rule_ids", ()))
        profile = cls(**values, profile_sha256=digest)
        profile.validate_static_contract()
        return profile

    def validate_static_contract(self) -> None:
        if self.scoring_method != "attribute_contrastive":
            raise ValueError("The live profile must use attribute_contrastive scoring.")
        if self.score_space != "within_pool_zscore_equal_dimension_mean":
            raise ValueError("The live profile has an unsupported score space.")
        if self.aggregation_version != "within_pool_z_equal_dimension_mean_v1":
            raise ValueError("The live profile has an unsupported aggregation version.")
        if self.zero_pair_fallback != "raw_positive_chunked_v1":
            raise ValueError("The live profile has an unsupported zero-pair fallback.")
        if not self.standardise_components:
            raise ValueError("The live profile must standardise contrast components.")
        if not self.approved_rule_ids or len(self.approved_rule_ids) != len(
            set(self.approved_rule_ids)
        ):
            raise ValueError("The live profile requires unique, non-empty approved_rule_ids.")

        expected: dict[str, Any] = {
            "schema_name": STRICT_V1_REGISTRY.name,
            "schema_version": STRICT_V1_REGISTRY.version,
            "schema_sha256": STRICT_V1_REGISTRY.hash,
            "parser_version": PARSER_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "scorer_version": SCORER_VERSION,
            "pool_profile": POOL_PROFILE,
        }
        for field_name, expected_value in expected.items():
            actual = getattr(self, field_name)
            if actual != expected_value:
                raise ValueError(
                    f"Live profile {field_name} does not match the installed parser/scorer: "
                    f"{actual!r} != {expected_value!r}."
                )

        known_rule_ids = {rule.rule_id for rule in STRICT_V1_REGISTRY.rules}
        unknown = sorted(set(self.approved_rule_ids) - known_rule_ids)
        if unknown:
            raise ValueError(f"Live profile contains unknown approved rules: {unknown}")
        for field_name in (
            "profile_id",
            "profile_version",
            "profile_status",
            "study_version",
            "model_name",
            "model_revision",
            "model_snapshot_sha256",
            "index_sha256",
            "candidate_embedding_sha256",
            "pool_sha256",
            "media_manifest_sha256",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Live profile {field_name} must not be empty.")
        if not re.fullmatch(r"[0-9a-f]{40}", self.model_revision):
            raise ValueError("Live profile model_revision must be a 40-character commit SHA.")
        for field_name in (
            "model_snapshot_sha256",
            "index_sha256",
            "candidate_embedding_sha256",
            "pool_sha256",
            "media_manifest_sha256",
            "schema_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(self, field_name))):
                raise ValueError(f"Live profile {field_name} must be a lowercase SHA-256.")

    def contract_sha256(self) -> str:
        payload = {
            "scoring_contract_version": SCORING_CONTRACT_VERSION,
            "profile_sha256": self.profile_sha256,
            "study_version": self.study_version,
            "scoring_method": self.scoring_method,
            "score_space": self.score_space,
            "aggregation_version": self.aggregation_version,
            "standardise_components": self.standardise_components,
            "zero_pair_fallback": self.zero_pair_fallback,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_PROFILE_PATH",
    "DEFAULT_STUDY_VERSION",
    "LiveSelectionProfile",
    "SCORING_CONTRACT_VERSION",
]
