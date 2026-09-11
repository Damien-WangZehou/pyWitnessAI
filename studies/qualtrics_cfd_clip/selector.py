from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import numpy as np

from pyWitnessAI.clip import ClipIndex, SentenceTransformerClipEncoder

from .description import (
    Abstention,
    ContrastPlan,
    EyewitnessContrastPromptBuilder,
    PARSER_VERSION,
    PROMPT_TEMPLATE,
    PROMPT_TEMPLATE_VERSION,
    STRICT_V1_REGISTRY,
)
from .media import file_sha256, media_manifest_integrity
from .profile import (
    DEFAULT_PROFILE_PATH,
    LiveSelectionProfile,
    SCORING_CONTRACT_VERSION,
)
from .scoring import ContrastiveClipScorer


class TextEncoder(Protocol):
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class SelectedFace:
    position: int
    image_id: str
    target_id: str
    selector_rank: int
    selector_score: float

    @property
    def clip_rank(self) -> int:
        """Compatibility alias; contrastive ranks are not raw CLIP ranks."""
        return self.selector_rank

    @property
    def clip_score(self) -> float:
        """Compatibility alias; contrastive scores are not cosine values."""
        return self.selector_score


@dataclass(frozen=True)
class LineupResult:
    lineup_id: str
    model_name: str
    index_sha256: str
    pool_sha256: str
    query_sha256: str
    effective_query_sha256: str
    query_strategy: str
    query_chunk_count: int
    query_token_count: int
    model_context_tokens: int
    prompt_version: str
    pool_version: str
    selection_algorithm: str
    pool_size: int
    study_version: str
    model_revision: str
    model_snapshot_sha256: str
    model_snapshot_match_verified: bool
    candidate_embedding_sha256: str
    media_manifest_sha256: str
    image_text_revision_match_verified: bool
    query_hash_kind: str
    prompt_set_sha256: str
    prompt_count: int
    scoring_contract_version: str
    scoring_contract_sha256: str
    selection_profile_id: str
    selection_profile_version: str
    selection_profile_sha256: str
    profile_status: str
    scoring_method: str
    effective_method: str
    score_space: str
    aggregation_version: str
    scorer_version: str
    plan_status: str
    contrast_plan_sha256: str
    schema_name: str
    schema_version: str
    schema_sha256: str
    parser_version: str
    prompt_template_version: str
    prompt_template_sha256: str
    contrast_pair_count: int
    contrast_dimension_count: int
    abstention_count: int
    abstention_reason_codes: tuple[str, ...]
    accepted_rule_ids: tuple[str, ...]
    accepted_dimensions: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str
    standardise_components: bool
    pool_profile: str
    items: tuple[SelectedFace, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["items"] = []
        for item in self.items:
            record = asdict(item)
            # Keep one transition release of the old names for imported QSFs.
            record["clip_rank"] = item.selector_rank
            record["clip_score"] = item.selector_score
            result["items"].append(record)
        return result


class CfdWhiteMaleLineupSelector:
    """Select a deterministic, position-randomised all-filler CFD lineup.

    Retrieval is performed only inside the eligible CFD pool. This is not the
    same as searching the full index and filtering the returned top-k rows.
    """

    def __init__(
        self,
        index: ClipIndex,
        encoder: TextEncoder,
        *,
        signing_secret: str,
        repository_root: str | Path,
        selection_profile: LiveSelectionProfile,
        image_root: str | Path | None = None,
        lineup_size: int = 6,
        excluded_target_ids: tuple[str, ...] | list[str] | set[str] = (),
    ) -> None:
        if not signing_secret or len(signing_secret) < 16:
            raise ValueError("signing_secret must contain at least 16 characters.")
        if lineup_size < 1:
            raise ValueError("lineup_size must be at least 1.")

        self.index = index
        self.encoder = encoder
        self.selection_profile = selection_profile
        self.signing_secret = signing_secret.encode("utf-8")
        self.repository_root = Path(repository_root).resolve()
        self.image_root = (
            Path(image_root).resolve()
            if image_root is not None
            else (self.repository_root / "data" / "CFD Version 3.0" / "Images").resolve()
        )
        self.lineup_size = int(lineup_size)
        self.prompt_version = PROMPT_TEMPLATE_VERSION
        self.pool_version = selection_profile.pool_version
        self.selection_algorithm = "contrast_profile_rank_distinct_then_seeded_shuffle_v4"
        self.prompt_builder = EyewitnessContrastPromptBuilder(
            schema=STRICT_V1_REGISTRY,
            policy="strict_v1",
        )

        manifest = self.index.manifest
        required = {"gender", "race", "expression", "image_id", "target_id", "image_path"}
        missing = sorted(required - set(manifest.columns))
        if missing:
            raise ValueError(f"The CLIP manifest is missing CFD pool columns: {missing}")

        excluded_targets = {str(value).strip() for value in excluded_target_ids if str(value).strip()}
        eligible = (
            manifest["gender"].astype(str).str.strip().str.casefold().eq("male")
            & manifest["race"].astype(str).str.strip().str.casefold().eq("white")
            & manifest["expression"].astype(str).str.strip().str.casefold().eq("n")
            & ~manifest["target_id"].astype(str).isin(excluded_targets)
        )
        self.pool_indices = np.flatnonzero(eligible.to_numpy())
        self.pool_manifest = manifest.iloc[self.pool_indices].reset_index(drop=True).copy()
        self.pool_embeddings = self.index.image_embeddings[self.pool_indices]
        if len(self.pool_manifest) < self.lineup_size:
            raise ValueError(
                f"Eligible CFD White Male pool has {len(self.pool_manifest)} rows, "
                f"fewer than lineup_size={self.lineup_size}."
            )

        duplicate_ids = self.pool_manifest["image_id"].astype(str).duplicated()
        if duplicate_ids.any():
            raise ValueError("Eligible CFD pool contains duplicate image_id values.")
        duplicate_targets = self.pool_manifest["target_id"].astype(str).duplicated()
        if duplicate_targets.any():
            raise ValueError(
                "Eligible CFD pool contains duplicate target_id values; within-pool "
                "standardisation requires one candidate per identity."
            )

        pool_rows = [
            f"{record.image_id}\t{record.target_id}"
            for record in self.pool_manifest[["image_id", "target_id"]].itertuples(index=False)
        ]
        self.pool_sha256 = hashlib.sha256("\n".join(pool_rows).encode("utf-8")).hexdigest()
        self.candidate_embedding_sha256 = hashlib.sha256(
            np.ascontiguousarray(self.pool_embeddings, dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        (
            self.media_manifest_sha256,
            self._image_paths,
            self._image_content_sha256,
        ) = media_manifest_integrity(
            self.pool_manifest,
            repository_root=self.repository_root,
            image_root=self.image_root,
        )

        pool_metadata = dict(self.index.metadata)
        pool_metadata["embedding_sha256"] = self.candidate_embedding_sha256
        pool_metadata["image_count"] = int(len(self.pool_manifest))
        self.pool_index = ClipIndex(
            self.pool_embeddings,
            self.pool_manifest,
            metadata=pool_metadata,
        )
        self.scorer = ContrastiveClipScorer(
            self.pool_index,
            self.encoder,
            standardise_components=selection_profile.standardise_components,
            show_progress=False,
        )
        self._approved_rule_ids = frozenset(selection_profile.approved_rule_ids)

        self._validate_live_profile()

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        signing_secret: str,
        repository_root: str | Path,
        image_root: str | Path | None = None,
        model_name: str | None = None,
        model_revision: str | None = None,
        local_model_path: str | Path | None = None,
        selection_profile_path: str | Path = DEFAULT_PROFILE_PATH,
        device: str | None = None,
        **kwargs: Any,
    ) -> "CfdWhiteMaleLineupSelector":
        selection_profile = LiveSelectionProfile.load(selection_profile_path)
        index = ClipIndex.load(index_dir)
        cached_model = str(index.metadata.get("model_name") or "")
        cached_revision = str(index.metadata.get("model_revision") or "")
        if not cached_model or not cached_revision:
            raise ValueError(
                "The live CLIP index must record both model_name and a frozen model_revision."
            )
        if model_name is not None and model_name != cached_model:
            raise ValueError(
                "CFD_CLIP_MODEL must match the model used to build the cached index: "
                f"{model_name!r} != {cached_model!r}."
            )
        if model_revision is not None and model_revision != cached_revision:
            raise ValueError(
                "CFD_CLIP_MODEL_REVISION must match the cached image index: "
                f"{model_revision!r} != {cached_revision!r}."
            )
        configured_model = model_name or cached_model
        configured_revision = model_revision or cached_revision
        if local_model_path is None:
            raise ValueError(
                "The live CLIP service requires CFD_CLIP_LOCAL_MODEL_PATH so the exact "
                "model snapshot contents can be verified against the selection profile."
            )
        encoder = SentenceTransformerClipEncoder(
            configured_model,
            device=device,
            model_revision=configured_revision,
            local_model_path=local_model_path,
        )
        selector = cls(
            index,
            encoder,
            signing_secret=signing_secret,
            repository_root=repository_root,
            selection_profile=selection_profile,
            image_root=image_root,
            **kwargs,
        )
        # Warm the text path during service startup rather than during a participant request.
        encoder.encode_texts([PROMPT_TEMPLATE.format(descriptor="large eyes")], show_progress=False)
        return selector

    @property
    def model_name(self) -> str:
        return str(self.index.metadata.get("model_name") or getattr(self.encoder, "model_name", ""))

    @property
    def index_sha256(self) -> str:
        return str(self.index.metadata.get("embedding_sha256") or "")

    @property
    def model_revision(self) -> str:
        return str(
            self.index.metadata.get("model_revision")
            or getattr(self.encoder, "model_revision", "")
            or ""
        )

    @property
    def scoring_contract_sha256(self) -> str:
        return self.selection_profile.contract_sha256()

    @property
    def model_snapshot_sha256(self) -> str:
        return str(getattr(self.encoder, "model_snapshot_sha256", "") or "")

    @property
    def model_snapshot_match_verified(self) -> bool:
        return self._model_snapshot_match_verified()

    def select(
        self,
        description: str,
        participant_id: str,
        *,
        study_version: str | None = None,
    ) -> LineupResult:
        clean_description = " ".join(str(description).split())
        if not clean_description:
            raise ValueError("description must not be empty.")
        if len(clean_description) > 2000:
            raise ValueError("description must contain at most 2000 characters.")
        clean_participant_id = str(participant_id).strip()
        if not clean_participant_id:
            raise ValueError("participant_id must not be empty.")
        if len(clean_participant_id) > 128:
            raise ValueError("participant_id must contain at most 128 characters.")
        active_study_version = str(study_version or self.selection_profile.study_version)
        if active_study_version != self.selection_profile.study_version:
            raise ValueError(
                "study_version does not match the deployed selection profile: "
                f"{active_study_version!r} != {self.selection_profile.study_version!r}."
            )

        full_plan = self.prompt_builder.build(clean_description)
        plan = self.build_selection_plan(full_plan)
        accepted_rule_ids = tuple(sorted(pair.pair_id for pair in plan.pairs))
        accepted_dimensions = tuple(
            sorted({f"{pair.family}.{pair.dimension}" for pair in plan.pairs})
        )
        abstention_reason_codes = tuple(
            sorted({item.reason_code for item in plan.abstentions})
        )
        plan_payload = {
            "accepted_rule_ids": accepted_rule_ids,
            "accepted_dimensions": accepted_dimensions,
            "abstention_reason_codes": abstention_reason_codes,
            "profile_sha256": self.selection_profile.profile_sha256,
            "schema_sha256": self.selection_profile.schema_sha256,
            "parser_version": PARSER_VERSION,
        }
        contrast_plan_sha256 = self._private_fingerprint("contrast-plan", plan_payload)
        query_sha256 = self._private_fingerprint("description", clean_description)

        try:
            if plan.pairs:
                effective_method = self.selection_profile.scoring_method
                fallback_used = False
                fallback_reason = ""
                score_result = self.scorer.score(
                    [plan],
                    methods=(self.selection_profile.scoring_method,),
                )
                prompt_inputs = tuple(
                    sorted(
                        {
                            text
                            for pair in plan.pairs
                            for text in (*pair.positive_texts, *pair.negative_texts)
                        }
                    )
                )
                query_strategy = self.selection_profile.aggregation_version
                score_space = self.selection_profile.score_space
                prompt_inputs, query_token_count, model_context_tokens = (
                    self._audit_structured_prompts(prompt_inputs)
                )
            else:
                effective_method = "raw_positive"
                fallback_used = True
                fallback_reason = "zero_approved_pairs"
                score_result = self.scorer.score(
                    [plan],
                    raw_descriptions=[clean_description],
                    methods=("raw_positive",),
                )
                prompt_inputs, query_token_count, model_context_tokens = self._raw_chunks(
                    clean_description
                )
                query_strategy = "raw_positive_token_weighted_chunks_within_pool_zscore_v1"
                score_space = "within_pool_zscore_raw_positive_cosine"
        except (ValueError, KeyError, FloatingPointError) as exc:
            # Invalid participant input is rejected before this point.  Any
            # scoring failure is therefore a technical failure and must never
            # silently change the assigned retrieval method.
            raise RuntimeError("CLIP contrastive scoring failed.") from exc

        ranked = score_result.ranked
        if ranked.empty or len(ranked) != len(self.pool_manifest):
            raise RuntimeError("The CLIP scorer did not return a complete candidate ranking.")
        if not np.isfinite(ranked["selector_score"].to_numpy(dtype=np.float32)).all():
            raise RuntimeError("The CLIP scorer returned non-finite candidate scores.")
        effective_query_sha256 = self._private_fingerprint(
            "effective-query", list(prompt_inputs)
        )
        prompt_set_sha256 = self._private_fingerprint("prompt-set", list(prompt_inputs))

        selected: list[SelectedFace] = []
        seen_target_ids: set[str] = set()
        for record in ranked.itertuples(index=False):
            target_id = str(record.target_id)
            if target_id in seen_target_ids:
                continue
            seen_target_ids.add(target_id)
            selected.append(
                SelectedFace(
                    position=0,
                    image_id=str(record.image_id),
                    target_id=target_id,
                    selector_rank=int(record.rank),
                    selector_score=float(record.selector_score),
                )
            )
            if len(selected) == self.lineup_size:
                break
        if len(selected) != self.lineup_size:
            raise RuntimeError("The eligible CFD pool did not contain enough distinct identities.")

        seed_message = "\0".join(
            [
                clean_participant_id,
                query_sha256,
                self.index_sha256,
                self.pool_sha256,
                self.media_manifest_sha256,
                effective_query_sha256,
                self.prompt_version,
                active_study_version,
                query_strategy,
                self.scoring_contract_sha256,
                self.selection_profile.profile_sha256,
                self.model_snapshot_sha256,
                contrast_plan_sha256,
                effective_method,
                score_space,
                self.selection_algorithm,
                str(self.lineup_size),
            ]
        )
        seed_digest = hmac.new(
            self.signing_secret,
            seed_message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        rng = random.Random(int.from_bytes(seed_digest[:8], byteorder="big", signed=False))
        rng.shuffle(selected)
        positioned = tuple(
            SelectedFace(
                position=position,
                image_id=item.image_id,
                target_id=item.target_id,
                selector_rank=item.selector_rank,
                selector_score=item.selector_score,
            )
            for position, item in enumerate(selected, start=1)
        )
        ordered_members = "|".join(
            f"{item.position}:{item.image_id}:{item.target_id}:{item.selector_rank}"
            for item in positioned
        )
        lineup_id = hmac.new(
            self.signing_secret,
            f"{seed_message}\0{ordered_members}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]
        return LineupResult(
            lineup_id=lineup_id,
            model_name=self.model_name,
            index_sha256=self.index_sha256,
            pool_sha256=self.pool_sha256,
            query_sha256=query_sha256,
            effective_query_sha256=effective_query_sha256,
            query_strategy=query_strategy,
            query_chunk_count=len(prompt_inputs),
            query_token_count=int(query_token_count),
            model_context_tokens=int(model_context_tokens),
            prompt_version=self.prompt_version,
            pool_version=self.pool_version,
            selection_algorithm=self.selection_algorithm,
            pool_size=len(self.pool_manifest),
            study_version=active_study_version,
            model_revision=self.model_revision,
            model_snapshot_sha256=self.model_snapshot_sha256,
            model_snapshot_match_verified=self._model_snapshot_match_verified(),
            candidate_embedding_sha256=self.candidate_embedding_sha256,
            media_manifest_sha256=self.media_manifest_sha256,
            image_text_revision_match_verified=self._revision_match_verified(),
            query_hash_kind="hmac_sha256_service_secret_v1",
            prompt_set_sha256=prompt_set_sha256,
            prompt_count=len(prompt_inputs),
            scoring_contract_version=SCORING_CONTRACT_VERSION,
            scoring_contract_sha256=self.scoring_contract_sha256,
            selection_profile_id=self.selection_profile.profile_id,
            selection_profile_version=self.selection_profile.profile_version,
            selection_profile_sha256=self.selection_profile.profile_sha256,
            profile_status=self.selection_profile.profile_status,
            scoring_method=self.selection_profile.scoring_method,
            effective_method=effective_method,
            score_space=score_space,
            aggregation_version=self.selection_profile.aggregation_version,
            scorer_version=self.selection_profile.scorer_version,
            plan_status=plan.status,
            contrast_plan_sha256=contrast_plan_sha256,
            schema_name=self.selection_profile.schema_name,
            schema_version=self.selection_profile.schema_version,
            schema_sha256=self.selection_profile.schema_sha256,
            parser_version=self.selection_profile.parser_version,
            prompt_template_version=self.selection_profile.prompt_template_version,
            prompt_template_sha256=str(plan.provenance["prompt_template_sha256"]),
            contrast_pair_count=len(plan.pairs),
            contrast_dimension_count=len(accepted_dimensions),
            abstention_count=len(plan.abstentions),
            abstention_reason_codes=abstention_reason_codes,
            accepted_rule_ids=accepted_rule_ids,
            accepted_dimensions=accepted_dimensions,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            standardise_components=self.selection_profile.standardise_components,
            pool_profile=self.selection_profile.pool_profile,
            items=positioned,
        )

    def build_selection_plan(self, plan: ContrastPlan) -> ContrastPlan:
        """Apply the frozen live allowlist without retaining participant text spans."""
        pairs = tuple(pair for pair in plan.pairs if pair.pair_id in self._approved_rule_ids)
        approved_evidence_ids = {
            evidence_id for pair in pairs for evidence_id in pair.evidence_ids
        }
        evidence = tuple(
            item for item in plan.evidence if item.evidence_id in approved_evidence_ids
        )
        removed = tuple(pair for pair in plan.pairs if pair.pair_id not in self._approved_rule_ids)
        abstentions = tuple(plan.abstentions) + tuple(
            Abstention(
                reason_code="rule_not_approved_for_live_profile",
                feature=pair.family,
                dimension=pair.dimension,
                rule_id=pair.pair_id,
            )
            for pair in removed
        )
        status = "abstain" if not pairs else "partial" if abstentions else "ready"
        return ContrastPlan(
            status=status,
            pairs=pairs,
            abstentions=abstentions,
            evidence=evidence,
            provenance=plan.provenance,
        )

    def _raw_chunks(self, text: str) -> tuple[tuple[str, ...], int, int]:
        prepare_chunks = getattr(self.encoder, "prepare_text_chunks", None)
        if not callable(prepare_chunks):
            return (text,), 0, 0
        chunks, weights, token_count, context_tokens = prepare_chunks(text)
        if not chunks or len(chunks) != len(weights):
            raise RuntimeError("The text encoder returned invalid raw-description chunks.")
        return tuple(chunks), int(token_count), int(context_tokens)

    def _audit_structured_prompts(
        self,
        prompts: tuple[str, ...],
    ) -> tuple[tuple[str, ...], int, int]:
        if not prompts:
            raise RuntimeError("Contrastive scoring requires at least one generated prompt.")
        prepare_chunks = getattr(self.encoder, "prepare_text_chunks", None)
        if not callable(prepare_chunks):
            return prompts, 0, 0
        total_tokens = 0
        contexts: set[int] = set()
        for prompt in prompts:
            chunks, weights, token_count, context_tokens = prepare_chunks(prompt)
            if len(chunks) != 1 or len(weights) != 1:
                raise RuntimeError(
                    "A frozen contrast prompt exceeded the CLIP context window; "
                    "the live profile must be revised before use."
                )
            total_tokens += int(token_count)
            contexts.add(int(context_tokens))
        if len(contexts) != 1:
            raise RuntimeError("The text encoder returned inconsistent context-window sizes.")
        return prompts, total_tokens, contexts.pop()

    def _private_fingerprint(self, label: str, value: Any) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hmac.new(
            self.signing_secret,
            f"{label}\0{payload}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _revision_match_verified(self) -> bool:
        index_revision = str(self.index.metadata.get("model_revision") or "")
        encoder_revision = str(getattr(self.encoder, "model_revision", "") or "")
        return bool(index_revision and encoder_revision and index_revision == encoder_revision)

    def _model_snapshot_match_verified(self) -> bool:
        index_snapshot = str(self.index.metadata.get("model_snapshot_sha256") or "")
        encoder_snapshot = self.model_snapshot_sha256
        expected_snapshot = self.selection_profile.model_snapshot_sha256
        return bool(
            index_snapshot
            and encoder_snapshot
            and index_snapshot == encoder_snapshot == expected_snapshot
        )

    def _validate_live_profile(self) -> None:
        self.selection_profile.validate_static_contract()
        checks = {
            "model_name": (self.selection_profile.model_name, self.model_name),
            "model_revision": (self.selection_profile.model_revision, self.model_revision),
            "model_snapshot_sha256": (
                self.selection_profile.model_snapshot_sha256,
                self.model_snapshot_sha256,
            ),
            "index_sha256": (self.selection_profile.index_sha256, self.index_sha256),
            "candidate_embedding_sha256": (
                self.selection_profile.candidate_embedding_sha256,
                self.candidate_embedding_sha256,
            ),
            "pool_sha256": (self.selection_profile.pool_sha256, self.pool_sha256),
            "media_manifest_sha256": (
                self.selection_profile.media_manifest_sha256,
                self.media_manifest_sha256,
            ),
        }
        for name, (expected, actual) in checks.items():
            if str(expected) != str(actual):
                raise ValueError(
                    f"Live selection profile {name} does not match the loaded index/pool: "
                    f"{expected!r} != {actual!r}."
                )
        encoder_model = str(getattr(self.encoder, "model_name", "") or "")
        encoder_revision = str(getattr(self.encoder, "model_revision", "") or "")
        if encoder_model and encoder_model != self.model_name:
            raise ValueError("The text encoder model does not match the image index model.")
        if not encoder_revision or not self._revision_match_verified():
            raise ValueError(
                "The text encoder and image index must use the same frozen model revision."
            )
        index_snapshot = str(self.index.metadata.get("model_snapshot_sha256") or "")
        if not self._model_snapshot_match_verified():
            raise ValueError(
                "The text encoder, image index, and live profile must use the same "
                "verified model snapshot contents; "
                f"index={index_snapshot!r}, encoder={self.model_snapshot_sha256!r}."
            )
        index_media = str(self.index.metadata.get("media_manifest_sha256") or "")
        if not index_media or index_media != self.media_manifest_sha256:
            raise ValueError(
                "The displayed CFD media do not match the manifest/content digest "
                "recorded by the image index."
            )

    def create_image_token(self, image_id: str, expires: int) -> str:
        if image_id not in self._image_paths:
            raise KeyError("Unknown or ineligible image_id.")
        message = "\n".join(
            (
                image_id,
                str(int(expires)),
                self.selection_profile.profile_sha256,
                self.media_manifest_sha256,
            )
        ).encode("utf-8")
        return hmac.new(self.signing_secret, message, hashlib.sha256).hexdigest()

    def verify_image_token(
        self,
        image_id: str,
        expires: int,
        signature: str,
        *,
        now: int | None = None,
    ) -> bool:
        current_time = int(time.time()) if now is None else int(now)
        if int(expires) < current_time or image_id not in self._image_paths:
            return False
        expected = self.create_image_token(image_id, int(expires))
        return hmac.compare_digest(expected, str(signature))

    def signed_image_path(self, image_id: str, expires: int) -> str:
        signature = self.create_image_token(image_id, expires)
        return (
            f"/v1/images/{quote(image_id, safe='')}?expires={int(expires)}"
            f"&signature={signature}"
        )

    def image_path(self, image_id: str) -> Path:
        try:
            path = self._image_paths[image_id]
        except KeyError as exc:
            raise KeyError("Unknown or ineligible image_id.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"CFD image is missing: {path}")
        expected_digest = self._image_content_sha256[image_id]
        if not hmac.compare_digest(file_sha256(path), expected_digest):
            raise RuntimeError(
                "CFD image contents changed after live-profile verification; "
                "the service will not display unverified media."
            )
        return path

    def image_bytes(self, image_id: str) -> tuple[str, bytes]:
        """Return the exact bytes that were integrity-checked for this response."""

        try:
            path = self._image_paths[image_id]
        except KeyError as exc:
            raise KeyError("Unknown or ineligible image_id.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"CFD image is missing: {path}")
        payload = path.read_bytes()
        expected_digest = self._image_content_sha256[image_id]
        actual_digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise RuntimeError(
                "CFD image contents changed after live-profile verification; "
                "the service will not display unverified media."
            )
        return path.name, payload
