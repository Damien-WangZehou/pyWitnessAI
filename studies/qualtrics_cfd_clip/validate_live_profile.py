"""Controlled-query validation for the frozen Qualtrics CFD profile."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "studies.qualtrics_cfd_clip"


import numpy as np
import pandas as pd

from .selector import CfdWhiteMaleLineupSelector
from .profile import DEFAULT_PROFILE_PATH
from .description import ContrastPlan, ContrastPromptPair, STRICT_V1_REGISTRY

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser(
    *, repository_root: str | Path | None = None,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
) -> argparse.ArgumentParser:
    root = Path(repository_root) if repository_root is not None else PROJECT_ROOT
    parser = argparse.ArgumentParser(
        description="Validate the frozen Qualtrics live profile on controlled CFD queries."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=root / "build" / "qualtrics_cfd_clip" / "clip_index_vit_l_14",
    )
    parser.add_argument(
        "--queries", type=Path, required=True,
        help="Controlled-query CSV with query_id, target_id, description, gender, race and stage.",
    )
    parser.add_argument("--profile", type=Path, default=Path(profile_path))
    parser.add_argument("--local-model-path", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "build" / "qualtrics_cfd_clip" / "l14_profile_validation",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *, repository_root: str | Path | None = None,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
) -> int:
    root = Path(repository_root) if repository_root is not None else PROJECT_ROOT
    args = build_parser(repository_root=root, profile_path=profile_path).parse_args(argv)
    selector = CfdWhiteMaleLineupSelector.load(
        args.index_dir,
        signing_secret="offline-profile-validation-secret-v1",
        repository_root=root,
        selection_profile_path=args.profile,
        local_model_path=args.local_model_path,
        device=args.device,
    )
    queries = pd.read_csv(args.queries, low_memory=False)
    per_query, direction_validation, summary = validate_profile(selector, queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(args.output_dir / "per_query_ranks.csv", index=False)
    direction_validation.to_csv(
        args.output_dir / "approved_rule_direction_validation.csv", index=False
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def validate_profile(
    selector: CfdWhiteMaleLineupSelector, queries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return target ranks, rule direction checks, and the existing audit summary."""
    controlled = queries.loc[
        queries["gender"].eq("male")
        & queries["race"].eq("White")
        & queries["stage"].eq("15_add_teeth")
    ].reset_index(drop=True)
    if len(controlled) != len(selector.pool_manifest):
        raise ValueError(
            "Controlled-query count must equal the frozen candidate-pool size: "
            f"{len(controlled)} != {len(selector.pool_manifest)}."
        )
    if set(controlled["target_id"].astype(str)) != set(
        selector.pool_manifest["target_id"].astype(str)
    ):
        raise ValueError("Controlled-query targets do not match the live candidate pool.")

    plans = [
        selector.build_selection_plan(selector.prompt_builder.build(description))
        for description in controlled["description"].astype(str)
    ]
    usable_indices = [index for index, plan in enumerate(plans) if plan.pairs]
    raw = selector.scorer.score(
        plans,
        raw_descriptions=controlled["description"].astype(str).tolist(),
        methods=("raw_positive",),
    ).ranked
    usable_plans = [plans[index] for index in usable_indices]
    structured = selector.scorer.score(
        usable_plans,
        methods=("structured_positive", "attribute_contrastive"),
    ).ranked
    query_map = dict(enumerate(usable_indices))
    structured["query_index"] = structured["query_index"].map(query_map)

    raw_ranks = _target_ranks(raw, controlled)
    positive_ranks = _target_ranks(
        structured.loc[structured["method"].eq("structured_positive")], controlled
    )
    contrast_ranks = _target_ranks(
        structured.loc[structured["method"].eq("attribute_contrastive")], controlled
    )
    live_ranks = raw_ranks.copy()
    live_ranks.loc[contrast_ranks.index] = contrast_ranks
    direction_validation = _direction_validation(selector)

    per_query = pd.DataFrame(
        {
            "query_index": np.arange(len(controlled), dtype=int),
            "query_id": controlled["query_id"].astype(str),
            "target_id": controlled["target_id"].astype(str),
            "approved_pair_count": [len(plan.pairs) for plan in plans],
            "approved_dimension_count": [
                len({(pair.family, pair.dimension) for pair in plan.pairs}) for plan in plans
            ],
            "effective_method": [
                "attribute_contrastive" if plan.pairs else "raw_positive" for plan in plans
            ],
            "raw_positive_rank": raw_ranks.to_numpy(dtype=int),
            "structured_positive_rank": [
                int(positive_ranks.get(index, raw_ranks.loc[index]))
                for index in range(len(controlled))
            ],
            "attribute_contrastive_rank": [
                int(contrast_ranks[index]) if index in contrast_ranks.index else np.nan
                for index in range(len(controlled))
            ],
            "live_effective_rank": live_ranks.to_numpy(dtype=int),
        }
    )

    raw_top6 = raw_ranks.le(6).to_numpy()
    live_top6 = live_ranks.le(6).to_numpy()
    gains = int((live_top6 & ~raw_top6).sum())
    losses = int((~live_top6 & raw_top6).sum())
    structured_top6 = positive_ranks.le(6).to_numpy()
    contrast_top6 = contrast_ranks.le(6).to_numpy()
    matched_gains = int((contrast_top6 & ~structured_top6).sum())
    matched_losses = int((~contrast_top6 & structured_top6).sum())
    summary = {
        "interpretation": (
            "Exploratory controlled-query validation only; metadata-derived descriptions are "
            "not independent eyewitness reports and do not establish production accuracy."
        ),
        "controlled_n_targets": int(len(controlled)),
        "approved_plan_n": int(len(usable_indices)),
        "raw_fallback_n": int(len(controlled) - len(usable_indices)),
        "chance_recall_at_6": 6 / len(controlled),
        "raw_positive": _metrics(raw_ranks),
        "live_effective": _metrics(live_ranks),
        "live_minus_raw_recall_at_6": float(live_top6.mean() - raw_top6.mean()),
        "live_minus_raw_mean_rank_improvement": float(
            (raw_ranks - live_ranks).mean()
        ),
        "live_vs_raw_top6_gains": gains,
        "live_vs_raw_top6_losses": losses,
        "live_vs_raw_exact_mcnemar_p": _exact_mcnemar(gains, losses),
        "usable_structured_positive": _metrics(positive_ranks),
        "usable_attribute_contrastive": _metrics(contrast_ranks),
        "matched_contrast_minus_structured_recall_at_6": float(
            contrast_top6.mean() - structured_top6.mean()
        ),
        "matched_contrast_minus_structured_mean_rank_improvement": float(
            (positive_ranks - contrast_ranks).mean()
        ),
        "matched_contrast_vs_structured_top6_gains": matched_gains,
        "matched_contrast_vs_structured_top6_losses": matched_losses,
        "matched_contrast_vs_structured_exact_mcnemar_p": _exact_mcnemar(
            matched_gains, matched_losses
        ),
        "model_name": selector.model_name,
        "model_revision": selector.model_revision,
        "model_snapshot_sha256": selector.model_snapshot_sha256,
        "model_snapshot_match_verified": selector.model_snapshot_match_verified,
        "index_sha256": selector.index_sha256,
        "candidate_embedding_sha256": selector.candidate_embedding_sha256,
        "pool_sha256": selector.pool_sha256,
        "media_manifest_sha256": selector.media_manifest_sha256,
        "selection_profile_id": selector.selection_profile.profile_id,
        "selection_profile_sha256": selector.selection_profile.profile_sha256,
        "scoring_contract_sha256": selector.scoring_contract_sha256,
        "approved_rule_ids": list(selector.selection_profile.approved_rule_ids),
        "directional_rho_min": float(direction_validation["directional_rho"].min()),
        "directional_rho_mean": float(direction_validation["directional_rho"].mean()),
        "directional_rho_all_positive": bool(
            direction_validation["directional_rho"].gt(0).all()
        ),
    }

    return per_query, direction_validation, summary


def _target_ranks(ranked: pd.DataFrame, controlled: pd.DataFrame) -> pd.Series:
    expected = controlled["target_id"].astype(str).to_dict()
    target_rows = ranked.loc[
        ranked.apply(
            lambda row: str(row["target_id"]) == expected[int(row["query_index"])],
            axis=1,
        ),
        ["query_index", "rank"],
    ]
    output = target_rows.set_index("query_index")["rank"].astype(int).sort_index()
    return output


def _metrics(ranks: pd.Series) -> dict[str, float]:
    values = ranks.to_numpy(dtype=float)
    return {
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_6": float(np.mean(values <= 6)),
        "recall_at_10": float(np.mean(values <= 10)),
        "mean_reciprocal_rank": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
    }


def _exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains + losses)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _direction_validation(selector: CfdWhiteMaleLineupSelector) -> pd.DataFrame:
    axis_specs = {
        ("eyes", "size"): ("EyeSize", {"large": 1, "small": -1}),
        ("nose", "length"): ("NoseLength", {"long": 1, "short": -1}),
        ("nose", "width"): ("NoseWidth", {"broad": 1, "narrow": -1}),
        ("lips", "fullness"): ("LipFullness", {"full": 1, "thin": -1}),
        ("chin", "length"): ("ChinLength", {"long": 1, "short": -1}),
        ("forehead", "height"): ("ForeheadHeight", {"high": 1, "low": -1}),
    }
    rules = [
        rule
        for rule in STRICT_V1_REGISTRY.rules
        if rule.rule_id in selector.selection_profile.approved_rule_ids
    ]
    pairs = [
        ContrastPromptPair(
            pair_id=rule.rule_id,
            family=rule.family,
            dimension=rule.dimension,
            positive_value=rule.value,
            negative_values=rule.alternatives,
            positive_texts=(rule.template.format(descriptor=rule.descriptor),),
            negative_texts=tuple(
                rule.template.format(descriptor=value) for value in rule.alternatives
            ),
            contrast_kind=rule.contrast_kind,
            weight=rule.weight,
            evidence_ids=(),
        )
        for rule in rules
    ]
    plans = [ContrastPlan("ready", (pair,), (), (), {}) for pair in pairs]
    components = selector.scorer.score(
        plans, methods=("attribute_contrastive",)
    ).components
    records = []
    for query_index, pair in enumerate(pairs):
        metric, directions = axis_specs[(pair.family, pair.dimension)]
        values = components.loc[
            components["query_index"].eq(query_index)
        ].sort_values("image_index")["contrast_score"]
        raw_rho = float(
            pd.Series(values.to_numpy()).corr(
                pd.Series(selector.pool_manifest[metric].to_numpy()), method="spearman"
            )
        )
        expected_direction = int(directions[pair.positive_value])
        records.append(
            {
                "rule_id": pair.pair_id,
                "family": pair.family,
                "dimension": pair.dimension,
                "positive_value": pair.positive_value,
                "cfd_metric": metric,
                "n_candidates": len(selector.pool_manifest),
                "raw_spearman_rho": raw_rho,
                "expected_direction": expected_direction,
                "directional_rho": raw_rho * expected_direction,
            }
        )
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    raise SystemExit(main())
