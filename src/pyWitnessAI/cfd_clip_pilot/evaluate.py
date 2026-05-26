from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clip_backend import SentenceTransformerClipEncoder
from .index import ClipIndex


def evaluate_retrieval(
    index: ClipIndex,
    queries: pd.DataFrame,
    encoder: SentenceTransformerClipEncoder,
    top_k: int = 50,
    batch_size: int = 32,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate whether CLIP retrieves the expected CFD target for each description."""
    _validate_queries(queries)
    query_table = queries.reset_index(drop=True).copy()
    results = index.search_texts(
        query_table["description"].astype(str).tolist(),
        encoder=encoder,
        top_k=top_k,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    results = attach_query_metadata(results, query_table)
    results["is_target"] = _is_target_result(results)

    per_query = summarise_per_query(results, query_table, top_k=top_k)
    summary = summarise_retrieval(per_query, top_k=top_k)
    return results, per_query, summary


def attach_query_metadata(results: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    for column in ("query_id", "target_id", "image_id", "description"):
        if column in queries.columns:
            enriched[f"query_{column}"] = enriched["query_index"].map(queries[column].to_dict())
    return enriched


def summarise_per_query(
    results: pd.DataFrame,
    queries: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    per_query = queries.reset_index(drop=True).copy()
    target_hits = results[results["is_target"]].groupby("query_index")["rank"].min()
    per_query["target_rank"] = [target_hits.get(i, np.nan) for i in range(len(per_query))]
    per_query["found_in_top_k"] = per_query["target_rank"].notna()
    for k in (1, 5, 10, top_k):
        column = f"hit_top_{k}"
        per_query[column] = per_query["target_rank"].le(k).fillna(False)
    return per_query


def summarise_retrieval(per_query: pd.DataFrame, top_k: int = 50) -> dict:
    ranks = per_query["target_rank"]
    reciprocal_ranks = ranks.map(lambda rank: 0.0 if pd.isna(rank) else 1.0 / float(rank))
    summary = {
        "n_queries": int(len(per_query)),
        "top_k": int(top_k),
        "found_in_top_k": int(per_query["found_in_top_k"].sum()),
        "not_found_in_top_k": int((~per_query["found_in_top_k"]).sum()),
        "mrr_with_misses_as_zero": float(reciprocal_ranks.mean()) if len(per_query) else 0.0,
    }
    for k in (1, 5, 10, top_k):
        summary[f"hit_rate_top_{k}"] = float(per_query["target_rank"].le(k).fillna(False).mean())

    found_ranks = ranks.dropna()
    summary["mean_rank_found_only"] = float(found_ranks.mean()) if len(found_ranks) else None
    summary["median_rank_found_only"] = float(found_ranks.median()) if len(found_ranks) else None
    return summary


def write_evaluation_outputs(
    output_dir: str | Path,
    retrieval_results: pd.DataFrame,
    per_query: pd.DataFrame,
    summary: dict,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    retrieval_results.to_csv(root / "retrieval_results.csv", index=False)
    per_query.to_csv(root / "per_query_metrics.csv", index=False)
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _validate_queries(queries: pd.DataFrame) -> None:
    if "description" not in queries.columns:
        raise ValueError("Queries must contain a description column.")
    if "target_id" not in queries.columns and "image_id" not in queries.columns:
        raise ValueError("Queries must contain target_id or image_id for evaluation.")


def _is_target_result(results: pd.DataFrame) -> pd.Series:
    by_target = (
        results["query_target_id"].notna()
        & (results["target_id"].astype(str) == results["query_target_id"].astype(str))
        if "query_target_id" in results.columns
        else pd.Series(False, index=results.index)
    )
    by_image = (
        results["query_image_id"].notna()
        & (results["image_id"].astype(str) == results["query_image_id"].astype(str))
        if "query_image_id" in results.columns
        else pd.Series(False, index=results.index)
    )
    return by_target | by_image
