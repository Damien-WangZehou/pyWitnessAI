from __future__ import annotations

import pandas as pd

from .clip_backend import SentenceTransformerClipEncoder
from .index import ClipIndex


def build_filler_sets(
    index: ClipIndex,
    queries: pd.DataFrame,
    encoder: SentenceTransformerClipEncoder,
    top_k: int = 50,
    filler_count: int = 5,
    batch_size: int = 32,
    show_progress: bool = False,
    exclude_same_target: bool = True,
    max_pairwise_clip_similarity: float | None = None,
) -> pd.DataFrame:
    """Select top CLIP-matched fillers for each query."""
    if "description" not in queries.columns:
        raise ValueError("Queries must contain a description column.")

    query_table = queries.reset_index(drop=True).copy()
    exclude_target_ids = None
    if exclude_same_target and "target_id" in query_table.columns:
        exclude_target_ids = [{str(value)} for value in query_table["target_id"]]

    results = index.search_texts(
        query_table["description"].astype(str).tolist(),
        encoder=encoder,
        top_k=top_k,
        batch_size=batch_size,
        show_progress=show_progress,
        exclude_target_ids=exclude_target_ids,
    )

    records = []
    for query_index, candidates in results.groupby("query_index", sort=True):
        selected_indices: list[int] = []
        query_record = query_table.iloc[int(query_index)].to_dict()
        for candidate in candidates.sort_values("rank").to_dict(orient="records"):
            image_index = int(candidate["image_index"])
            if not _passes_diversity(
                index=index,
                image_index=image_index,
                selected_indices=selected_indices,
                max_pairwise_clip_similarity=max_pairwise_clip_similarity,
            ):
                continue

            selected_indices.append(image_index)
            records.append(
                {
                    "query_index": int(query_index),
                    "query_id": query_record.get("query_id", query_index),
                    "query_target_id": query_record.get("target_id", ""),
                    "filler_position": len(selected_indices),
                    "candidate_rank": int(candidate["rank"]),
                    "clip_score": float(candidate["clip_score"]),
                    "filler_image_id": candidate["image_id"],
                    "filler_target_id": candidate["target_id"],
                    "filler_image_path": candidate["image_path"],
                    "description": query_record.get("description", ""),
                }
            )
            if len(selected_indices) >= filler_count:
                break

    return pd.DataFrame.from_records(records)


def _passes_diversity(
    index: ClipIndex,
    image_index: int,
    selected_indices: list[int],
    max_pairwise_clip_similarity: float | None,
) -> bool:
    if max_pairwise_clip_similarity is None or not selected_indices:
        return True

    candidate = index.image_embeddings[image_index]
    selected = index.image_embeddings[selected_indices]
    pairwise_scores = selected @ candidate
    return bool((pairwise_scores <= max_pairwise_clip_similarity).all())
