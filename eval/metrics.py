"""Retrieval metrics — MRR@K, nDCG@K, Recall@K.

See SPARDA_BUILD_PLAN.md §8.5.
"""

from __future__ import annotations

import numpy as np


def mrr_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance_map: dict[str, int], k: int) -> float:
    dcg = sum(relevance_map.get(doc_id, 0) / np.log2(rank + 1)
              for rank, doc_id in enumerate(ranked_ids[:k], 1))
    ideal = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    found = sum(1 for d in ranked_ids[:k] if d in relevant_ids)
    return found / len(relevant_ids) if relevant_ids else 0.0
