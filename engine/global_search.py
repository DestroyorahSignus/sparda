"""Global search — community-summary retrieval for market-level queries.

See SPARDA_BUILD_PLAN.md §6.3.

`encoder` is DANTE's fine-tuned bi-encoder (the SAME one used to embed the cached community
summaries offline, so dimensions match — §6.5/B4). `community_summaries` and
`summary_embeddings` are the cached VERGIL artifacts.
"""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity


def global_search(query: str, community_summaries: list[dict],
                  summary_embeddings, encoder, top_k: int = 5) -> dict:
    """
    For broad queries: "compare smart home ecosystems", "trends in wireless audio".
    Basic RAG cannot answer these — requires aggregated community knowledge.
    """
    query_emb = encoder.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(query_emb, summary_embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        s = community_summaries[idx]
        results.append({
            "community_id": s["community_id"],
            "summary": s["summary"],
            "num_products": s["num_products"],
            "key_brands": s["key_brands"],
            "sample_product_ids": s["product_ids"][:5],
            "score": float(sims[idx]),
        })

    return {"communities": results, "retrieval_method": "global"}
