"""VERGIL RAG ablation — SPARDA (graph + retrieval) vs vanilla RAG (ColBERT only).

Produces the §8.2 table over the curated ``TEST_QUERIES`` per query type, manually scored
1-5 for faithfulness / relevance / completeness. Also reports router accuracy against the
labeled expected routes (RISKS R4) and the ASIN join rate (R1).

The plan (§8.2) gives the table shape but not the runner code, so this is a typed stub.
See SPARDA_BUILD_PLAN.md §8.2.
"""

from __future__ import annotations

from eval.test_queries import TEST_QUERIES  # noqa: F401 (used by the impl)


def router_accuracy(pipeline, test_queries=TEST_QUERIES) -> dict:
    """
    Report router accuracy + a confusion matrix over local/global/multi_hop, comparing
    ``pipeline.router.classify(q)`` against each query's labeled expected ``type``
    (RISKS R4 / §8.2).
    """
    raise NotImplementedError(
        "TODO: SPARDA_BUILD_PLAN.md §8.2 — compare routed vs expected route per query, "
        "return accuracy + confusion matrix."
    )


def run_rag_ablation(pipeline, vanilla_baseline, test_queries=TEST_QUERIES) -> dict:
    """
    Run SPARDA vs vanilla-RAG (ColBERT-only) over the curated queries, per query type,
    manually scored for faithfulness/relevance/completeness; multi-hop and global are
    where SPARDA wins (§8.2). Runs on the free T4 — ~0h marginal A100 (§9 Step 7).
    """
    raise NotImplementedError(
        "TODO: SPARDA_BUILD_PLAN.md §8.2 — implement the SPARDA-vs-vanilla-RAG ablation "
        "over TEST_QUERIES with 1-5 faithfulness/relevance/completeness scoring; "
        "the plan specifies the table, not the runner."
    )
