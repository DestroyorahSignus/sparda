"""DANTE retrieval ablation — 7 retriever configs on the ESCI test split.

Produces the §8.1 table: BM25 only / Dense only / SPLADE only / Dense+BM25 (RRF) /
Dense+SPLADE (RRF) / Dense+BM25+SPLADE (RRF) / + ColBERT rerank, each scored
MRR@10 / nDCG@10 / Recall@{10,100,200} via ``eval.metrics``.

The plan (§8.1) gives the table shape but not the runner code, so this is a typed stub.
See SPARDA_BUILD_PLAN.md §8.1.
"""

from __future__ import annotations

from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k  # noqa: F401 (used by the impl)


def run_retrieval_ablation(dante, esci_test, ks=(10, 100, 200)) -> dict:
    """
    Run the 7-config DANTE ablation on the ESCI test split and return a results table
    (config name -> {metric -> value}). Runs on the free T4 — ~0h marginal A100 (§9 Step 7).

    Args:
        dante: a ``dante.DanteSearchEngine`` (≡ DanteRetriever, §2.1) exposing the
            individual legs (dense/SPLADE/BM25) plus RRF fusion and ColBERT rerank.
        esci_test: the ESCI test split (query → graded relevance: Exact=3/Substitute=2/
            Complement=1/Irrelevant=0, §3.1).
        ks: recall cutoffs to report.
    """
    raise NotImplementedError(
        "TODO: SPARDA_BUILD_PLAN.md §8.1 — implement the 7-config ablation runner "
        "(BM25/Dense/SPLADE/their RRF fusions/+ColBERT) scoring MRR@10, nDCG@10, "
        "Recall@{10,100,200}; the plan specifies the table, not the runner."
    )
