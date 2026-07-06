"""DANTE retrieval ablation — 7 retriever configs on the ESCI test split.

Produces the §8.1 table: BM25 only / Dense only / SPLADE only / Dense+BM25 (RRF) /
Dense+SPLADE (RRF) / Dense+BM25+SPLADE (RRF) / + ColBERT rerank, each scored
MRR@10 / nDCG@10 / Recall@{10,100,200} via ``eval.metrics``.

The plan (§8.1) gives the table shape but not the runner code, so this is a typed stub.
See SPARDA_BUILD_PLAN.md §8.1.
"""

from __future__ import annotations


def _unpack_esci_test(esci_test):
    """Split ``esci_test`` into DANTE's ``(queries, qrels)`` shape.

    Accepts either a ``(queries, qrels)`` pair or a mapping with ``queries``/``qrels``
    keys, where ``queries = {query_id: query_text}`` and
    ``qrels = {query_id: {product_id: grade}}`` (Exact=3/Substitute=2/Complement=1/
    Irrelevant=0). DANTE's ``run_all_ablations`` counts recall positives as grade>=2.
    """
    if isinstance(esci_test, dict) and "queries" in esci_test and "qrels" in esci_test:
        return esci_test["queries"], esci_test["qrels"]
    if isinstance(esci_test, (tuple, list)) and len(esci_test) == 2:
        return esci_test[0], esci_test[1]
    raise TypeError(
        "esci_test must be (queries, qrels) or a dict with 'queries' and 'qrels' keys "
        "(queries={qid: text}, qrels={qid: {pid: grade}})."
    )


def run_retrieval_ablation(dante, esci_test, ks=(10, 100, 200),
                           max_queries: int = 2000, seed: int = 42) -> dict:
    """
    Run the DANTE retriever ablation on the ESCI test split (§8.1).

    THIN by design: this reuses DANTE's own ``run_all_ablations`` /``evaluate_ranker``
    (the exact same retrieval + metric code that produced DANTE's own §5.2 table), so
    every SPARDA row shares production's retrieval path — no re-implementation, no metric
    drift. Runs on the free T4 → ~0h marginal A100 (§9 Step 7).

    Args:
        dante: a constructed ``dante.DanteSearchEngine`` (≡ DanteRetriever, §2.1)
            exposing the per-leg helpers (dense/SPLADE/BM25) + RRF fusion + ColBERT/CE
            rerank that ``run_all_ablations`` drives.
        esci_test: the ESCI test split as ``(queries, qrels)`` or a
            ``{"queries":..., "qrels":...}`` mapping (Exact=3/Substitute=2/Complement=1/
            Irrelevant=0, §3.1).
        ks: recall cutoffs to report.
        max_queries: subsample cap (keeps ColBERT in budget); DANTE default ~2000.
        seed: subsample seed (shared with DANTE so rows reproduce).

    Returns:
        DANTE's ablation dict: ``{"results": {config: {metric: val}}, "table": str,
        "n_queries": int}``.
    """
    # Imported inside the function so this module parses without `dante` installed
    # (scaffold/CI); on Modal the vendored dante/ is on sys.path in the container.
    from dante.eval.evaluate import run_all_ablations

    queries, qrels = _unpack_esci_test(esci_test)
    return run_all_ablations(dante, queries, qrels, ks=tuple(ks),
                             max_queries=max_queries, seed=seed)
