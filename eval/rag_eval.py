"""VERGIL RAG ablation — SPARDA (graph + retrieval) vs vanilla RAG (ColBERT only).

Produces the §8.2 table over the curated ``TEST_QUERIES`` per query type, manually scored
1-5 for faithfulness / relevance / completeness. Also reports router accuracy against the
labeled expected routes (RISKS R4) and the ASIN join rate (R1).

The plan (§8.2) gives the table shape but not the runner code, so this is a typed stub.
See SPARDA_BUILD_PLAN.md §8.2.
"""

from __future__ import annotations

from eval.test_queries import TEST_QUERIES

ROUTES = ("local", "global", "multi_hop")


def _query_text(q) -> str:
    """SPARDA test queries key the text under 'q'; VERGIL's key it under 'query'."""
    if isinstance(q, dict):
        return q.get("q") or q.get("query") or ""
    return str(q)


def router_accuracy(pipeline, test_queries=TEST_QUERIES) -> dict:
    """
    Router accuracy + confusion matrix over local/global/multi_hop (RISKS R4 / §8.3).

    Runs each §8.3 typed query through the router and compares the CLASSIFIED route to
    the query's labeled ``type``. The router is called WITHOUT the coverage guard
    (``coverage=None``) so this measures the classifier itself, not the coverage-aware
    degradation that would rewrite unavailable routes to ``local`` at serve time.

    Args:
        pipeline: a ``SpardaPipeline`` (uses ``pipeline.router.classify``).
        test_queries: list of ``{"q": str, "type": local|global|multi_hop}`` (§8.3).

    Returns:
        ``{"accuracy": float, "n": int, "per_type": {type: {"n","correct","accuracy"}},
        "confusion": {expected: {predicted: count}}, "errors": [ ... ]}``.
    """
    confusion = {exp: {pred: 0 for pred in ROUTES} for exp in ROUTES}
    per_type = {t: {"n": 0, "correct": 0} for t in ROUTES}
    errors, n, correct = [], 0, 0

    for q in test_queries:
        expected = q["type"] if isinstance(q, dict) else None
        if expected not in ROUTES:
            continue  # only score the three labeled routes
        decision = pipeline.router.classify(_query_text(q))  # no coverage guard
        predicted = decision.route
        n += 1
        per_type[expected]["n"] += 1
        confusion[expected].setdefault(predicted, 0)
        confusion[expected][predicted] += 1
        if predicted == expected:
            correct += 1
            per_type[expected]["correct"] += 1
        else:
            errors.append({
                "query": _query_text(q), "expected": expected, "predicted": predicted,
                "method": decision.method, "confidence": decision.confidence,
                "reason": decision.reason,
            })

    for t, agg in per_type.items():
        agg["accuracy"] = round(agg["correct"] / agg["n"], 3) if agg["n"] else 0.0

    return {
        "accuracy": round(correct / n, 3) if n else 0.0,
        "n": n,
        "per_type": per_type,
        "confusion": confusion,
        "errors": errors,
    }


def format_router_table(acc: dict) -> str:
    """Printable per-type router-accuracy table + confusion matrix."""
    lines = ["=" * 52, "ROUTER ACCURACY (§8.3)", "=" * 52,
             f"overall: {acc['accuracy']:.3f}  (n={acc['n']})", "",
             f"{'expected':<11} {'n':>3} {'correct':>8} {'acc':>6}"]
    for t in ROUTES:
        a = acc["per_type"][t]
        lines.append(f"{t:<11} {a['n']:>3} {a['correct']:>8} {a['accuracy']:>6.3f}")
    lines += ["", "confusion (rows=expected, cols=predicted):",
              f"{'':<11}" + "".join(f"{p:>11}" for p in ROUTES)]
    for exp in ROUTES:
        row = acc["confusion"][exp]
        lines.append(f"{exp:<11}" + "".join(f"{row.get(p, 0):>11}" for p in ROUTES))
    lines.append("=" * 52)
    return "\n".join(lines)


def run_rag_ablation(rag, vector_index, llm, test_queries=TEST_QUERIES) -> dict:
    """
    SPARDA vs vanilla-RAG (vector-only) ablation — a thin REUSE of VERGIL's harness (§8.2).

    Delegates straight to ``vergil.eval.evaluate.run_rag_ablation`` (same mechanical
    scoring: answer_nonempty / cites_sources / used_graph, no LLM-judge), so SPARDA and
    VERGIL report the identical ablation. SPARDA's ``TEST_QUERIES`` key the text under
    ``"q"``; VERGIL's harness expects ``"query"`` — so the queries are re-keyed here
    before delegating (the only adaptation). Runs on the free T4 — ~0h marginal A100.

    Args:
        rag: a built ``VergilRAG`` (routed GraphRAG pipeline) whose ``.answer()`` returns
            VERGIL's ``{answer, query_type, sources, retrieval_method, fallback}`` schema.
        vector_index: the ``VectorIndex`` used as the vanilla (graph-free) baseline
            retriever — typically the same index ``rag`` uses for local search.
        llm: the shared ``QwenLLM`` (controlled: only retrieval differs between arms).
        test_queries: SPARDA ``{"q","type"}`` or VERGIL ``{"query","type"}`` dicts.

    Returns:
        VERGIL's ablation dict: ``{"per_query", "by_type", "side_by_side"}``.
    """
    from vergil.eval.evaluate import run_rag_ablation as _vergil_run_rag_ablation

    normalized = [
        {"query": _query_text(q), "type": (q.get("type") if isinstance(q, dict) else None)}
        for q in test_queries
    ]
    return _vergil_run_rag_ablation(rag, vector_index, llm, queries=normalized)
