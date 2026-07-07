"""SPARDA runtime integration layer — build ONE ``SpardaPipeline`` from the reused
DANTE + VERGIL artifacts, sharing a single Qwen generator.

This is the composition boundary (§2.1): it constructs ``dante.DanteSearchEngine`` from
the ``dante-artifacts`` indices and stands up VERGIL's graph/communities/summaries from
``vergil-artifacts`` (optionally the SPARDA-enriched graph), injecting ONE shared
``QwenLLM`` into the router, entity extractor, and answer generator. Both ``modal_run.py``
(e2e stage) and ``modal_demo.py`` (deployed UI) call ``build_pipeline`` so the two never
drift.

ALL heavy imports (dante/vergil/torch/sentence-transformers) live INSIDE functions, and
``_ensure_vendored_on_path`` puts the Modal-vendored ``/root/dante`` + ``/root/vergil`` on
``sys.path`` first — so this module imports fine in the scaffold/CI sandbox where neither
dependency is installed.

Two small ADAPTERS bridge scaffold↔real interface gaps (the engine/demo were written
against a slightly different DANTE surface than the shipped one):

  * ``DanteEngineAdapter`` — the shipped ``DanteSearchEngine`` has NO ``.colbert``
    attribute (it calls the module-level ``colbert_rerank``) and its ``.splade`` is a
    ``SpladeEncoder`` with no ``visualize_expansion`` METHOD (that's a module function).
    ``engine/multi_hop_search.py`` calls ``dante.colbert.rerank(...)`` and
    ``demo/app.py`` calls ``dante.splade.visualize_expansion(...)`` — the adapter exposes
    both, delegating everything else to the real engine. This keeps ``engine/`` and
    ``demo/`` untouched.
"""

from __future__ import annotations

import os
import sys

# Where the Modal image vendors the two sibling packages (see modal_run/modal_demo).
_VENDOR_ROOT = "/root"


def _ensure_vendored_on_path() -> None:
    """Put the Modal-vendored dante/vergil packages on sys.path (idempotent)."""
    if _VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _VENDOR_ROOT)


# ── DANTE reranker/splade shims ────────────────────────────────────────────────
class _ColbertShim:
    """Exposes ``.rerank(query, candidates, top_k)`` backed by dante's ``colbert_rerank``.

    ``engine/multi_hop_search.py`` expects ``dante.colbert.rerank(...)``; the shipped
    ``DanteSearchEngine`` has no ``.colbert`` (it uses the module-level function), so
    this bridges the gap without editing the engine.
    """

    def __init__(self, colbert_rerank_fn):
        self._fn = colbert_rerank_fn

    def rerank(self, query, candidates, top_k: int = 20):
        return self._fn(query, candidates, top_k=top_k)


class _SpladeShim:
    """Wraps the real ``SpladeEncoder`` and adds a ``visualize_expansion`` METHOD.

    ``demo/app.py`` calls ``dante.splade.visualize_expansion(query, top_k_terms=...)``;
    on the shipped DANTE that is a module-level function, so this method forwards to it.
    All other attribute access delegates to the real encoder.
    """

    def __init__(self, real_splade, visualize_expansion_fn):
        self._splade = real_splade
        self._viz = visualize_expansion_fn

    def visualize_expansion(self, query, top_k_terms: int = 20):
        return self._viz(query, self._splade, top_k_terms=top_k_terms)

    def __getattr__(self, name):
        return getattr(self._splade, name)


class DanteEngineAdapter:
    """Thin wrapper over ``dante.DanteSearchEngine`` exposing ``.colbert`` + a
    ``.splade`` with ``visualize_expansion``; everything else delegates to the engine."""

    def __init__(self, engine, colbert_rerank_fn, visualize_expansion_fn):
        self._engine = engine
        self.colbert = _ColbertShim(colbert_rerank_fn)
        self.splade = _SpladeShim(engine.splade, visualize_expansion_fn)

    def __getattr__(self, name):
        # search / product_db / biencoder / fused / dense_only / ... → real engine
        return getattr(self._engine, name)


def dante_config(dante_artifacts: str) -> dict:
    """DANTE serving config pointing at the mounted ``dante-artifacts`` volume.

    Mirrors DANTE's own ``_default_config`` (modal_train.py) with the artifact root
    swapped to the SPARDA mount point. Layout on the volume:
      ``{root}/biencoder_gte_hn`` — DANTE v0.2 winner bi-encoder (gte-modernbert-base +
                                    hard negatives; Dense R@200 0.698 vs v0.1 0.627)
      ``{root}/index_gte/``       — dense.faiss, bm25.pkl, splade.npz, product_ids.json
      ``{root}/data/catalog.parquet`` — [product_id, product_text] retrieval pool
    (G2 promotion: v0.2 ablation picked gte-modernbert-HN as the dense winner + Dense+SPLADE
    as the best fusion (R@200 0.7296 / nDCG@10 0.4461). The v0.1 biencoder_final/index remain
    on the volume for reproducibility.)
    """
    return {
        "biencoder": {"path": f"{dante_artifacts}/biencoder_gte_hn"},
        "splade": {
            "model": "opensearch-project/opensearch-neural-sparse-encoding-v2-distill",
            "max_length": 256,
        },
        "colbert": {"model": "answerdotai/answerai-colbert-small-v1"},
        "serving": {
            "catalog_path": f"{dante_artifacts}/data/catalog.parquet",
            "index_dir": f"{dante_artifacts}/index_gte",
            "rrf_k": 60,
            "top_n": 200,
            "leg_top_k": 1000,
        },
    }


def build_dante(dante_artifacts: str = "/dante-artifacts"):
    """Construct the DANTE search engine (wrapped in the adapter) from its artifacts."""
    _ensure_vendored_on_path()
    from dante.models.colbert_reranker import colbert_rerank
    from dante.models.splade import visualize_expansion
    from dante.serving.search_engine import DanteSearchEngine

    engine = DanteSearchEngine(dante_config(dante_artifacts))
    return DanteEngineAdapter(engine, colbert_rerank, visualize_expansion)


# VERGIL's own encoder — MUST match the space the cached summary_embeddings.npy were
# built in (bge-small, 384-d). engine/global_search embeds the query with this encoder
# and cosine-compares against those cached embeddings, so injecting DANTE's 768-d
# bi-encoder here would be a dimension mismatch. DANTE's bi-encoder stays the dense
# retrieval leg INSIDE DanteSearchEngine; this encoder only serves community search.
VERGIL_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # shared generator (Apache-2.0, non-thinking)


def build_pipeline(
    dante_artifacts: str = "/dante-artifacts",
    vergil_artifacts: str = "/vergil-artifacts",
    enriched_graph_path: str | None = "/sparda-artifacts/enriched_graph.pkl",
    llm=None,
):
    """Build the full ``SpardaPipeline`` once, sharing a single Qwen generator.

    Args:
        dante_artifacts: mount of the read-only ``dante-artifacts`` volume.
        vergil_artifacts: mount of the read-only ``vergil-artifacts`` volume.
        enriched_graph_path: SPARDA's complement-edge-enriched graph (written by the
            ``link`` stage). If it exists it is used; otherwise we fall back to VERGIL's
            plain ``graph.pkl`` (so the demo works even before ``link`` has run).
        llm: an optional pre-built ``QwenLLM`` to share; if None, one is constructed.

    Returns:
        a ready ``engine.pipeline.SpardaPipeline``.
    """
    _ensure_vendored_on_path()
    import json
    import pickle

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from vergil.generation.llm import QwenLLM

    from engine.pipeline import SpardaPipeline

    dante = build_dante(dante_artifacts)

    # ── VERGIL artifacts (reuse cached graph + communities + summaries) ──
    graph_path = enriched_graph_path if (enriched_graph_path and os.path.exists(enriched_graph_path)) \
        else f"{vergil_artifacts}/graph.pkl"
    print(f"[sparda] loading graph: {graph_path}")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    with open(f"{vergil_artifacts}/summaries.json") as f:
        summaries = json.load(f)
    summary_embs = np.load(f"{vergil_artifacts}/summary_embeddings.npy")

    # engine.coverage.global_coverage enables the 'global' path only when the graph
    # advertises communities; the cached graph doesn't carry this, so set it from the
    # summaries count (they ARE the built communities).
    graph.graph["num_communities"] = len(summaries)

    encoder = SentenceTransformer(VERGIL_ENCODER_NAME)  # matches summary_embeddings dims

    if llm is None:
        print(f"[sparda] loading shared generator: {QWEN_MODEL}")
        llm = QwenLLM(QWEN_MODEL, backend="transformers")

    # linked_db defaults to dante.product_db inside the pipeline, so coverage.join_rate
    # is measured honestly over the WHOLE DANTE catalog vs the graph (not just overlaps).
    pipeline = SpardaPipeline(
        dante=dante,
        vergil_graph=graph,
        llm=llm,
        communities=summaries,
        summary_embs=summary_embs,
        encoder=encoder,
    )
    return pipeline
