# ⚔️ SPARDA — Hybrid Neural Search + GraphRAG Product Discovery Engine

SPARDA is a **unified e-commerce search and discovery engine** that routes each query to the
optimal retrieval strategy, then generates a grounded answer with graph-backed citations. It
is a thin **router + LLM-orchestration layer** that **composes two of my own standalone
projects** — it does not vendor them:

- **DANTE** — multi-stage hybrid retrieval (ModernBERT bi-encoder + SPLADE-v3 + BM25, fused
  with Reciprocal Rank Fusion, reranked by an `answerai-colbert-small-v1` late-interaction
  reranker).
- **VERGIL** — a product knowledge graph (brand / category / feature / `similar_to` edges)
  with Leiden community detection + LLM community summaries (GraphRAG).

Both are installed as **pip-from-git** dependencies; SPARDA imports their public APIs. The
headline is reuse: SPARDA adds **~0 marginal GPU hours** — it points its config at DANTE's
trained checkpoints and VERGIL's already-built graph + cached summaries and runs everything
else on a free Kaggle T4. A single shared **Qwen2.5-7B-Instruct Q4** generator serves all
three paths.

## Routing

Three paths, chosen by the router (heuristic → LLM fallback → coverage-aware degrade):

| Query type | Path | Stack |
|---|---|---|
| Product search ("best NC headphones under $300") | **local** | DANTE retrieval + VERGIL graph expansion |
| Market overview ("compare smart home ecosystems") | **global** | VERGIL community summaries |
| Relational ("accessories from Sony that work with the WH-1000XM5") | **multi_hop** | VERGIL graph traversal + DANTE ColBERT scoring |

Every decision (route, method, confidence, reason) is logged and shown in the demo badge.

## Composition contract (§2.1)

SPARDA does **not** vendor DANTE/VERGIL. It pip-installs them and imports their public APIs:

```python
from dante import DanteSearchEngine, reciprocal_rank_fusion
from vergil import (
    build_product_graph, add_similarity_edges, detect_communities,
    summarize_communities, embed_summaries,
    local_search, global_search, multi_hop_search,
)
```

- `DanteRetriever ≡ dante.DanteSearchEngine` — exposes `.search()`, `.colbert`, `.splade`,
  `.product_db`.
- SPARDA injects DANTE's fine-tuned bi-encoder into VERGIL by argument:
  `add_similarity_edges(G, encoder=dante.biencoder)` — shared state by injection, never by
  importing internals.

> `requirements.txt` pins `dante @ …@v0.2.0` and `vergil @ …@v0.1.0` for reproducible
> **local** installs. The **Modal** images do *not* pip-install them — they **vendor** the
> package source via `add_local_dir` from sibling clones (`../dante-src`, `../vergil-src`),
> which avoids private-repo auth at image-build time.

## The Complement-edge advantage (§3.3)

Amazon-Reviews-2023 ships **no co-purchase data** (`bought_together` is empty; `also_buy` /
`also_view` columns don't exist — audited 2026-06-26). So SPARDA synthesizes the missing
"goes-with"/accessory relationship from **ESCI `Complement` pairs**, injecting `complement_of`
edges into VERGIL's graph **where the two datasets overlap by ASIN**. This is a SPARDA-only
edge type that standalone VERGIL cannot produce. Report `complement_edges_added` together with
the **ASIN join rate** — together they are the project's truth gauge.

## Project structure

```
sparda/
├── configs/default.yaml      # all hyperparameters + the artifact-reuse contract
├── data/                     # SPARDA-own: ESCI↔Amazon linking + complement-edge synthesis
│   ├── __init__.py
│   └── link_datasets.py
├── engine/                   # the unified pipeline (SPARDA's own code)
│   ├── router.py             # query classification → RouteDecision
│   ├── coverage.py           # ASIN-link coverage gating
│   ├── local_search.py       # DANTE retrieval + VERGIL graph expansion
│   ├── global_search.py      # community-summary retrieval
│   ├── multi_hop_search.py   # graph traversal + DANTE scoring
│   ├── llm.py                # ONE shared Qwen2.5-7B Q4 wrapper
│   ├── prompts.py            # prompt templates
│   └── pipeline.py           # top-level orchestrator → unified answer schema
├── eval/                     # MRR/nDCG/Recall + DANTE & VERGIL ablations + test queries
├── demo/                     # Gradio UI (route badge + citations) + pyvis graph viz
└── scripts/                  # Phase-B bootstrap stubs (reuse-first; no-op if artifacts exist)
```

## Install

```bash
pip install -e .
# dante/vergil are private during development — install via token or local path if needed:
#   pip install git+https://${GH_TOKEN}@github.com/DestroyorahSignus/dante.git
#   pip install -e ../dante ../vergil
```

## Quickstart

```python
from engine import SpardaPipeline  # constructs the router; injects the shared Qwen
result = pipeline.answer("What accessories from Sony work with the WH-1000XM5?")
print(result["route"], result["answer"])
for c in result["citations"]:
    print(c["type"], c["name"], "—", c["evidence"])
```

## Run on Modal (link · e2e · deploy the demo)

The Modal images **vendor** DANTE + VERGIL from sibling clones (no private-repo auth).
Clone them next to this repo **once**, then run:

```bash
# 0) one-time: sibling clones the images vendor via add_local_dir
git clone git@github.com:DestroyorahSignus/dante.git  ../dante-src   # -> ../dante-src/dante
git clone git@github.com:DestroyorahSignus/vergil.git ../vergil-src  # -> ../vergil-src/vergil
pip install modal && modal token new

# 1) LINK: measure the ASIN join rate + enrich VERGIL's graph with ESCI complement edges
modal run modal_run.py --stage link      # CPU; writes /sparda-artifacts/{enriched_graph.pkl,link_stats.json}

# 2) E2E: run the §8.3 typed queries through the full SpardaPipeline + router accuracy
modal run modal_run.py --stage e2e       # A100-80GB; writes /sparda-artifacts/sparda_e2e.json

# 3) DEPLOY the demo → a persistent, shareable *.modal.run URL
modal deploy modal_demo.py               # -> https://<workspace>--sparda-demo-web.modal.run
```

`dante-artifacts` and `vergil-artifacts` are mounted **read-only**; SPARDA writes only its
own `sparda-artifacts`. One shared **Qwen3-4B-Instruct-2507** generator serves the router,
entity extractor, and all three answer paths.

## Results (fill in from `eval/`)

**DANTE retrieval ablation** (ESCI test split, §8.1):

| Configuration | MRR@10 | nDCG@10 | R@10 | R@100 | R@200 |
|---|---|---|---|---|---|
| BM25 only | | | | | |
| Dense only (ModernBERT) | | | | | |
| SPLADE only | | | | | |
| Dense + BM25 (RRF) | | | | | |
| Dense + SPLADE (RRF) | | | | | |
| Dense + BM25 + SPLADE (RRF) | | | | | |
| ↑ + ColBERT rerank | | | | | |

**VERGIL RAG ablation** (curated queries, §8.2):

| Query type | Vanilla RAG (ColBERT only) | SPARDA (graph + retrieval) | Δ |
|---|---|---|---|
| Local (product search) | | | |
| Global (market overview) | | | |
| Multi-hop (relational) | | | |
| Comparison (brand vs brand) | | | |

**Graph stats** (§8.4): nodes/edges by type, communities (L0/L1), avg community size, density,
most-connected brands, avg path length — plus **ASIN join rate** and **`complement_edges_added`**.

## Notes

- No OpenAI/Anthropic API calls — everything self-hosted (local Qwen2.5-7B Q4). This is the point.
- NetworkX (not Neo4j) is correct at this scale (50K-node graph).
- Built on **Amazon ESCI** (retrieval) + **Amazon Reviews 2023 Electronics** metadata (graph).
