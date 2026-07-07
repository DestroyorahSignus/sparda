# ⚔️ SPARDA — A Query Router that Composes Two Retrieval Engines into One Answer

> **Live demo → [https://rapid-claims--sparda-demo-web.modal.run](https://rapid-claims--sparda-demo-web.modal.run)**
> Ask about products in natural language; watch it pick a route, stream a grounded answer, and cite its sources.

SPARDA is a **query router** for e-commerce discovery. It classifies each incoming query,
routes it to the right retrieval strategy across **two of my own standalone engines**, and
returns a single unified answer with typed citations and a visible "why this path" decision.

- **DANTE** — a multi-stage **hybrid product search** engine (dense bi-encoder + SPLADE
  learned-sparse + BM25, fused with Reciprocal Rank Fusion, reranked by a ColBERT
  late-interaction reranker).
- **VERGIL** — a **GraphRAG knowledge graph** over the product catalog (brand / category /
  feature / `similar_to` edges, Leiden community detection + LLM community summaries).

SPARDA does not re-implement either. It **vendors them as source** and orchestrates their
public APIs behind one router, one shared generator, and one answer schema. The engineering
story is the router + the composition + a **deployed, streaming web demo** — and a long,
honest list of real bugs fixed to get there (see [Engineering journey](#engineering-journey--what-we-tried--what-broke)).

### Headline facts

| | |
|---|---|
| **Router accuracy** | **15/15 = 1.000** on the typed test set; all three routes execute end-to-end |
| **Routes** | `local` (DANTE + 1-hop graph expand) · `global` (VERGIL community summaries) · `multi_hop` (VERGIL graph traversal + path citations) |
| **Generator** | **Qwen3-30B-A3B-Instruct-2507** (MoE, 30B total / 3B active, Apache-2.0), temp 0.0, **token-streamed** |
| **Graph** | VERGIL's own ~570K-edge product graph + **1,209** synthesized `complement_of` ("goes-with") edges |
| **Cross-dataset join** | **0.82%** ASIN overlap (2,898 of 351,961 ESCI × 49,997 electronics) — reported honestly; see [Limitations](#limitations) |
| **Deployment** | Modal ASGI + Gradio, one A100-80GB, scale-to-zero, read-only artifact volumes |

The 0.82% number is deliberately front-and-center. SPARDA's value is the **router**, not a
deep cross-dataset fusion — `multi_hop` runs on VERGIL's own graph via query→entity linking,
independent of the tiny ASIN join. Being honest about that is part of the project.

---

## Architecture

```mermaid
flowchart TD
    Q([User query]) --> R

    subgraph ROUTER["🧭 Router — engine/router.py"]
        R[classify] --> H{heuristic<br/>keyword match?}
        H -->|"conf ≥ 0.5"| DEC[RouteDecision]
        H -->|"conf < 0.5<br/>(ambiguous)"| LLMC[LLM classify<br/>shared Qwen · ≤4 tokens]
        LLMC --> DEC
        DEC --> COV{coverage gate<br/>path available?}
        COV -->|"graph path<br/>unavailable"| DEGRADE[degrade → local]
        COV -->|available| ROUTE
        DEGRADE --> ROUTE{route}
    end

    ROUTE -->|local| L
    ROUTE -->|global| G
    ROUTE -->|multi_hop| M

    subgraph LOCAL["🔴 LOCAL — DANTE hybrid search"]
        L[dante.search] --> L1["dense (bi-encoder/FAISS)<br/>+ SPLADE + BM25<br/>each top-1000"]
        L1 --> L2[RRF fuse k=30 → top-200]
        L2 --> L3[ColBERT MaxSim rerank → top-K]
        L3 --> L4[VERGIL 1-hop expand<br/>complement_of · has_brand ·<br/>has_feature · similar_to]
    end

    subgraph GLOBAL["🔵 GLOBAL — VERGIL communities"]
        G[encode query] --> G1[cosine vs cached<br/>community-summary embeddings]
        G1 --> G2[top-5 clusters<br/>summary · brands · products]
    end

    subgraph MULTIHOP["🟣 MULTI_HOP — VERGIL traversal"]
        M[extract entities<br/>Qwen → flat list str] --> M1[fuzzy entity-link<br/>rapidfuzz ≥ 70]
        M1 -->|no match| MFB[fallback → local]
        M1 --> M2[BFS ≤ 2 hops → product nodes]
        M2 --> M3[ColBERT score → top-20]
        M3 --> M4[nx.shortest_path<br/>→ readable path citations]
    end

    L4 --> GEN
    G2 --> GEN
    M4 --> GEN
    MFB --> L

    subgraph SHARED["⚙️ Shared generation"]
        GEN[assemble grounded prompt] --> QWEN[Qwen3-30B-A3B<br/>temp 0.0 · streamed]
    end

    QWEN --> OUT([Unified answer<br/>answer · route · citations · join_rate])
```

### Components

**Router (`engine/router.py`) — the core IP.** A three-stage decision that is fast and
cheap first, smart only when it has to be:

1. **Heuristic.** Regex over a `MULTI_HOP_KEYWORDS` set ("works with", "compatible",
   "accessories", "goes with", …) then a `GLOBAL_KEYWORDS` set ("compare", "vs", "trend",
   "ecosystem", …). A strong match returns immediately with high confidence — free and
   deterministic. No keyword → a weak `local` default at confidence 0.4.
2. **LLM fallback.** Only if heuristic confidence `< 0.5` and the shared LLM exists, ask
   Qwen for a **one-word** label (`local` / `global` / `multi_hop`) at temp 0.0. Ambiguous
   queries get real classification without paying for the LLM on the easy ones.
3. **Coverage gate.** If the chosen path is graph-dependent (`global` / `multi_hop`) but the
   graph can't serve it, **degrade to local** — DANTE always works.

Every decision is returned as a `RouteDecision` (route, method, confidence, matched rule,
reason, available paths) so the pipeline logs it and the demo renders a "why this path" badge.

**LOCAL route (`engine/local_search.py`).** DANTE's 4-signal retrieval — dense
(bi-encoder → FAISS inner-product), SPLADE learned-sparse, and BM25, each top-1000 → **RRF
fuse** (k=30) → top-200 → **ColBERT MaxSim rerank** → top-K. Then VERGIL expands the top-5
results with 1-hop graph neighbors along the edges that actually exist on the data
(`complement_of`, `has_brand`, `has_feature`, `similar_to` — **not** `bought_together`,
which is empty on Amazon-2023). If nothing retrieved is in the graph, expansion is simply
empty and local still answers.

**GLOBAL route (`engine/global_search.py`).** For market-level questions ("compare smart
home ecosystems"). Encode the query with VERGIL's own encoder (bge-small, 384-d — the same
space the summaries were embedded in) and cosine-match against the **pre-computed,
cached** community-summary embeddings; return the top-5 clusters. No per-query
summarization, no graph walk — the summaries were generated offline during VERGIL's build.

**MULTI_HOP route (`engine/multi_hop_search.py`).** For relational questions ("accessories
from Sony that work with the WH-1000XM5"): (1) extract entities via Qwen; (2) fuzzy-link each
to a graph node (`rapidfuzz`, threshold 70) — if none match, the pipeline falls back to
local; (3) BFS up to 2 hops to discover product nodes; (4) score them with DANTE's ColBERT
reranker → top-20; (5) build a readable **reasoning path** per product
(`A --[has_brand]--> Sony --[…]--> B`) used as a citation.

**Shared generator + unified schema (`engine/pipeline.py`).** One `Qwen3-30B-A3B` instance
serves the router, the entity extractor, and all three answer paths — never reloaded per
mode. `_prepare()` does routing + retrieval once, shared by the blocking `answer()` and the
streaming `answer_stream()`. Every route returns the same schema:

```python
{
  "answer":    "...",                       # grounded natural-language answer
  "route":     "local" | "global" | "multi_hop",
  "routing":   {method, confidence, reason, available_paths},
  "citations": [{type, id, name, evidence}, ...],   # product / graph_edge / community / path
  "join_rate": 0.0082,                      # informational
  "context":   {...},                       # raw retrieval context
}
```

**Composition boundary (`sparda_runtime.py`).** `build_pipeline()` constructs the DANTE
engine from the read-only `dante-artifacts` volume, loads VERGIL's graph / communities /
cached summaries from `vergil-artifacts` (preferring SPARDA's complement-edge-enriched graph
if present), and injects one shared `QwenLLM`. Two small **adapters** bridge scaffold↔shipped
interface gaps without touching the engine code: a `_ColbertShim` (exposes
`dante.colbert.rerank(...)` over the module-level `colbert_rerank`) and a `_SpladeShim`
(adds `visualize_expansion` as a method for the demo panel). All heavy imports live inside
functions, so the module imports fine in CI where neither dependency is installed.

### Deployment

`modal_demo.py` builds the pipeline **once per container** inside the ASGI factory (not at
import) and serves a Gradio Blocks UI via Modal's ASGI pattern on one **A100-80GB**.
`scaledown_window=300` keeps the (expensive) warm model resident for 5 minutes of idle before
scaling to zero. `@modal.concurrent(max_inputs=8)` lets one warm model serve several UI
requests. DANTE + VERGIL are **vendored as source** via `add_local_dir` from the
`../dante-src` / `../vergil-src` siblings (no private-repo auth at image-build time). The
three artifact volumes (`dante-artifacts`, `vergil-artifacts`, `sparda-artifacts`) mount
**read-only** — the 30B is pre-cached into the volume by `fetch_model.py` beforehand, since a
read-only volume can't download at serve time.

---

## Engineering journey — what we tried / what broke

This is the real story. SPARDA is a thin layer, but shipping the composition and a live demo
surfaced a long chain of genuine bugs. Each is presented **problem → root cause → fix**,
honestly (including one thing I initially got wrong and corrected).

### 1. Coverage gate silently disabled multi_hop for *every* query
- **Problem:** `multi_hop` never ran — the router always degraded to `local`.
- **Root cause:** `global_coverage` gated multi_hop on the ESCI↔Amazon **ASIN join rate ≥ 5%**.
  The real join is **0.82%**, so the gate was never satisfied and every relational query fell
  back to local — even though VERGIL's graph has ~570K edges and multi_hop links the *query's*
  entities to graph nodes, never needing a DANTE product to be in the graph at all.
- **Fix:** gate path availability on the **graph's own** properties (edges present →
  multi_hop; communities present → global). `join_rate` is retained but reported as
  **informational only**. (`engine/coverage.py`.)

### 2. multi_hop crashed on every query, taking the demo down with it (G1)
- **Problem:** the flagship "accessories that work with X" example errored, and the whole
  demo container died on it.
- **Root cause:** DANTE's `colbert.rerank(...)` returns candidate dicts with `product_id`
  at the **top level**, but the code read `result["product"]["product_id"]` → `""`. That empty
  id was passed to `nx.shortest_path(G, start, "")`, which raised `NodeNotFound` — a *sibling*
  of `NetworkXNoPath`, and the `except` only caught the latter, so the crash propagated.
- **Fix:** read the id at the correct level (`result.get("product_id") or …`), skip empty /
  non-graph ids, and catch **both** `NetworkXNoPath` and `NodeNotFound`.
  (`engine/multi_hop_search.py`.)

### 3. Entity extraction crashed multi_hop on LLM output shape
- **Problem:** even after G1, multi_hop crashed inside entity linking.
- **Root cause:** the prompt asked for a JSON list of strings, but Qwen sometimes emitted a
  **list of dicts** (e.g. `[{"brand":"Sony"}]`). Downstream `entity.lower()` blew up on a dict.
- **Fix:** `_extract_entities` now flattens **any** LLM output shape — list-of-str,
  list-of-dict, `{"entities":[...]}`, prose-wrapped JSON — down to a flat `list[str]`, with a
  salient-token fallback if extraction yields nothing. (`engine/pipeline.py`.)

### 4. Serving the wrong (v0.1) DANTE model (G2 — winner promotion)
- **Problem:** SPARDA was serving DANTE's older checkpoint, not the ablation winner.
- **Root cause:** `sparda_runtime.dante_config` hardcoded the v0.1 bi-encoder + index.
- **Fix:** repointed to DANTE's **v0.2 winner** — `biencoder_gte_hn` (gte-modernbert +
  hard negatives) with `index_gte` and **RRF k=30** (the ablation showed k≤30 beats k=60 on
  nDCG@10 at equal recall). SPARDA now serves the improved retriever. (`sparda_runtime.py`.)

### 5. Demo boot crash-loops (two separate dependency traps)
- **(a) No parquet engine.** `pd.read_parquet(catalog)` threw `ImportError` on boot. Pandas
  3.0 ships **no bundled parquet engine**, and the demo image (unlike the e2e image, which got
  it transitively via `datasets`) had no `pyarrow`. → Added `pyarrow>=17,<21` explicitly.
- **(b) Gradio version drift.** Unpinned `gradio` resolved to v6, which is
  **messages-format-only** and dropped the `Chatbot(type=...)` kwarg → boot crash. But
  `gradio 5` conflicts with pandas 3.0 and `gradio 6.19` conflicts with transformers 4.57. →
  Pinned `gradio>=6.0,<6.19` (the only compatible lane) **and** converted the chat handler to
  role/content messages format. (`modal_demo.py`, `demo/app.py`.)

### 6. "Connection lost" on every search
- **Problem:** the browser showed "connection lost" on every query, even though the backend
  answered fine when called via `gradio_client`. Server logs showed Starlette "response
  already started".
- **Root cause:** the ~20s **non-streaming** 30B response left the Gradio SSE connection
  idle, and Modal's proxy dropped the idle connection before the answer arrived.
- **Fix:** **token streaming**. Added `QwenLLM.generate_stream` (`TextIteratorStreamer` +
  a background generation thread); `SpardaPipeline.answer_stream` routes/retrieves once then
  yields an accumulating answer; the demo `chat()` is now a generator yielding partial history
  with a `▌` cursor. SSE receives continuous data and never idles. **Verified: 244 incremental
  updates over one query, connection stayed alive end-to-end** — and much better UX.

### 7. Weak / "no products found" answers
- **Problem:** the local route often replied "there are no products in the results".
- **Root cause:** DANTE's ESCI catalog has **no `name` field** — only `product_id` and
  `product_text`. The formatters keyed on `name`, so the LLM literally saw `1. ? 2. ?` and
  concluded there were no products.
- **Fix:** `_fmt_products` / `_cite_products` fall back to `product_text` (truncated to 320
  chars — enough concrete detail to anchor on). Named products with specs now come back
  (e.g. *LETSCOM U8L, Letsfit, Cshidworld, JBL TUNE 750BTNC* for a headphones query).
  (`engine/pipeline.py`.)

### 8. Grounding hardening (and an honest self-correction)
- **What I did:** rewrote the LOCAL / GLOBAL / MULTI_HOP prompts around a shared strict
  **anti-fabrication block** (`_GROUNDING` in `engine/prompts.py`): use only the numbered
  items; never invent or recall a product name / spec / price from memory; never compare
  against an unlisted product; only state specs written in the item's own text. Answers run
  at **temperature 0.0**.
- **Honest correction:** I initially flagged an "S20 FE hallucination". On re-checking the
  citations, the Galaxy S20 FE was **actually a retrieved product** (citations #4/#5) — I'd
  only looked at the first three citations. It was a **misdiagnosis**, not a hallucination.
  The hardening is **kept** anyway as genuine robustness (it prevents real fabrication on
  other queries), and the local route now verifiably recommends listed products by `#N`.
- **Marketing neutrality:** also added a rule not to endorse sellers' own title claims
  ("Best", "#1", "Award-winning", "CNET's Award") as fact. ESCI ranks by **text relevance**,
  not reviews or sales, so superlative queries read like a search engine ("here are the
  closest matches"), not a shill.

### 9. UI redesign — dark "devil-hunter" theme
- **From:** the bland Gradio default. **To:** a dark themed UI matching the portfolio —
  Dante-crimson / Vergil-steel / Sparda-violet — applied by overriding Gradio's own CSS
  custom properties on `.gradio-container` (robust across versions), plus a hero header with
  a 3-route legend, **route-badge pills** (`sanitize_html=False`), tabbed side panels (pyvis
  knowledge-graph view / SPLADE term-expansion), and **Enter-to-search** (`query.submit`).
  System fonts only — no external fetches (CSP-safe). (`demo/app.py`, `demo/graph_viz.py`.)

### 10. Deploy gotcha (worth remembering)
- `modal app stop` on a *deployed* app **404s** until you re-deploy, and stale warm
  containers keep serving old code within the scaledown window. The reliable update for a
  deployed app is **stop + deploy**, not stop alone.

---

## How to run

DANTE + VERGIL are vendored from sibling clones (no private-repo auth at build time). Clone
them **once** next to this repo, then:

```bash
# 0) one-time: sibling clones the Modal images vendor via add_local_dir
git clone git@github.com:DestroyorahSignus/dante.git  ../dante-src   # -> ../dante-src/dante
git clone git@github.com:DestroyorahSignus/vergil.git ../vergil-src  # -> ../vergil-src/vergil
pip install modal && modal token new

# 1) LINK (CPU): measure the ASIN join rate + enrich VERGIL's graph with ESCI complement edges
modal run modal_run.py --stage link      # writes /sparda-artifacts/{enriched_graph.pkl,link_stats.json}

# 2) E2E (A100-80GB): run the typed queries through the full SpardaPipeline + router accuracy
modal run modal_run.py --stage e2e       # writes /sparda-artifacts/sparda_e2e.json

# 3) DEPLOY the demo → a persistent, shareable *.modal.run URL
modal deploy modal_demo.py               # -> https://<workspace>--sparda-demo-web.modal.run
```

Notes:
- All three artifact volumes mount **read-only**; the 30B generator must be pre-cached into
  the `sparda-artifacts` volume by `fetch_model.py` first (a read-only volume can't download
  at serve time).
- Generator load on A100-80GB is clean: **16 shards, ~71s, no OOM.**

---

## Limitations

Stated up front, because honesty is the point of this write-up:

- **ESCI is a fixed research catalog with no ratings / sales signal** — it ranks by
  **relevance only**. So "best X" queries surface the *closest matches*, not curated picks,
  and the UI + prompts say so explicitly rather than pretending otherwise.
- **The cross-dataset ASIN join is 0.82%.** SPARDA is a **router**, not a fused index. Its
  value is classifying a query and routing it to the right engine — and `multi_hop` runs on
  VERGIL's own ~570K-edge graph via query→entity linking, so it does not depend on that tiny
  overlap. The 1,209 synthesized `complement_of` edges are a real SPARDA-only signal but a
  small one; where overlap is near-zero, expansion gracefully degrades to
  brand/category/feature/similar_to.
- **Router accuracy (15/15)** is measured on a small typed test set — a sanity signal that
  the classification + routing works end-to-end, not a large-scale benchmark.

---

## Repository layout

```
sparda/
├── engine/                    # SPARDA's own code — the router + orchestration
│   ├── router.py              # heuristic → LLM fallback → coverage gate → RouteDecision
│   ├── coverage.py            # graph-based path availability (join_rate is informational)
│   ├── local_search.py        # DANTE hybrid retrieval + VERGIL 1-hop graph expansion
│   ├── global_search.py       # cached community-summary retrieval
│   ├── multi_hop_search.py    # entity-link → BFS → ColBERT score → path citations
│   ├── prompts.py             # grounded, anti-fabrication prompt templates
│   └── pipeline.py            # top-level orchestrator → unified answer schema (+ streaming)
├── demo/
│   ├── app.py                 # Gradio UI: route badge, streaming chat, graph + SPLADE panels
│   └── graph_viz.py           # pyvis subgraph rendering
├── data/link_datasets.py      # ESCI↔Amazon ASIN link + complement-edge synthesis
├── eval/                      # DANTE + VERGIL ablations + typed test queries + router accuracy
├── sparda_runtime.py          # composition boundary: build_pipeline() + DANTE/VERGIL adapters
├── modal_run.py               # link · e2e stages
├── modal_demo.py              # deployed ASGI + Gradio web app
└── fetch_model.py             # pre-cache the 30B generator into the read-only volume
```

## Notes

- **No OpenAI / Anthropic API calls** — the generator is self-hosted Qwen3-30B-A3B. That is
  deliberate.
- **NetworkX (not Neo4j)** is the right tool at this scale (~50K-node graph).
- Built on **Amazon ESCI** (search relevance) + **Amazon Reviews 2023 Electronics** metadata
  (product graph).
</content>
</invoke>
