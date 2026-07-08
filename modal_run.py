"""SPARDA — Modal jobs: dataset LINK (complement-edge enrichment) + E2E typed-query run.

================================================================================
WHAT THIS IS
================================================================================
Two stages over the reused DANTE + VERGIL artifacts (SPARDA adds ~0 marginal GPU):

  * ``link`` (CPU)  — load DANTE ``catalog.parquet`` + VERGIL ``electronics_meta.parquet``,
      measure the ASIN join rate (``data.link_datasets.link_esci_to_amazon`` — the headline
      "does the graph enrich retrieval" number), then mine ESCI ``Complement`` pairs into
      VERGIL's graph as ``complement_of`` edges (``data.link_datasets.add_complement_edges``,
      the SPARDA-only "goes-with" signal Amazon-2023 lacks). Saves the enriched graph +
      link stats to ``sparda-artifacts``.
  * ``e2e`` (A100-80GB) — build ONE ``SpardaPipeline`` (DanteSearchEngine + VERGIL graph +
      shared Qwen3-4B), run the §8.3 typed queries through it, and write per-query
      route/answer/citations + a router-accuracy table to ``/sparda-artifacts/sparda_e2e.json``.

================================================================================
HOW THE IMAGE VENDORS DANTE + VERGIL (private-repo-auth-free)
================================================================================
Instead of ``pip install git+https://…`` (which needs a token for the PRIVATE dante/vergil
repos), the image VENDORS the two package source trees via ``add_local_dir``. The
orchestrator MUST clone the two repos as SIBLINGS of this repo BEFORE ``modal run``/``deploy``::

    <parent>/
      sparda/       <- this repo (contains modal_run.py)
      dante-src/    <- git clone git@github.com:DestroyorahSignus/dante.git dante-src
      vergil-src/   <- git clone git@github.com:DestroyorahSignus/vergil.git vergil-src

so that ``../dante-src/dante`` and ``../vergil-src/vergil`` (the PACKAGE dirs) exist. They
are copied to ``/root/dante`` and ``/root/vergil`` in the image (``/root`` is on sys.path),
and ``sparda_runtime`` imports them inside the container.

================================================================================
HOW TO RUN  (orchestrator runs these; CODE-ONLY here — do not deploy from this file)
================================================================================
    pip install modal && modal token new           # one-time
    modal run modal_run.py --stage link            # CPU: enrich graph + join stats
    modal run modal_run.py --stage e2e             # A100: typed-query e2e + router acc
    modal run modal_run.py --stage all             # link then e2e
"""

import os

import modal

# ── Sibling package paths the orchestrator must clone before deploy (see header) ──
HERE = os.path.dirname(os.path.abspath(__file__))
DANTE_PKG = os.path.abspath(os.path.join(HERE, "..", "dante-src", "dante"))
VERGIL_PKG = os.path.abspath(os.path.join(HERE, "..", "vergil-src", "vergil"))

app = modal.App("sparda-run")

# ── The shared ML stack (pins mirror the DANTE/VERGIL-validated Modal combo) ──
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
        "transformers==4.57.6",
        "sentence-transformers==4.1.0",
        "rerankers[transformers]==0.10.0",
        "accelerate==1.14.0",          # QwenLLM transformers backend (device_map="cuda")
        "faiss-cpu==1.14.3",
        "rank-bm25==0.2.2",
        "scipy==1.17.1",               # SPLADE CSR-matmul scoring
        "scikit-learn>=1.3",           # engine/global_search imports sklearn at module top
        "networkx>=3.2",
        "cdlib>=0.4.0",
        "leidenalg>=0.10.0",
        "python-igraph>=0.11.0",
        "rapidfuzz>=3.6.0",
        "pandas==3.0.3",
        "numpy==2.2.6",
        "datasets==5.0.0",             # link stage: load ESCI for Complement pairs
    )
    .env({"HF_HOME": "/sparda-artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
    # VENDOR dante + vergil source (copy=True layers must precede the runtime mount below).
    .add_local_dir(DANTE_PKG, "/root/dante", copy=True)
    .add_local_dir(VERGIL_PKG, "/root/vergil", copy=True)
    # SPARDA's own packages (runtime mount; keep last).
    .add_local_python_source("engine", "eval", "data", "demo", "sparda_runtime")
)


def _read_only(vol):
    """Return a read-only handle if the installed Modal supports it, else the volume.

    dante-artifacts / vergil-artifacts are consumed READ-ONLY — SPARDA never commits to
    them (it only writes its own sparda-artifacts)."""
    ro = getattr(vol, "read_only", None)
    return ro() if callable(ro) else vol


dante_vol = _read_only(modal.Volume.from_name("dante-artifacts", create_if_missing=True))
vergil_vol = _read_only(modal.Volume.from_name("vergil-artifacts", create_if_missing=True))
sparda_vol = modal.Volume.from_name("sparda-artifacts", create_if_missing=True)

VOLUMES = {
    "/dante-artifacts": dante_vol,
    "/vergil-artifacts": vergil_vol,
    "/sparda-artifacts": sparda_vol,
}

ENRICHED_GRAPH = "/sparda-artifacts/enriched_graph.pkl"
LINK_STATS = "/sparda-artifacts/link_stats.json"
E2E_JSON = "/sparda-artifacts/sparda_e2e.json"

# ESCI label normalization (mirror labels differ: full words vs single letters).
_LABELS = {
    "e": "Exact", "s": "Substitute", "c": "Complement", "i": "Irrelevant",
    "exact": "Exact", "substitute": "Substitute",
    "complement": "Complement", "irrelevant": "Irrelevant",
}


@app.function(image=image, volumes=VOLUMES, cpu=8.0, memory=32768, timeout=90 * 60)
def link(esci_dataset: str = "tasksource/esci"):
    """CPU: measure the ASIN join rate + enrich VERGIL's graph with ESCI complement edges."""
    import json
    import pickle

    import pandas as pd
    from datasets import load_dataset

    from data.link_datasets import add_complement_edges, link_esci_to_amazon

    # ── 1. Headline join rate: DANTE catalog × VERGIL graph metadata ──
    dante_catalog = pd.read_parquet("/dante-artifacts/data/catalog.parquet")
    vergil_meta = pd.read_parquet("/vergil-artifacts/electronics_meta.parquet")
    stats = link_esci_to_amazon(dante_catalog, vergil_meta)
    print(f"[link] ASIN join rate: {stats['join_rate']:.4%}  "
          f"(overlap {stats['n_overlap']:,} / dante {stats['n_dante']:,} / "
          f"vergil {stats['n_vergil']:,})")

    # ── 2. Load VERGIL's graph ──
    with open("/vergil-artifacts/graph.pkl", "rb") as f:
        G = pickle.load(f)
    graph_nodes = set(G.nodes)

    # ── 3. Mine ESCI Complement pairs → complement_of edges (CPU, encoder=None) ──
    print(f"[link] loading ESCI ({esci_dataset}) for Complement pairs ...")
    ds = load_dataset(esci_dataset, split="train")
    cols = ds.column_names
    keep = [c for c in ("query", "product_id", "esci_label", "product_locale",
                        "small_version") if c in cols]
    esci_df = ds.select_columns(keep).to_pandas()
    if "product_locale" in esci_df.columns:
        esci_df = esci_df[esci_df["product_locale"].astype(str).str.lower() == "us"]
    if "small_version" in esci_df.columns:
        esci_df = esci_df[esci_df["small_version"].astype(bool)]
    esci_df = esci_df[["query", "product_id", "esci_label"]].copy()
    esci_df["product_id"] = esci_df["product_id"].astype(str)
    esci_df["esci_label"] = esci_df["esci_label"].map(
        lambda x: _LABELS.get(str(x).strip().lower(), str(x))
    )

    # linked_db maps ESCI product_id -> {"asin": <graph node id>} for products in the graph.
    linked_db = {}
    for pid in esci_df["product_id"].unique():
        for cand in (pid, pid.strip().upper()):
            if cand in graph_nodes:
                linked_db[pid] = {"asin": cand}
                break
    print(f"[link] ESCI products resolvable to graph nodes: {len(linked_db):,}")

    before = G.number_of_edges()
    G = add_complement_edges(G, esci_df, linked_db, encoder=None)
    added = G.graph.get("complement_edges_added", 0)
    print(f"[link] complement_of edges added: {added:,}  "
          f"(edges {before:,} -> {G.number_of_edges():,})")

    # ── 4. Persist enriched graph + stats to sparda-artifacts ──
    os.makedirs("/sparda-artifacts", exist_ok=True)
    with open(ENRICHED_GRAPH, "wb") as f:
        pickle.dump(G, f)
    out = {**{k: v for k, v in stats.items() if k != "linked_db"},
           "n_esci_products_in_graph": len(linked_db),
           "complement_edges_added": int(added),
           "edges_before": int(before), "edges_after": int(G.number_of_edges())}
    with open(LINK_STATS, "w") as f:
        json.dump(out, f, indent=2)
    sparda_vol.commit()
    print(f"[link] wrote {ENRICHED_GRAPH} and {LINK_STATS}")
    return out


@app.function(image=image, volumes=VOLUMES, gpu="A100-80GB", timeout=2 * 60 * 60)
def e2e():
    """A100: run the §8.3 typed queries through the full SpardaPipeline + router accuracy."""
    import json

    from eval.rag_eval import format_router_table, router_accuracy
    from eval.test_queries import TEST_QUERIES

    from sparda_runtime import build_pipeline

    pipeline = build_pipeline(enriched_graph_path=ENRICHED_GRAPH)

    per_query = []
    for q in TEST_QUERIES:
        query = q["q"]
        try:
            res = pipeline.answer(query)
            per_query.append({
                "query": query,
                "expected_type": q["type"],
                "route": res["route"],
                "routing": res["routing"],
                "answer": res["answer"],
                "citations": res["citations"],
                "join_rate": res.get("join_rate"),
            })
            print(f"[e2e] {q['type']:>9} -> {res['route']:<9} | {query}")
        except Exception as exc:  # keep the run alive; record the failure
            per_query.append({
                "query": query, "expected_type": q["type"],
                "route": "error", "error": repr(exc),
            })
            print(f"[e2e] ERROR on {query!r}: {exc!r}")

    acc = router_accuracy(pipeline, TEST_QUERIES)
    print("\n" + format_router_table(acc))

    out = {"per_query": per_query, "router_accuracy": acc}
    with open(E2E_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    sparda_vol.commit()
    print(f"[e2e] wrote {E2E_JSON}")
    return {"n_queries": len(per_query), "router_accuracy": acc["accuracy"]}


@app.local_entrypoint()
def main(stage: str = "all"):
    """Orchestrate. stage: all | link | e2e."""
    if stage in ("all", "link"):
        print(link.remote())
    if stage in ("all", "e2e"):
        print(e2e.remote())
