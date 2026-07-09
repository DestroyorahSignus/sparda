"""SPARDA — the DEPLOYED demo (a persistent, reachable *.modal.run URL).

A hand-built single-page frontend (web/index.html — minimalist, constellation-particle
background, streaming) served by a FastAPI backend that exposes a small JSON/SSE API:
  GET  /              → the static UI
  POST /api/query     → Server-Sent-Events stream (route meta → answer token deltas → citations)
  GET  /api/graph?q=  → {nodes, edges} JSON for the frontend's neuron-graph canvas
  GET  /api/splade?q= → {term: weight}
  GET  /api/health    → warm check

Deploy:  (clone dante + vergil as siblings ../dante-src ../vergil-src first)
    modal deploy modal_demo.py   → https://<workspace>--sparda-demo-web.modal.run

The pipeline (DanteSearchEngine + VERGIL graph + one shared Qwen3-30B-A3B) is LAZY-loaded on
the first /api/query, NOT at container start — so the page itself serves fast; only the first
query pays the model cold-load. ``scaledown_window=600`` releases the A100 after 10 min of
idle — closing the tab sends no signal to Modal, so a longer warm-hold just burns idle GPU
on a demo that rarely gets two visitors inside the window. The frontend is also
Vercel-deployable as-is (static): host web/ on a CDN and set ``window.SPARDA_API`` to this
Modal URL (CORS is open). Image/vendoring/read-only volumes match ``modal_run.py``.
"""

import os

import modal

# ── Sibling package paths the orchestrator must clone before deploy (see header) ──
HERE = os.path.dirname(os.path.abspath(__file__))
DANTE_PKG = os.path.abspath(os.path.join(HERE, "..", "dante-src", "dante"))
VERGIL_PKG = os.path.abspath(os.path.join(HERE, "..", "vergil-src", "vergil"))

app = modal.App("sparda-demo")

# Same shared ML stack as modal_run.py — EXCEPT: this serving image adds vLLM (10x the
# decode speed of HF .generate() on the 30B-A3B; HF crawled at ~5-10 tok/s) and drops the
# explicit torch pin so vLLM installs its own tightly-pinned torch build.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.11.0",                # brings its own torch; QwenLLM(backend='vllm')
        "transformers==4.57.6",
        "sentence-transformers==4.1.0",
        "rerankers[transformers]==0.10.0",
        "accelerate==1.14.0",
        "faiss-cpu==1.14.3",
        "rank-bm25==0.2.2",
        "scipy==1.17.1",
        "scikit-learn>=1.3",           # engine/global_search imports sklearn at module top
        "networkx>=3.2",
        "cdlib>=0.4.0",
        "leidenalg>=0.10.0",
        "python-igraph>=0.11.0",
        "rapidfuzz>=3.6.0",
        "pandas==3.0.3",
        "pyarrow>=17,<21",             # parquet engine for pd.read_parquet(catalog) — the
                                       # e2e image got it transitively via datasets; the demo
                                       # image has no datasets, so it must be explicit (else
                                       # the web container crash-loops on boot: ImportError)
        "numpy==2.2.6",
        "fastapi>=0.110",              # the demo API host (static page + JSON/SSE endpoints)
        "anthropic>=0.116",            # Claude comparison arm (engine/claude_llm.py)
    )
    .env({"HF_HOME": "/sparda-artifacts/hf", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_CACHE_ROOT": "/vllm-cache"})
    # VENDOR dante + vergil source (copy=True layers precede the runtime mount).
    .add_local_dir(DANTE_PKG, "/root/dante", copy=True)
    .add_local_dir(VERGIL_PKG, "/root/vergil", copy=True)
    .add_local_dir(os.path.join(HERE, "web"), "/root/web", copy=True)   # the static frontend
    .add_local_python_source("engine", "eval", "data", "demo", "sparda_runtime")
)


def _read_only(vol):
    """Read-only handle if the installed Modal supports it, else the volume (never committed)."""
    ro = getattr(vol, "read_only", None)
    return ro() if callable(ro) else vol


VOLUMES = {
    "/dante-artifacts": _read_only(modal.Volume.from_name("dante-artifacts", create_if_missing=True)),
    "/vergil-artifacts": _read_only(modal.Volume.from_name("vergil-artifacts", create_if_missing=True)),
    "/sparda-artifacts": _read_only(modal.Volume.from_name("sparda-artifacts", create_if_missing=True)),
    # READ-WRITE: vLLM writes its torch.compile artifacts here (~/.cache/vllm is
    # symlinked to it via VLLM_CACHE_ROOT below); only the first-ever cold start pays
    # the full compile, later cold starts reuse the cache (committed on scaledown).
    "/vllm-cache": modal.Volume.from_name("sparda-vllm-cache", create_if_missing=True),
}


# Personal Anthropic key for the Claude arm. An EMPTY value keeps Claude cleanly
# disabled (build_pipeline checks for a non-empty ANTHROPIC_API_KEY); set the real
# key with:  modal secret create anthropic-personal ANTHROPIC_API_KEY=sk-ant-...
@app.function(image=image, volumes=VOLUMES, gpu="A100-80GB", scaledown_window=600,
              timeout=60 * 60,
              secrets=[modal.Secret.from_name("anthropic-personal")])
@modal.concurrent(max_inputs=8)   # one warm model serves several requests concurrently
@modal.asgi_app()
def web():
    """ASGI factory: a FastAPI app serving the static frontend + a JSON/SSE query API.

    The heavy pipeline is LAZY-loaded on first /api/query (thread-safe singleton), so the
    page itself serves fast — only the first query pays the model cold-load.
    """
    import json
    import pathlib
    import threading

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    from demo.graph_viz import subgraph_data
    from sparda_runtime import build_pipeline

    _cache: dict = {}
    _lock = threading.Lock()

    def get_pipeline():
        if "p" not in _cache:
            with _lock:
                if "p" not in _cache:
                    _cache["p"] = build_pipeline(
                        enriched_graph_path="/sparda-artifacts/enriched_graph.pkl")
        return _cache["p"]

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    index_html = pathlib.Path("/root/web/index.html").read_text()

    api = FastAPI(title="SPARDA")
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])  # open CORS so a Vercel/CDN frontend can call it

    @api.get("/")
    def index():
        return HTMLResponse(index_html)

    @api.get("/api/health")
    def health():
        import os as _os
        return {"ok": True, "warm": "p" in _cache,
                "claude": bool(_os.environ.get("ANTHROPIC_API_KEY"))}

    @api.post("/api/query")
    async def query(req: Request):
        # Guarded parse: a malformed body ('not json') or a valid-but-non-dict body
        # ('[]', '"hi"') must yield the graceful SSE error below, not a raw HTTP 500
        # (the API is public/CORS-open). Also bound the query length.
        try:
            body = await req.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            body = {}
        q = (str(body.get("query") or "")).strip()[:2000]
        generator = str(body.get("generator") or "local")
        if generator not in ("local", "claude"):
            generator = "local"

        def gen():
            if not q:
                yield sse("error", {"message": "empty query"})
                return
            # Fail fast BEFORE the pipeline cold-load: a Compare click shouldn't wait
            # minutes just to learn the Claude arm has no key configured.
            import os as _os
            if generator == "claude" and not _os.environ.get("ANTHROPIC_API_KEY"):
                yield sse("error", {"message": "Claude arm not configured — set the "
                                    "anthropic-personal Modal secret (ANTHROPIC_API_KEY)"})
                return
            try:
                pipe = get_pipeline()
                sent_meta = False
                acc, citations, decision = "", [], None
                sent_len = 0
                for acc, decision, citations, _ctx in pipe.answer_stream(q, generator=generator):
                    if not sent_meta and decision is not None:
                        yield sse("meta", {"route": decision.route, "method": decision.method,
                                           "confidence": round(decision.confidence, 2),
                                           "generator": generator})
                        sent_meta = True
                    # Send only the NEW text: re-sending the full accumulated answer on
                    # every token made the stream O(n^2) bytes over a long generation.
                    if len(acc) > sent_len:
                        yield sse("token", {"delta": acc[sent_len:]})
                        sent_len = len(acc)
                yield sse("cite", {"citations": [
                    {"type": c["type"], "name": c["name"], "evidence": c["evidence"]}
                    for c in (citations or [])[:10]]})
                yield sse("done", {})
            except Exception as exc:  # surface errors to the client instead of a dead stream
                yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})

        return StreamingResponse(gen(), media_type="text/event-stream")

    @api.get("/api/graph")
    def graph(q: str = ""):
        if not q.strip():
            return JSONResponse({"nodes": [], "edges": []})
        try:
            return JSONResponse(subgraph_data(get_pipeline(), q))
        except Exception:
            return JSONResponse({"nodes": [], "edges": []})

    @api.get("/api/splade")
    def splade(q: str = ""):
        if not q.strip():
            return JSONResponse({})
        try:
            exp = get_pipeline().dante.splade.visualize_expansion(q, top_k_terms=15)
            return JSONResponse({t: round(w, 2) for t, w in exp})
        except Exception:
            return JSONResponse({})

    return api
