"""SPARDA — the DEPLOYED demo UI (a persistent, reachable *.modal.run URL).

================================================================================
DEPLOY  (the orchestrator runs this; produces the shareable URL)
================================================================================
    # 1) clone dante + vergil as SIBLINGS of this repo (private-repo-auth-free vendoring):
    #      git clone git@github.com:DestroyorahSignus/dante.git  ../dante-src
    #      git clone git@github.com:DestroyorahSignus/vergil.git ../vergil-src
    # 2) (optional but recommended) build the enriched graph first so multi-hop uses the
    #    SPARDA-only complement edges:  modal run modal_run.py --stage link
    # 3) deploy the persistent web app:
    #
    #      modal deploy modal_demo.py
    #
    #    → prints a persistent URL like  https://<workspace>--sparda-demo-web.modal.run

The image + vendoring + read-only artifact volumes are identical to ``modal_run.py``.
A single container builds the ``SpardaPipeline`` ONCE at container start (inside the ASGI
factory, NOT at import) — DanteSearchEngine + VERGIL graph + one shared Qwen3-4B — then
serves the Gradio Blocks UI (route badge + answer + citations + pyvis subgraph + SPLADE
expansion) via Modal's ASGI pattern. ``scaledown_window=300`` keeps the (expensive) warm
model resident for 5 min of idle before scaling to zero.
"""

import os

import modal

# ── Sibling package paths the orchestrator must clone before deploy (see header) ──
HERE = os.path.dirname(os.path.abspath(__file__))
DANTE_PKG = os.path.abspath(os.path.join(HERE, "..", "dante-src", "dante"))
VERGIL_PKG = os.path.abspath(os.path.join(HERE, "..", "vergil-src", "vergil"))

app = modal.App("sparda-demo")

# Same shared ML stack as modal_run.py (kept in sync deliberately).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.12.1",
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
        "fastapi>=0.110",              # ASGI host for gradio.mount_gradio_app
        "gradio>=4.0.0",
        "pyvis>=0.3.2",
    )
    .env({"HF_HOME": "/sparda-artifacts/hf", "TOKENIZERS_PARALLELISM": "false"})
    # VENDOR dante + vergil source (copy=True layers precede the runtime mount).
    .add_local_dir(DANTE_PKG, "/root/dante", copy=True)
    .add_local_dir(VERGIL_PKG, "/root/vergil", copy=True)
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
}


@app.function(image=image, volumes=VOLUMES, gpu="A100-80GB", scaledown_window=300,
              timeout=60 * 60)
@modal.concurrent(max_inputs=8)   # one warm model serves several UI requests concurrently
@modal.asgi_app()
def web():
    """ASGI factory: build the pipeline ONCE per container, then mount the Gradio UI."""
    from fastapi import FastAPI
    from gradio import mount_gradio_app

    from demo.app import create_demo

    from sparda_runtime import build_pipeline

    # Built at container start (inside the factory, not at import) — one shared model.
    pipeline = build_pipeline(enriched_graph_path="/sparda-artifacts/enriched_graph.pkl")

    demo = create_demo(pipeline)
    demo.queue(max_size=16)   # queue so concurrent requests don't drop

    fastapi_app = FastAPI()
    return mount_gradio_app(fastapi_app, demo, path="/")
