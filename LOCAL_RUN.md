# Running SPARDA off Modal (fully local)

The Modal deployment was decommissioned on 2026-07-09. Everything needed to run the full
pipeline locally was exported to an offline archive first — this doc is the recipe.

**TL;DR:** retrieval (FAISS + BM25 + SPLADE + ColBERT) runs fine on a laptop CPU;
generation should come from the **Claude API arm** (no GPU needed) unless you have a
~60GB-VRAM GPU for the Qwen3-30B.

---

## 0. What you need

| thing | where |
|---|---|
| The artifact archive | `dmc-artifacts/` (exported 2026-07-09, ~4.1 GB — `dante/`, `vergil/`, `sparda/` + `MANIFEST.md`). Original home: the SanDisk-ULT drive. |
| The three repos | `github.com/DestroyorahSignus/{dante,vergil,sparda}` |
| Python | 3.11 (matches the Modal image) |
| RAM | ~8 GB free (graph pickle + FAISS + BM25 + encoders) |
| An Anthropic API key | for generation without a GPU (`console.anthropic.com`) |

First run also downloads ~1.5 GB of public models from HuggingFace (bge-small, SPLADE,
ColBERT — the bi-encoder itself loads from the archive).

## 1. Clone + install

```bash
mkdir dmc && cd dmc
git clone git@github.com:DestroyorahSignus/dante.git  dante-src
git clone git@github.com:DestroyorahSignus/vergil.git vergil-src
git clone git@github.com:DestroyorahSignus/sparda.git sparda

python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel is fine
pip install "transformers==4.57.6" "sentence-transformers==4.1.0" \
    "rerankers[transformers]==0.10.0" accelerate faiss-cpu rank-bm25 \
    scipy scikit-learn networkx rapidfuzz pandas pyarrow numpy anthropic fastapi uvicorn
pip install -e ./dante-src -e ./vergil-src                            # the two packages
```

> The pins mirror the Modal image (`sparda/modal_demo.py`) — transformers 5.x breaks the
> ColBERT reranker via `rerankers`, so stay on 4.57.x.

## 2. Point at the artifacts

Copy `dmc-artifacts/` off the drive (or use it in place) and note the paths. The layout
already matches what `sparda_runtime.dante_config` expects:

```
dmc-artifacts/
├── dante/    biencoder_gte_hn/  index_gte/  data/catalog.parquet  ...
├── vergil/   graph.pkl  summaries.json  summary_embeddings.npy  product_embeddings.npy ...
└── sparda/   enriched_graph.pkl
```

## 3. Build the pipeline (Claude generation — no GPU)

```python
# run from inside the sparda/ clone: python local_ask.py "your query"
import sys
sys.path.insert(0, ".")
from engine.claude_llm import ClaudeLLM          # needs ANTHROPIC_API_KEY in env
from sparda_runtime import build_pipeline

ART = "/path/to/dmc-artifacts"                    # <- edit me

pipe = build_pipeline(
    dante_artifacts=f"{ART}/dante",
    vergil_artifacts=f"{ART}/vergil",
    enriched_graph_path=f"{ART}/sparda/enriched_graph.pkl",
    llm=ClaudeLLM(),   # generation + router-fallback + entity extraction via Claude
)

result = pipe.answer(" ".join(sys.argv[1:]) or "wireless earbuds for running")
print(f"[{result['route']}]", result["answer"])
for c in result["citations"][:5]:
    print(" -", c["type"], c["name"][:80])
```

`export ANTHROPIC_API_KEY=sk-ant-...` first. Passing `llm=ClaudeLLM()` short-circuits the
Qwen load entirely — cold build is ~1-2 min (graph unpickle + index load + encoder
downloads on first run), then each answer costs one Claude call (~$0.02-0.05).

### Streaming / the web UI locally

The demo's FastAPI app is defined inside `modal_demo.py::web()` for Modal, but the pieces
are plain: serve `web/index.html` and wire `POST /api/query` to
`pipe.answer_stream(q, generator=...)` exactly as `modal_demo.py` does (copy the `gen()`
generator + `sse()` helper into a small `uvicorn` app). `web/index.html` works unchanged —
set `window.SPARDA_API` to `http://localhost:8000`.

## 4. Generation alternatives (if you don't want the Claude API)

| option | needs | how |
|---|---|---|
| Qwen3-30B-A3B via vLLM | ~60 GB VRAM GPU | omit `llm=` (default path); `pip install vllm`; model auto-downloads from HF |
| Qwen3-30B-A3B via HF | same VRAM, ~6 tok/s | `SPARDA_LLM_BACKEND=transformers` + omit `llm=` |
| Qwen3-4B GGUF (CPU/small GPU, offline) | ~3 GB | `pip install llama-cpp-python`, download `Qwen3-4B-Instruct-2507-GGUF` Q4_K_M, `llm=QwenLLM("/path/model.gguf", backend="llama_cpp")` (from `vergil.generation.llm`) |

## 5. Rebuilding anything that wasn't exported

The archive's `MANIFEST.md` maps every artifact to the Modal stage that rebuilds it
(`dante/modal_train.py`, `vergil/modal_build.py`, `sparda/modal_run.py`). The skipped
items (`index/`, `index_mbert/`, HF caches) rebuild from kept weights / re-download from HF.

## Troubleshooting

- **`ModuleNotFoundError: dante`** — `pip install -e ./dante-src` (the package dir is
  `dante-src/dante`; the editable install handles it).
- **Slow first query** — encoder downloads + BM25 unpickle; subsequent queries are seconds.
- **`Claude generator not configured`** — `ANTHROPIC_API_KEY` isn't set in the shell.
- **Wrong catalog/graph counts** — verify against MANIFEST.md: 351,961 catalog rows,
  65,570 nodes / 570,720 edges (enriched), 73 summaries.
