"""One-off: pre-download a HF model into the sparda-artifacts volume's HF cache.

The SPARDA demo (modal_demo.py) mounts all volumes READ-ONLY, so it can load models
from the volume's HF_HOME but cannot download new ones. Run this once (read-write) to
populate the cache, then the demo serves the model from it.

    modal run fetch_model.py                                   # default = Qwen3-30B-A3B
    modal run fetch_model.py --model-id Qwen/Qwen3-14B-Instruct-2507
"""
import modal

app = modal.App("sparda-fetch")

# HF_HOME must match the demo's (modal_demo.py .env HF_HOME=/sparda-artifacts/hf) so the
# download lands exactly where the read-only demo looks for it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.26", "hf_transfer>=0.1.8")
    .env({"HF_HOME": "/sparda-artifacts/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
vol = modal.Volume.from_name("sparda-artifacts", create_if_missing=True)


@app.function(image=image, volumes={"/sparda-artifacts": vol}, timeout=60 * 60, cpu=8.0)
def fetch(model_id: str):
    from huggingface_hub import snapshot_download

    print(f"[fetch] downloading {model_id} -> /sparda-artifacts/hf ...")
    path = snapshot_download(
        model_id,
        ignore_patterns=["*.gguf", "*.pth", "original/*"],  # skip non-transformers weights
    )
    vol.commit()
    print(f"[fetch] DONE — cached at {path}")
    return path


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
    print(fetch.remote(model_id))
