"""SPARDA LLM generator — ONE shared Qwen2.5-7B-Instruct Q4_K_M via llama.cpp.

See SPARDA_BUILD_PLAN.md §6.5.

This SpardaLLM is constructed ONCE and injected into the router (LLM fallback), the entity
extractor, and the answer generator — never reloaded per mode. On a 16GB T4 you cannot
afford a second 4.5GB resident model. Community summaries are generated offline (during
VERGIL's build, cached to disk — §1 artifacts), so at query time the only generation cost
is one router-classify call (≤4 tokens, only on ambiguous queries) plus one answer call.

The ``llama_cpp`` import is deferred into ``__init__`` so this module parses/imports even
when llama-cpp-python is not installed (e.g. CI / scaffold sandbox).
"""

from __future__ import annotations


class SpardaLLM:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        """
        Qwen2.5-7B-Instruct Q4_K_M via llama.cpp.
        ~4.5GB VRAM on T4 16GB. Fits alongside all encoder models.

        Download:
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
            qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/
        """
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                         n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        response = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]
