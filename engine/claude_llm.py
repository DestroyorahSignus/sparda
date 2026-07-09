"""Claude API generator — the demo's frontier-model comparison arm.

Same ``generate(prompt, max_tokens, temperature) -> str`` / ``generate_stream(...)``
contract as ``vergil.generation.llm.QwenLLM``, so ``SpardaPipeline`` can swap
generators per query (the "SLM vs Claude" demo toggle): routing + retrieval are
identical, only the generation step differs.

Notes:
- Credentials come from the environment (``ANTHROPIC_API_KEY`` via a Modal secret);
  ``anthropic.Anthropic()`` resolves them itself — never hardcode a key.
- Claude Opus 4.8 removed sampling parameters (``temperature``/``top_p``/``top_k``
  return a 400), so the ``temperature`` argument is accepted for interface
  compatibility and deliberately NOT forwarded. Grounding lives in the prompt.
- Adaptive thinking is set explicitly (on Opus 4.8, omitting ``thinking`` runs
  WITHOUT thinking); only text deltas are forwarded to the caller, so any
  thinking happens silently before the first visible token.
"""

from __future__ import annotations

MODEL = "claude-opus-4-8"


class ClaudeLLM:
    """Minimal Claude generator with the QwenLLM call shape (see module docstring)."""

    def __init__(self, model: str = MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()  # key from env: ANTHROPIC_API_KEY

    def generate_stream(self, prompt: str, max_tokens: int = 800,
                        temperature: float | None = None):
        """Yield the assistant text incrementally (token chunks).

        ``temperature`` is ignored — removed on Opus 4.8 (400 if sent).
        """
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def generate(self, prompt: str, max_tokens: int = 800,
                 temperature: float | None = None) -> str:
        """Single-turn completion; returns the stripped assistant text."""
        return "".join(self.generate_stream(prompt, max_tokens=max_tokens)).strip()
