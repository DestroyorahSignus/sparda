"""SPARDA Gradio demo — chat + route badge + pyvis graph viz + SPLADE expansion.

See SPARDA_BUILD_PLAN.md §7.

Dark "devil-hunter" UI matching the portfolio dashboard: Dante-crimson / Vergil-steel /
Sparda-violet accents, system fonts (no external fetches — CSP-safe), styled route-badge
pills, a hero header, and tabbed side panels. The look is applied by overriding Gradio's
own CSS custom properties on ``.gradio-container`` (robust across gradio versions) plus a
handful of custom classes. The ``gradio`` import is deferred into ``create_demo`` so this
module imports even when gradio is absent (scaffold sandbox / CI).
"""

from __future__ import annotations

from demo.graph_viz import render_subgraph

# Accent palette — matches the dashboard artifact.
_CRIMSON = "#e0455c"   # DANTE (local / product search)
_STEEL = "#4d8ff0"     # VERGIL communities (global)
_VIOLET = "#9d6be0"    # VERGIL graph (multi_hop) / SPARDA

_CSS = """
.gradio-container {
  --body-background-fill: #0a0c11;
  --background-fill-primary: #12151d;
  --background-fill-secondary: #0f1219;
  --block-background-fill: #12151dcc;
  --block-border-color: #242a38;
  --block-label-text-color: #9aa3b2;
  --block-title-text-color: #e6e8ee;
  --body-text-color: #e6e8ee;
  --body-text-color-subdued: #8b93a4;
  --border-color-primary: #242a38;
  --border-color-accent: #9d6be0;
  --input-background-fill: #0e1118;
  --input-border-color: #2a3142;
  --button-primary-background-fill: linear-gradient(92deg, #9d6be0 0%, #4d8ff0 100%);
  --button-primary-background-fill-hover: linear-gradient(92deg, #ac7ef0 0%, #5c9bf5 100%);
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: #1a1f2b;
  --button-secondary-text-color: #cdd3df;
  --color-accent: #9d6be0;
  --color-accent-soft: #9d6be022;
  --link-text-color: #7fb0ff;
  color-scheme: dark;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background:
    radial-gradient(1100px 520px at 82% -8%, #9d6be01f, transparent 60%),
    radial-gradient(900px 480px at 6% 4%, #e0455c14, transparent 55%),
    #0a0c11 !important;
}
.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 { color: #eef0f5; }
code, kbd, .route-meta, .sparda-title, .chip { font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace; }

/* ---- hero ---- */
.sparda-hero {
  border: 1px solid #242a38;
  border-radius: 16px;
  padding: 22px 26px;
  margin-bottom: 6px;
  background:
    linear-gradient(180deg, #14171fcc, #0d0f15cc),
    radial-gradient(700px 200px at 100% 0%, #4d8ff01f, transparent 70%);
  box-shadow: 0 1px 0 #ffffff0a inset, 0 18px 40px -30px #9d6be066;
}
.sparda-title {
  font-size: 30px; font-weight: 800; letter-spacing: 2px; margin: 0;
  background: linear-gradient(92deg, #e0455c, #9d6be0 55%, #4d8ff0);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.sparda-tag { color: #9aa3b2; font-size: 14px; margin: 6px 0 14px; max-width: 760px; }
.sparda-legend { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  font-size: 11px; letter-spacing: .4px; padding: 5px 11px; border-radius: 999px;
  border: 1px solid; background: #ffffff08; white-space: nowrap;
}
.chip-local  { color: #ff8a98; border-color: #e0455c66; }
.chip-global { color: #9dc2ff; border-color: #4d8ff066; }
.chip-multi  { color: #c6a6f5; border-color: #9d6be066; }

/* ---- route badge (rendered inside chat markdown) ---- */
.route-badge {
  display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .5px;
  padding: 3px 10px; border-radius: 999px; color: #fff; text-transform: uppercase;
}
.route-local  { background: linear-gradient(92deg,#e0455c,#b5303f); }
.route-global { background: linear-gradient(92deg,#4d8ff0,#2f6fd0); }
.route-multi  { background: linear-gradient(92deg,#9d6be0,#7346c4); }
.route-meta { color: #7d8697; font-size: 11px; margin-left: 8px; }

/* ---- panels ---- */
.panel { border: 1px solid #242a38 !important; border-radius: 14px !important; padding: 12px !important; }
#sparda-chat { border-radius: 12px; }
.gradio-container .tab-nav button.selected { color: #c6a6f5; border-bottom-color: #9d6be0; }
footer { display: none !important; }
"""

_HERO = """
<div class="sparda-hero">
  <div class="sparda-title">⚔ SPARDA</div>
  <div class="sparda-tag">One router over two retrieval engines — it reads your query and routes to
    <b>DANTE</b> (multi-signal hybrid product search) or <b>VERGIL</b> (a GraphRAG product knowledge graph),
    then answers with grounded citations.</div>
  <div class="sparda-legend">
    <span class="chip chip-local">DANTE · product search</span>
    <span class="chip chip-global">VERGIL · market clusters</span>
    <span class="chip chip-multi">VERGIL · graph traversal</span>
  </div>
</div>
"""


def create_demo(pipeline: "SpardaPipeline"):
    import gradio as gr

    _CLASS = {"local": "route-local", "global": "route-global", "multi_hop": "route-multi"}
    _LABEL = {"local": "DANTE search", "global": "VERGIL communities", "multi_hop": "VERGIL graph"}

    def chat(query, history):
        # Streaming generator (keeps the SSE alive during 30B gen → no 'connection lost') that
        # emits an HTML route-badge pill + live tokens. Messages format for gradio 6.
        history = history + [{"role": "user", "content": query},
                             {"role": "assistant", "content": "_routing + retrieving…_"}]
        yield "", history

        badge = None
        acc, citations = "", []
        for acc, decision, citations, _ctx in pipeline.answer_stream(query):
            if badge is None:
                cls = _CLASS.get(decision.route, "route-local")
                lbl = _LABEL.get(decision.route, decision.route)
                badge = (f'<span class="route-badge {cls}">{lbl}</span>'
                         f'<span class="route-meta">{decision.method} · conf '
                         f'{decision.confidence:.2f}</span>')
            history[-1]["content"] = f"{badge}\n\n{acc}▌"
            yield "", history

        if badge is None:
            history[-1]["content"] = "_(no response generated)_"
            yield "", history
            return
        cites = "\n".join(f"- `{c['type']}` {c['name']} — {c['evidence']}"
                          for c in citations[:10])  # show all retrieved so the cited #N is visible
        history[-1]["content"] = (f"{badge}\n\n{acc}\n\n"
                                  f"<details><summary>citations</summary>\n\n{cites}\n</details>")
        yield "", history

    def show_graph(query):
        return render_subgraph(pipeline, query)

    def show_splade(query):
        expansion = pipeline.dante.splade.visualize_expansion(query, top_k_terms=15)
        return {term: round(weight, 2) for term, weight in expansion}

    # Dark theme skeleton; the CSS var-overrides above do the heavy styling. System fonts only.
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SF Mono", "Menlo", "monospace"],
    )

    with gr.Blocks(title="SPARDA", theme=theme, css=_CSS, fill_height=True) as demo:
        gr.HTML(_HERO)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_classes="panel"):
                chatbot = gr.Chatbot(height=520, show_label=False, elem_id="sparda-chat",
                                     avatar_images=None, sanitize_html=False)
                with gr.Row():
                    query = gr.Textbox(
                        placeholder="Ask about products… e.g. 'wireless noise cancelling headphones under $300'",
                        show_label=False, scale=8, container=False)
                    btn = gr.Button("⚔ Search", variant="primary", scale=1, min_width=120)
                with gr.Row():
                    clear = gr.ClearButton([chatbot, query], value="Clear", size="sm")
                gr.Examples(
                    examples=[
                        "best wireless noise cancelling headphones under $300",
                        "What accessories from Sony work with the WH-1000XM5?",
                        "Compare the smart home ecosystems — Alexa vs Google Home",
                        "USB-C chargers from Anker that people buy with MacBooks",
                        "What are the trends in wireless earbuds?",
                        "Find a portable speaker similar to JBL Flip but a different brand",
                    ],
                    inputs=query, label="Try one",
                )

            with gr.Column(scale=2, elem_classes="panel"):
                with gr.Tabs():
                    with gr.Tab("Knowledge graph"):
                        graph_html = gr.HTML()
                    with gr.Tab("SPLADE expansion"):
                        splade_json = gr.JSON(label=None, show_label=False)

        # Search on click AND on Enter. The three handlers per trigger fire in parallel and
        # each reads `query` at trigger time (before `chat` clears it), so graph/splade see the
        # real query — same pattern as the original, plus keyboard submit.
        for trigger in (btn.click, query.submit):
            trigger(chat, [query, chatbot], [query, chatbot])
            trigger(show_graph, [query], graph_html)
            trigger(show_splade, [query], splade_json)

    return demo
