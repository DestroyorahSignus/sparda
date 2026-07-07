"""SPARDA Gradio demo — chat + route badge + pyvis graph viz + SPLADE expansion.

See SPARDA_BUILD_PLAN.md §7.

The ``gradio`` import is deferred into ``create_demo`` so this module parses/imports even
when gradio is not installed (scaffold sandbox / CI).
"""

from __future__ import annotations

from demo.graph_viz import render_subgraph


def create_demo(pipeline: "SpardaPipeline"):
    import gradio as gr

    def chat(query, history):
        # STREAMING generator: yield partial answers as the LLM produces tokens. This keeps
        # the Gradio SSE connection fed with data throughout the ~15-25s 30B generation —
        # otherwise the idle connection is dropped by the proxy ("connection lost"). Also
        # much better UX. Messages format (role/content dicts) for gradio 6.
        badges = {"local": "🔍 DANTE search", "global": "🌐 VERGIL communities",
                  "multi_hop": "🔗 VERGIL graph traversal"}
        history = history + [{"role": "user", "content": query},
                             {"role": "assistant", "content": "_🔍 routing + retrieving…_"}]
        yield "", history

        badge = why = None
        acc, citations = "", []
        for acc, decision, citations, _ctx in pipeline.answer_stream(query):
            if badge is None:
                badge = badges.get(decision.route, decision.route)
                why = f"_{decision.method} · conf {decision.confidence:.2f} · {decision.reason}_"
            history[-1]["content"] = f"**{badge}**  {why}\n\n{acc}▌"
            yield "", history

        if badge is None:  # stream produced nothing
            history[-1]["content"] = "_(no response generated)_"
            yield "", history
            return
        cites = "\n".join(f"- `{c['type']}` {c['name']} — {c['evidence']}"
                          for c in citations[:6])
        history[-1]["content"] = (f"**{badge}**  {why}\n\n{acc}\n\n"
                                  f"<details><summary>citations</summary>\n\n{cites}\n</details>")
        yield "", history

    def show_graph(query):
        """Render relevant subgraph as interactive pyvis HTML."""
        return render_subgraph(pipeline, query)

    def show_splade(query):
        """Show SPLADE term expansion for interpretability."""
        expansion = pipeline.dante.splade.visualize_expansion(query, top_k_terms=15)
        return {term: round(weight, 2) for term, weight in expansion}

    with gr.Blocks(title="SPARDA", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚔️ SPARDA — Hybrid Search + GraphRAG Product Discovery")
        gr.Markdown("Product search → DANTE stack | Relational queries → VERGIL graph | Market analysis → VERGIL communities")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=450, label="Chat")  # gradio 6 = messages-only, no type=
                query = gr.Textbox(placeholder="Try: 'What accessories from Sony work with the WH-1000XM5?'",
                                   label="Query")
                with gr.Row():
                    btn = gr.Button("Search", variant="primary")
                    clear = gr.ClearButton([chatbot, query])

            with gr.Column(scale=2):
                gr.Markdown("### Knowledge graph")
                graph_html = gr.HTML(label="Subgraph")
                gr.Markdown("### SPLADE expansion")
                splade_json = gr.JSON(label="Term weights")

        btn.click(chat, [query, chatbot], [query, chatbot])
        btn.click(show_graph, [query], graph_html)
        btn.click(show_splade, [query], splade_json)

        gr.Examples([
            "best wireless noise cancelling headphones under $300",
            "What accessories from Sony work with the WH-1000XM5?",
            "Compare the smart home ecosystems — Alexa vs Google Home",
            "USB-C chargers from Anker that people buy with MacBooks",
            "What are the trends in wireless earbuds?",
            "Find a portable speaker similar to JBL Flip but from a different brand",
        ], query)

    return demo
