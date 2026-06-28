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
        result = pipeline.answer(query)
        route = result["route"]
        routing = result["routing"]
        answer = result["answer"]

        badge = {"local": "🔍 DANTE search", "global": "🌐 VERGIL communities",
                 "multi_hop": "🔗 VERGIL graph traversal"}[route]
        # SHOW the routing decision: heuristic vs LLM fallback vs degraded, and why.
        why = f"_{routing['method']} · conf {routing['confidence']:.2f} · {routing['reason']}_"
        cites = "\n".join(f"- `{c['type']}` {c['name']} — {c['evidence']}"
                          for c in result["citations"][:6])
        formatted = (f"**{badge}**  {why}\n\n{answer}\n\n"
                     f"<details><summary>citations</summary>\n\n{cites}\n</details>")
        history.append((query, formatted))
        return "", history

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
                chatbot = gr.Chatbot(height=450, label="Chat")
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
