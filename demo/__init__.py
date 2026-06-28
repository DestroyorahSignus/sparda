"""SPARDA demo — Gradio UI (route badge + citations) + pyvis graph viz (§7)."""

from demo.app import create_demo
from demo.graph_viz import render_subgraph

__all__ = ["create_demo", "render_subgraph"]
