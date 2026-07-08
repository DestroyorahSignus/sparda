"""SPARDA demo helpers — subgraph JSON for the custom frontend's neuron-graph canvas.

(The Gradio UI that used to live here was replaced by the hand-built static frontend in
web/index.html + the FastAPI backend in modal_demo.py.)
"""

from demo.graph_viz import subgraph_data

__all__ = ["subgraph_data"]
