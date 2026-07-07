"""SPARDA Gradio demo — chat + route badge + pyvis graph viz + SPLADE expansion.

See SPARDA_BUILD_PLAN.md §7.

UI-v2: minimalist, type-forward, monochrome (near-black + a single violet accent), with an
interactive constellation-particle background and a polished "command bar" for the prompt.
The particle canvas is injected via ``gr.Blocks(head=...)`` (a raw <script> — gradio's
gr.HTML does NOT execute scripts); the look is CSS via ``css=`` (overriding Gradio's CSS
custom properties on ``.gradio-container`` + custom classes). System fonts only (CSP-safe,
no external fetches). The ``gradio`` import is deferred into ``create_demo`` so this module
imports even when gradio is absent (scaffold sandbox / CI).
"""

from __future__ import annotations

from demo.graph_viz import render_subgraph

_ACCENT = "139,92,246"   # violet #8b5cf6 (the single accent — RGB for rgba())

# --- interactive constellation-particle background (vanilla JS, no libs) ------------------
_HEAD = """
<script>
(function () {
  if (window.__spardaParticles) return;
  window.__spardaParticles = true;
  var ACCENT = '""" + _ACCENT + """';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function init() {
    var cv = document.createElement('canvas');
    cv.id = 'sparda-particles';
    cv.style.cssText = 'position:fixed;inset:0;z-index:-1;pointer-events:none;';
    document.body.appendChild(cv);
    var ctx = cv.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 0, H = 0, parts = [], mouse = { x: -9999, y: -9999 };
    var LINK, MOUSE_R;

    function resize() {
      W = cv.width = Math.floor(innerWidth * dpr);
      H = cv.height = Math.floor(innerHeight * dpr);
      cv.style.width = innerWidth + 'px';
      cv.style.height = innerHeight + 'px';
      LINK = 130 * dpr; MOUSE_R = 170 * dpr;
      var target = Math.max(28, Math.min(90, Math.floor(innerWidth * innerHeight / 16000)));
      parts = [];
      for (var i = 0; i < target; i++) {
        parts.push({ x: Math.random() * W, y: Math.random() * H,
                     vx: (Math.random() - 0.5) * 0.28 * dpr,
                     vy: (Math.random() - 0.5) * 0.28 * dpr });
      }
    }

    function frame() {
      ctx.clearRect(0, 0, W, H);
      var i, j, p;
      for (i = 0; i < parts.length; i++) {
        p = parts[i];
        p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x += W; if (p.x > W) p.x -= W;
        if (p.y < 0) p.y += H; if (p.y > H) p.y -= H;
        if (mouse.x > -9999) {                              // gentle repulsion from cursor
          var dx = p.x - mouse.x, dy = p.y - mouse.y, d2 = dx * dx + dy * dy;
          if (d2 < MOUSE_R * MOUSE_R && d2 > 1) {
            var d = Math.sqrt(d2), f = (MOUSE_R - d) / MOUSE_R * 0.7;
            p.x += dx / d * f; p.y += dy / d * f;
          }
        }
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.4 * dpr, 0, 6.283);
        ctx.fillStyle = 'rgba(' + ACCENT + ',0.5)'; ctx.fill();
      }
      for (i = 0; i < parts.length; i++) {                  // links form/break by distance
        for (j = i + 1; j < parts.length; j++) {
          var a = parts[i], b = parts[j], ex = a.x - b.x, ey = a.y - b.y, e2 = ex * ex + ey * ey;
          if (e2 < LINK * LINK) {
            ctx.strokeStyle = 'rgba(' + ACCENT + ',' + (1 - Math.sqrt(e2) / LINK) * 0.16 + ')';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
          }
        }
        if (mouse.x > -9999) {                              // lines reach toward the cursor
          var mx = parts[i].x - mouse.x, my = parts[i].y - mouse.y, m2 = mx * mx + my * my;
          if (m2 < MOUSE_R * MOUSE_R) {
            ctx.strokeStyle = 'rgba(' + ACCENT + ',' + (1 - Math.sqrt(m2) / MOUSE_R) * 0.34 + ')';
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(parts[i].x, parts[i].y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke();
          }
        }
      }
    }

    var running = true;
    function loop() { if (running) frame(); requestAnimationFrame(loop); }
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', function (e) { mouse.x = e.clientX * dpr; mouse.y = e.clientY * dpr; });
    window.addEventListener('mouseout', function () { mouse.x = mouse.y = -9999; });
    document.addEventListener('visibilitychange', function () { running = !document.hidden; });
    resize();
    if (reduce) frame(); else loop();                       // reduced-motion → one static frame
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
</script>
"""

_CSS = """
:root { --acc: rgb(139,92,246); }
html, body { background: #08090c !important; }
.gradio-container {
  --body-background-fill: transparent;
  --background-fill-primary: transparent;
  --background-fill-secondary: transparent;
  --block-background-fill: rgba(255,255,255,0.028);
  --block-border-color: rgba(255,255,255,0.08);
  --block-label-text-color: #7a8090;
  --block-title-text-color: #e8eaf0;
  --body-text-color: #e8eaf0;
  --body-text-color-subdued: #7a8090;
  --border-color-primary: rgba(255,255,255,0.08);
  --border-color-accent: rgba(139,92,246,0.55);
  --input-background-fill: rgba(255,255,255,0.02);
  --input-border-color: rgba(255,255,255,0.10);
  --button-primary-background-fill: rgba(139,92,246,0.9);
  --button-primary-background-fill-hover: rgba(155,110,255,1);
  --button-primary-text-color: #ffffff;
  --button-secondary-background-fill: rgba(255,255,255,0.04);
  --button-secondary-text-color: #cdd3df;
  --color-accent: rgb(139,92,246);
  --color-accent-soft: rgba(139,92,246,0.16);
  --link-text-color: #b9a4f5;
  color-scheme: dark;
  background: transparent !important;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  max-width: 1180px; margin: 0 auto;
}
code, kbd, .route-meta, .wordmark, .kbd-hint { font-family: ui-monospace, "SF Mono", Menlo, monospace; }

/* glass panels */
.panel {
  background: rgba(255,255,255,0.028) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 16px !important;
  backdrop-filter: blur(14px) saturate(120%); -webkit-backdrop-filter: blur(14px) saturate(120%);
  padding: 14px !important;
}

/* hero */
.hero { padding: 26px 4px 10px; }
.wordmark {
  font-size: 34px; font-weight: 700; letter-spacing: 8px; margin: 0; color: #f3f4f8;
  text-shadow: 0 0 24px rgba(139,92,246,0.35);
}
.wordmark::after { content:""; display:block; width:54px; height:2px; margin-top:10px;
  background: linear-gradient(90deg, var(--acc), transparent); }
.hero .sub { color: #8a90a0; font-size: 13.5px; line-height: 1.6; margin-top: 12px; max-width: 720px; font-weight: 300; }
.hero .sub b { color: #cfd3de; font-weight: 500; }

/* command bar — the polished prompt dashboard */
.command-bar {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 16px !important;
  padding: 8px 8px 8px 18px !important;
  backdrop-filter: blur(16px) saturate(130%); -webkit-backdrop-filter: blur(16px) saturate(130%);
  box-shadow: 0 20px 50px -34px rgba(139,92,246,0.5);
  transition: border-color .25s, box-shadow .25s;
}
.command-bar:focus-within {
  border-color: rgba(139,92,246,0.6) !important;
  box-shadow: 0 0 0 3px rgba(139,92,246,0.18), 0 20px 60px -30px rgba(139,92,246,0.6);
}
.command-bar textarea, .command-bar input {
  background: transparent !important; border: none !important; box-shadow: none !important;
  font-size: 16px !important; color: #eef0f5 !important;
}
.command-bar textarea::placeholder { color: #6b7180 !important; }
.command-bar .gr-button, .command-bar button {
  border-radius: 12px !important; font-weight: 600 !important; letter-spacing: .3px;
}
.kbd-hint { color: #5f6572; font-size: 11px; margin: 8px 2px 2px; letter-spacing: .3px; }

/* example chips */
.chip-row { gap: 8px !important; flex-wrap: wrap !important; margin-top: 4px; }
.chip button, button.chip {
  background: rgba(255,255,255,0.035) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  color: #aab0c0 !important; border-radius: 999px !important;
  font-size: 12px !important; font-weight: 400 !important; padding: 6px 13px !important;
  min-width: 0 !important; transition: transform .15s, border-color .2s, color .2s, background .2s;
}
.chip button:hover, button.chip:hover {
  transform: translateY(-1px); color: #e8eaf0 !important;
  border-color: rgba(139,92,246,0.5) !important; background: rgba(139,92,246,0.10) !important;
}

/* route badge in chat (minimal, single violet accent) */
.route-badge {
  display:inline-block; font-size:10px; font-weight:600; letter-spacing:.7px;
  padding:3px 10px; border-radius:999px; text-transform:uppercase;
  color:#c9b8f5; background:rgba(139,92,246,0.14); border:1px solid rgba(139,92,246,0.35);
}
.route-meta { color:#6b7180; font-size:11px; margin-left:8px; }

/* chat + tabs */
#sparda-chat { border: none !important; background: transparent !important; }
.gradio-container .tab-nav button.selected { color:#c9b8f5; border-bottom-color: var(--acc); }
.gradio-container .tab-nav button { color:#7a8090; }
footer { display:none !important; }

/* animations */
@keyframes fadeInUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform:none; } }
@keyframes glowPulse { 0%,100% { opacity:.35; } 50% { opacity:1; } }
.hero { animation: fadeInUp .5s ease both; }
.command-bar { animation: fadeInUp .5s ease .06s both; }
.chip-row { animation: fadeInUp .5s ease .12s both; }
.main-row { animation: fadeInUp .5s ease .18s both; }
#sparda-chat .message, #sparda-chat .bubble, #sparda-chat [class*="message"] { animation: fadeInUp .3s ease both; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
"""

_HERO = """
<div class="hero">
  <div class="wordmark">SPARDA</div>
  <div class="sub">A query router over two retrieval engines — <b>DANTE</b> multi-signal hybrid
    product search &amp; <b>VERGIL</b> GraphRAG knowledge graph. Grounded, cited answers over the
    <b>Amazon&nbsp;ESCI</b> research catalog (~352K products), ranked by relevance — not reviews or
    sales, so "best"-style queries surface the closest matches, not curated picks.</div>
</div>
"""

# (label shown on the chip, full query it inserts)
_EXAMPLES = [
    ("NC headphones < $300", "wireless noise cancelling headphones under $300"),
    ("Sony WH-1000XM5 accessories", "What accessories from Sony work with the WH-1000XM5?"),
    ("Alexa vs Google Home", "Compare the smart home ecosystems — Alexa vs Google Home"),
    ("Anker chargers + MacBook", "USB-C chargers from Anker that people buy with MacBooks"),
    ("wireless earbud trends", "What are the trends in wireless earbuds?"),
    ("JBL Flip alternative", "Find a portable speaker similar to JBL Flip but a different brand"),
]


def create_demo(pipeline: "SpardaPipeline"):
    import gradio as gr

    _LABEL = {"local": "DANTE search", "global": "VERGIL communities", "multi_hop": "VERGIL graph"}

    def chat(query, history):
        # Streaming generator (keeps the SSE alive during 30B gen → no 'connection lost') that
        # emits a minimal route-badge pill + live tokens. Messages format for gradio 6.
        history = history + [{"role": "user", "content": query},
                             {"role": "assistant", "content": "_routing + retrieving…_"}]
        yield "", history

        badge = None
        acc, citations = "", []
        for acc, decision, citations, _ctx in pipeline.answer_stream(query):
            if badge is None:
                lbl = _LABEL.get(decision.route, decision.route)
                badge = (f'<span class="route-badge">{lbl}</span>'
                         f'<span class="route-meta">{decision.method} · conf '
                         f'{decision.confidence:.2f}</span>')
            history[-1]["content"] = f"{badge}\n\n{acc}▌"
            yield "", history

        if badge is None:
            history[-1]["content"] = "_(no response generated)_"
            yield "", history
            return
        cites = "\n".join(f"- `{c['type']}` {c['name']} — {c['evidence']}"
                          for c in citations[:10])
        history[-1]["content"] = (f"{badge}\n\n{acc}\n\n"
                                  f"<details><summary>citations</summary>\n\n{cites}\n</details>")
        yield "", history

    def show_graph(query):
        return render_subgraph(pipeline, query)

    def show_splade(query):
        expansion = pipeline.dante.splade.visualize_expansion(query, top_k_terms=15)
        return {term: round(weight, 2) for term, weight in expansion}

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SF Mono", "Menlo", "monospace"],
    )

    with gr.Blocks(title="SPARDA", theme=theme, css=_CSS, head=_HEAD, fill_height=True) as demo:
        gr.HTML(_HERO)

        # ── the polished command bar (prompt dashboard) ──
        with gr.Group(elem_classes="command-bar"):
            with gr.Row(equal_height=True):
                query = gr.Textbox(
                    placeholder="Ask about products…  e.g. wireless noise cancelling headphones under $300",
                    show_label=False, container=False, scale=9, lines=1, max_lines=3, autofocus=True)
                btn = gr.Button("Search", variant="primary", scale=1, min_width=110)
        gr.HTML('<div class="kbd-hint">⏎ to search · routed automatically to DANTE or VERGIL</div>')

        # ── example chips (clickable → fill the box) ──
        with gr.Row(elem_classes="chip-row"):
            for _lbl, _q in _EXAMPLES:
                gr.Button(_lbl, size="sm", elem_classes="chip").click(
                    lambda q=_q: q, None, query)

        # ── results: chat + side panels ──
        with gr.Row(elem_classes="main-row", equal_height=False):
            with gr.Column(scale=3, elem_classes="panel"):
                chatbot = gr.Chatbot(height=540, show_label=False, elem_id="sparda-chat",
                                     avatar_images=None, sanitize_html=False)
                clear = gr.ClearButton([chatbot, query], value="Clear", size="sm")
            with gr.Column(scale=2, elem_classes="panel"):
                with gr.Tabs():
                    with gr.Tab("Knowledge graph"):
                        graph_html = gr.HTML()
                    with gr.Tab("SPLADE expansion"):
                        splade_json = gr.JSON(show_label=False)

        # Search on click AND Enter; three handlers per trigger fire in parallel and each reads
        # `query` at trigger time (before `chat` clears it), so graph/splade see the real query.
        for trigger in (btn.click, query.submit):
            trigger(chat, [query, chatbot], [query, chatbot])
            trigger(show_graph, [query], graph_html)
            trigger(show_splade, [query], splade_json)

    return demo
