"""
Phase 5 — QuantRAG Gradio Demo (v3 — polished UI, same colour palette).

Visual changes only (no colour/theme changes):
  - Stat badge row under the header
  - Evidence + portfolio outputs rendered as styled HTML cards
    instead of plain markdown
  - Direction badges (bullish/bearish/neutral) with colour accents
  - Section dividers, better spacing, subtle shadows/borders,
    hover states, rounded cards throughout
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import yfinance as yf

from src.retrieval.embedder import load_embedder, get_qdrant_client
from src.retrieval.retriever import retrieve
from src.agent.graph import build_agent_graph

print("Loading embedding model and Qdrant connection...")
_model = load_embedder()
_client = get_qdrant_client()

print("Building ticker list from index...")

def _build_ticker_choices():
    seen = {}
    offset = None
    while True:
        records, offset = _client.scroll(
            collection_name=os.getenv("QDRANT_COLLECTION", "quantrag_phase2"),
            limit=1000, offset=offset,
            with_payload=["ticker", "company_name"],
        )
        for r in records:
            t = r.payload.get("ticker")
            c = r.payload.get("company_name", "")
            if t and t not in seen:
                seen[t] = c.title() if c else t
        if offset is None:
            break
    labels = sorted([f"{t} — {c}" for t, c in seen.items()])
    return labels, seen

TICKER_LABELS, TICKER_MAP = _build_ticker_choices()
print(f"Ready. {len(TICKER_LABELS)} tickers loaded.")

# ── Live SPY snapshot ────────────────────────────────────────────────────

def get_spy_snapshot() -> str:
    """Fetch SPY's latest close vs. previous close and render as an HTML stat card."""
    try:
        hist = yf.Ticker("SPY").history(period="5d")["Close"]
        if len(hist) < 2:
            return "<div class='spy-card spy-error'>SPY data unavailable</div>"

        latest, prev = hist.iloc[-1], hist.iloc[-2]
        change = latest - prev
        pct = (change / prev) * 100
        up = change >= 0
        arrow = "▲" if up else "▼"
        css_class = "spy-up" if up else "spy-down"
        date_str = hist.index[-1].strftime("%b %d, %Y")

        return f"""
        <div class="spy-card">
            <div class="spy-label">S&amp;P 500 (SPY) &middot; {date_str}</div>
            <div class="spy-row">
                <span class="spy-price">${latest:.2f}</span>
                <span class="spy-change {css_class}">{arrow} {change:+.2f} ({pct:+.2f}%)</span>
            </div>
        </div>
        """
    except Exception:
        return "<div class='spy-card spy-error'>SPY data unavailable</div>"



def _extract_ticker(label: str) -> str:
    return label.split(" — ")[0].strip() if label else ""


# ── Mode 1 — Evidence lookup (returns styled HTML) ──────────────────────

def lookup_evidence(ticker_label: str, question: str) -> str:
    if not ticker_label or not question:
        return "<div class='empty-state'>Select a ticker and enter a question to begin.</div>"

    ticker = _extract_ticker(ticker_label)
    try:
        results = retrieve(question, _model, _client, ticker=ticker, top_k=5)
    except Exception as e:
        return f"<div class='error-state'>Retrieval error: {str(e)[:200]}</div>"

    if not results:
        return f"<div class='empty-state'>No indexed filing data found for <b>{ticker}</b>.</div>"

    cards = [f"""
    <div class="query-header">
        <span class="pill">{ticker_label}</span>
        <span class="query-text">"{question}"</span>
    </div>
    """]

    for i, r in enumerate(results):
        cards.append(f"""
        <div class="evidence-card">
            <div class="evidence-rank">#{i+1}</div>
            <div class="evidence-body">
                <div class="evidence-source">{r['citation']}</div>
                <div class="evidence-text">{r['text'][:400]}...</div>
            </div>
        </div>
        """)

    return "".join(cards)


# llm retrival

def ask_question(ticker_label: str, question: str) -> str:
    if not ticker_label or not question:
        return "<div class='empty-state'>Select a company and enter a question.</div>"

    ticker = _extract_ticker(ticker_label)
    try:
        results = retrieve(question, _model, _client, ticker=ticker, top_k=5)
    except Exception as e:
        return f"<div class='error-state'>Retrieval error: {str(e)[:200]}</div>"

    if not results:
        return f"<div class='empty-state'>No indexed filing data found for <b>{ticker}</b>.</div>"

    context = "\n\n---\n\n".join(f"SOURCE: {r['citation']}\n{r['text']}" for r in results)
    prompt = f"""You are a financial analyst assistant.
Answer the following question based ONLY on the provided SEC filing excerpts.
Always cite the source of every claim using the SOURCE labels.
If the answer is not in the excerpts, say "Not found in provided documents."

QUESTION: {question}

FILING EXCERPTS:
{context}

ANSWER:"""

    from src.llm import get_llm_response
    try:
        answer = get_llm_response(prompt, max_tokens=1024)
    except Exception as e:
        return f"<div class='error-state'>LLM error: {str(e)[:200]}</div>"

    return f"""
    <div class="query-header">
        <span class="pill">{ticker_label}</span>
        <span class="query-text">"{question}"</span>
    </div>
    <div class="ticker-card">
        <div class="ticker-reasoning">{answer}</div>
    </div>
    """

# ── Mode 2 — Full portfolio agent run (returns styled HTML) ─────────────

DIRECTION_STYLE = {
    "bullish": ("dir-bull", "▲"),
    "bearish": ("dir-bear", "▼"),
    "neutral": ("dir-neutral", "●"),
}


def run_portfolio(ticker_labels: list, progress=gr.Progress()) -> tuple:
    if not ticker_labels or len(ticker_labels) < 2:
        return "<div class='empty-state'>Select at least 2 companies.</div>", None
    if len(ticker_labels) > 10:
        return "<div class='empty-state'>Please select up to 10 tickers for this demo.</div>", None

    tickers = [_extract_ticker(l) for l in ticker_labels]

    progress(0.1, desc="Fetching market data...")
    try:
        market_caps, price_data = {}, {}
        for t in tickers:
            tk = yf.Ticker(t)
            market_caps[t] = tk.info.get("marketCap", 1e9)
            price_data[t] = tk.history(period="1y")["Close"]
        price_history = pd.DataFrame(price_data).dropna()
    except Exception as e:
        return f"<div class='error-state'>Market data error: {str(e)[:200]}</div>", None

    progress(0.2, desc="Running QuantRAG agent...")
    try:
        agent = build_agent_graph()
        final_state = agent.invoke({
            "tickers": tickers, "market_caps": market_caps, "price_history": price_history,
        })
    except Exception as e:
        return f"<div class='error-state'>Agent error: {str(e)[:300]}</div>", None

    progress(1.0, desc="Done")

    weights_df = pd.DataFrame([
        {"Ticker": t, "Weight": final_state["weights"].get(t, 0)} for t in tickers
    ]).sort_values("Weight", ascending=False)

    cards = ['<div class="report-title">Portfolio Allocation Report</div>']
    for ticker in sorted(tickers, key=lambda t: -final_state["weights"].get(t, 0)):
        weight = final_state["weights"].get(ticker, 0)
        view = next((v for v in final_state["views"] if v["ticker"] == ticker), None)

        if view:
            direction = view.get("direction", "neutral").lower()
            css_class, arrow = DIRECTION_STYLE.get(direction, DIRECTION_STYLE["neutral"])
            magnitude = view.get("magnitude", 0.0)
            confidence = view.get("confidence", 0.0)
            reasoning = view.get("reasoning", "")
            citation = view.get("citation", "")

            cards.append(f"""
            <div class="ticker-card">
                <div class="ticker-card-head">
                    <span class="ticker-name">{ticker}</span>
                    <span class="ticker-weight">{weight:.2%}</span>
                </div>
                <div class="ticker-card-sub">
                    <span class="dir-badge {css_class}">{arrow} {direction.title()}</span>
                    <span class="magnitude">{magnitude:+.1%}</span>
                    <span class="confidence">conf. {confidence:.0%}</span>
                </div>
                <div class="ticker-reasoning">{reasoning}</div>
                <div class="ticker-source">Source: {citation}</div>
            </div>
            """)
        else:
            cards.append(f"""
            <div class="ticker-card">
                <div class="ticker-card-head">
                    <span class="ticker-name">{ticker}</span>
                    <span class="ticker-weight">{weight:.2%}</span>
                </div>
                <div class="ticker-card-sub"><span class="dir-badge dir-neutral">No view — held at market weight</span></div>
            </div>
            """)

    return "".join(cards), weights_df


# ── Theme (unchanged colours) ─────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0B0F19",
    body_background_fill_dark="#0B0F19",
    background_fill_primary="#131826",
    background_fill_secondary="#0F1420",
    border_color_primary="#232B3D",
    block_background_fill="#131826",
    block_border_color="#232B3D",
    block_title_text_color="#7DE0E6",
    block_label_text_color="#7DE0E6",
    body_text_color="#E4E8F1",
    body_text_color_subdued="#8A93A8",
    button_primary_background_fill="linear-gradient(90deg,#00D4C8,#7B5CFF)",
    button_primary_background_fill_hover="linear-gradient(90deg,#00E8DA,#8F73FF)",
    button_primary_text_color="#0B0F19",
    input_background_fill="#0F1420",
    input_border_color="#2A3348",
)

CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }

/* ── Header ── */
#title-block {
    background: radial-gradient(circle at top left, #17324A 0%, #0B0F19 55%);
    border: 1px solid #232B3D;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
#title-block::after {
    content: "";
    position: absolute; top: -40%; right: -10%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,200,0.15), transparent 70%);
}
#title-block h1 {
    background: linear-gradient(90deg, #00D4C8, #7B5CFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.6em !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
#title-block .subtitle { color: #A6AFC4; font-size: 1.05em; margin-top: 4px; }

#chart-decoration {
    position: absolute; top: 12px; right: 24px;
    width: 260px; height: 120px; opacity: 0.9;
    pointer-events: none;
}


.badge-row { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
.badge {
    background: #0F1420; border: 1px solid #2A3348; border-radius: 999px;
    padding: 6px 14px; font-size: 0.82em; color: #7DE0E6;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Tabs ── */
.tabitem { padding-top: 8px !important; }

/* ── Query header ── */
.query-header {
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; margin-bottom: 16px;
    background: #0F1420; border: 1px solid #232B3D; border-radius: 10px;
}
.pill {
    background: linear-gradient(90deg,#00D4C8,#7B5CFF); color: #0B0F19;
    padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85em;
    white-space: nowrap;
}
.query-text { color: #A6AFC4; font-style: italic; font-size: 0.95em; }

/* ── Evidence cards ── */
.evidence-card {
    display: flex; gap: 14px;
    background: #131826; border: 1px solid #232B3D; border-left: 3px solid #00D4C8;
    border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.evidence-card:hover { border-color: #00D4C8; transform: translateX(2px); }
.evidence-rank {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #7B5CFF;
    font-size: 1.1em; min-width: 28px;
}
.evidence-source { color: #7DE0E6; font-weight: 600; font-size: 0.9em; margin-bottom: 6px; }
.evidence-text { color: #C7CEDD; font-size: 0.92em; line-height: 1.55; }

/* ── Portfolio report cards ── */
.report-title {
    font-size: 1.4em; font-weight: 700; color: #E4E8F1; margin-bottom: 14px;
}
.ticker-card {
    background: #131826; border: 1px solid #232B3D; border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
    transition: border-color 0.15s ease;
}
.ticker-card:hover { border-color: #3A4560; }
.ticker-card-head { display: flex; justify-content: space-between; align-items: baseline; }
.ticker-name { font-size: 1.25em; font-weight: 700; color: #E4E8F1; font-family: 'JetBrains Mono', monospace; }
.ticker-weight { font-size: 1.25em; font-weight: 700; color: #00D4C8; font-family: 'JetBrains Mono', monospace; }
.ticker-card-sub { display: flex; gap: 10px; align-items: center; margin: 8px 0 12px 0; flex-wrap: wrap; }
.dir-badge { padding: 3px 10px; border-radius: 999px; font-size: 0.8em; font-weight: 700; }
.dir-bull { background: rgba(0,212,150,0.15); color: #00D496; border: 1px solid rgba(0,212,150,0.35); }
.dir-bear { background: rgba(255,90,90,0.15); color: #FF6B6B; border: 1px solid rgba(255,90,90,0.35); }
.dir-neutral { background: rgba(123,92,255,0.15); color: #A18AFF; border: 1px solid rgba(123,92,255,0.35); }
.magnitude { color: #A6AFC4; font-size: 0.85em; font-family: 'JetBrains Mono', monospace; }
.confidence { color: #6B7488; font-size: 0.85em; }
.ticker-reasoning { color: #C7CEDD; font-size: 0.92em; line-height: 1.55; margin-bottom: 8px; }
.ticker-source { color: #5B6478; font-size: 0.82em; font-style: italic; }

/* ── Empty / error states ── */
.empty-state, .error-state {
    text-align: center; padding: 40px 20px; color: #5B6478;
    border: 1px dashed #232B3D; border-radius: 12px; font-size: 0.95em;
}
.error-state { color: #FF6B6B; border-color: rgba(255,90,90,0.3); }

footer { display: none !important; }

/* ── Fix multiselect dropdown chip visibility ── */
.gr-dropdown .token,
[data-testid="dropdown"] .token,
.wrap .token {
    background: #1B2233 !important;
    border: 1px solid #2A3348 !important;
    color: #E4E8F1 !important;
}
.gr-dropdown .token span,
[data-testid="dropdown"] .token span,
.wrap .token span {
    color: #E4E8F1 !important;
}
.gr-dropdown .token svg,
[data-testid="dropdown"] .token svg,
.wrap .token svg {
    fill: #8A93A8 !important;
    color: #8A93A8 !important;
}


/* ── SPY live snapshot ── */
.spy-card {
    background: #0F1420; border: 1px solid #232B3D; border-radius: 12px;
    padding: 12px 18px; min-width: 220px;
}
.spy-label { color: #6B7488; font-size: 0.75em; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px; }
.spy-row { display: flex; align-items: baseline; gap: 12px; }
.spy-price { font-size: 1.4em; font-weight: 700; color: #E4E8F1; font-family: 'JetBrains Mono', monospace; }
.spy-change { font-size: 0.9em; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.spy-up { color: #00D496; }
.spy-down { color: #FF6B6B; }
.spy-error { color: #5B6478; font-size: 0.85em; text-align: center; }

"""

with gr.Blocks(title="QuantRAG Demo", theme=theme, css=CUSTOM_CSS) as demo:

    gr.HTML(f"""
        <div id="title-block">
            <svg id="chart-decoration" viewBox="0 0 300 140" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#00D4C8"/>
                        <stop offset="100%" stop-color="#7B5CFF"/>
                    </linearGradient>
                    <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#00D4C8" stop-opacity="0.25"/>
                        <stop offset="100%" stop-color="#00D4C8" stop-opacity="0"/>
                    </linearGradient>
                </defs>
                <path d="M0,110 L30,95 L60,100 L90,70 L120,80 L150,45 L180,55 L210,25 L240,35 L270,10 L300,20 L300,140 L0,140 Z" fill="url(#fillGrad)"/>
                <polyline points="0,110 30,95 60,100 90,70 120,80 150,45 180,55 210,25 240,35 270,10 300,20" fill="none" stroke="url(#lineGrad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="300" cy="20" r="4" fill="#00D4C8"/>
            </svg>
            <h1>QuantRAG</h1>
            <div class="subtitle">Agentic RAG · Black-Litterman · Explainable Portfolio Intelligence</div>
            <div class="badge-row">
                <span class="badge">◆ {len(TICKER_LABELS)} companies indexed</span>
                <span class="badge">◆ 281,784 filing passages</span>
                <span class="badge">◆ Claude tool-use agent</span>
                <span class="badge">◆ Black-Litterman optimiser</span>
            </div>
        </div>
    """)

    with gr.Row():
        spy_display = gr.HTML(value=get_spy_snapshot())
        spy_refresh_btn = gr.Button("↻ Refresh SPY", size="sm", scale=0)
    spy_refresh_btn.click(fn=get_spy_snapshot, outputs=spy_display)
    demo.load(fn=get_spy_snapshot, outputs=spy_display)

    with gr.Tab("⚡ Evidence Lookup"):
        gr.Markdown("Instant, free retrieval — no LLM calls. Pick a company, ask a question.")
        with gr.Row():
            ticker_dd = gr.Dropdown(choices=TICKER_LABELS, label="Company", filterable=True, scale=1)
            question_input = gr.Textbox(
                label="Question", placeholder="What are the main risks to revenue growth?", scale=3,
            )
        lookup_btn = gr.Button("🔍 Retrieve Evidence", variant="primary")
        lookup_output = gr.HTML()
        lookup_btn.click(fn=lookup_evidence, inputs=[ticker_dd, question_input], outputs=lookup_output)

    with gr.Tab("💬 Ask QuantRAG"):
        gr.Markdown("Ask any question about one company — retrieval + LLM-generated, cited answer.")
        with gr.Row():
            ask_ticker_dd = gr.Dropdown(choices=TICKER_LABELS, label="Company", filterable=True, scale=1)
            ask_question_input = gr.Textbox(
                label="Question", placeholder="Is this company capital-intensive?", scale=3,
        )
        ask_btn = gr.Button("💬 Ask", variant="primary")
        ask_output = gr.HTML()
        ask_btn.click(fn=ask_question, inputs=[ask_ticker_dd, ask_question_input], outputs=ask_output)

    with gr.Tab("🧠 Full Portfolio"):
        gr.Markdown("Runs the full agent: retrieval → Claude view extraction → Black-Litterman. 2-10 tickers, ~1-3 min.")
        portfolio_dd = gr.Dropdown(choices=TICKER_LABELS, label="Companies", multiselect=True, filterable=True)
        portfolio_btn = gr.Button("🚀 Build Portfolio", variant="primary")
        with gr.Row():
            portfolio_report = gr.HTML(scale=3)
            portfolio_weights = gr.BarPlot(x="Ticker", y="Weight", title="Portfolio Weights", y_lim=[0, 1], scale=2)
        portfolio_btn.click(fn=run_portfolio, inputs=[portfolio_dd], outputs=[portfolio_report, portfolio_weights])

    gr.Markdown(
        "<div style='text-align:center; color:#5B6478; font-size:0.8em; margin-top:20px;'>"
        "MSc AI &amp; Data Science Research Project &nbsp;·&nbsp; SEC 10-K Filings 2020-2024"
        "</div>"
    )

if __name__ == "__main__":
    demo.launch()