"""
Phase 3 — LangGraph agent nodes.

Four nodes forming the QuantRAG pipeline:
  A. Filing Retriever   — pulls cited evidence per ticker (Phase 2 retriever)
  B. View Extractor      — Claude tool-use forms a structured view per ticker
  C. Portfolio Optimiser — Black-Litterman turns views into weights
  D. Report Generator    — assembles a readable, cited final report
"""

import os
import json
from typing import TypedDict, List, Dict, Optional
from rich.console import Console

from src.retrieval.retriever import retrieve
from src.retrieval.embedder import load_embedder, get_qdrant_client
from src.agent.tools import (
    get_financial_metric,
    FINANCIAL_DATA_TOOL_SCHEMA,
    VIEW_EXTRACTION_TOOL_SCHEMA,
)
from src.optimizer.black_litterman import run_black_litterman

console = Console()


class AgentState(TypedDict, total=False):
    tickers: List[str]
    market_caps: Dict[str, float]
    price_history: object          # pandas DataFrame, set externally
    evidence: Dict[str, List[dict]]
    views: List[dict]
    weights: Dict[str, float]
    report: str


# ── Node A — Filing Retriever ──────────────────────────────────────────────

def node_retriever(state: AgentState) -> AgentState:
    """For each ticker, retrieve top-5 cited passages on outlook/risk."""
    console.print("\n[bold blue]Node A — Filing Retriever[/bold blue]")

    model = load_embedder()
    client = get_qdrant_client()

    evidence = {}
    query = "business outlook, risk factors, and forward-looking guidance"

    for ticker in state["tickers"]:
        results = retrieve(query, model, client, ticker=ticker, top_k=5)
        evidence[ticker] = results
        console.print(f"[green]  ✓ {ticker}: {len(results)} passages retrieved[/green]")

    return {**state, "evidence": evidence}


# ── Node B — View Extractor (Claude tool-use) ──────────────────────────────

def node_view_extractor(state: AgentState) -> AgentState:
    """
    For each ticker, Claude reads the retrieved evidence, optionally calls
    get_financial_metric for exact numbers, then submits a structured view
    via extract_investment_view tool-use.
    """
    console.print("\n[bold blue]Node B — View Extractor[/bold blue]")

    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    views = []

    for ticker in state["tickers"]:
        evidence_list = state["evidence"].get(ticker, [])
        if not evidence_list:
            console.print(f"[yellow]  ⚠ {ticker}: no evidence, skipping[/yellow]")
            continue

        context = "\n\n---\n\n".join(
            f"SOURCE: {e['citation']}\n{e['text']}" for e in evidence_list
        )

        prompt = f"""You are a financial analyst forming an investment view on {ticker}.

Read the filing evidence below. If you need exact financial figures
(revenue, margins, capex, etc.) to support your view, call
get_financial_metric. Once ready, submit your final view via
extract_investment_view.

FILING EVIDENCE:
{context}

Form a view: is {ticker} likely to outperform, underperform, or be
neutral relative to the market over the next year? Base your magnitude
and confidence on how strong and specific the evidence is."""

        messages = [{"role": "user", "content": prompt}]
        tools = [FINANCIAL_DATA_TOOL_SCHEMA, VIEW_EXTRACTION_TOOL_SCHEMA]

        view = _run_tool_loop(client, ticker, messages, tools)
        if view:
            views.append(view)
            console.print(
                f"[green]  ✓ {ticker}: {view['direction']} "
                f"({view['magnitude']:+.2%}, conf {view['confidence']:.0%})[/green]"
            )

    return {**state, "views": views}


def _run_tool_loop(client, ticker: str, messages: list, tools: list, max_turns: int = 4) -> Optional[dict]:
    """
    Runs the Claude tool-use conversation loop. Validates that the
    submitted view has all required fields — if incomplete, asks
    Claude to resubmit rather than silently accepting a partial view.
    """
    required_fields = {"direction", "magnitude", "confidence", "reasoning", "citation"}

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_investment_view":
                view = dict(block.input)
                missing = required_fields - set(view.keys())

                if not missing:
                    view["ticker"] = ticker
                    return view

                # Incomplete view — ask Claude to resubmit with the missing fields
                console.print(
                    f"[yellow]  ⚠ {ticker}: view missing {missing}, "
                    f"requesting resubmission[/yellow]"
                )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": (
                            f"Your submission is missing required field(s): "
                            f"{', '.join(missing)}. Please call "
                            f"extract_investment_view again with ALL fields filled in."
                        ),
                    }],
                })
                break
        else:
            # No extract_investment_view call this turn — handle tool calls or nudge
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "get_financial_metric":
                    result = get_financial_metric(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                messages.append({
                    "role": "user",
                    "content": "Please submit your view using extract_investment_view "
                               "with ALL required fields.",
                })

    console.print(f"[red]  ✗ {ticker}: exceeded max tool-use turns without a complete view[/red]")
    return None

# ── Node C — Portfolio Optimiser ───────────────────────────────────────────

def node_optimizer(state: AgentState) -> AgentState:
    """Runs Black-Litterman using the extracted views."""
    console.print("\n[bold blue]Node C — Portfolio Optimiser[/bold blue]")

    weights = run_black_litterman(
        tickers=state["tickers"],
        views=state["views"],
        price_history=state["price_history"],
        market_caps=state["market_caps"],
        max_weight=0.20,
    )

    return {**state, "weights": weights}


# ── Node D — Report Generator ──────────────────────────────────────────────

def node_report_generator(state: AgentState) -> AgentState:
    """Assembles a readable, cited final report."""
    console.print("\n[bold blue]Node D — Report Generator[/bold blue]")

    lines = ["# Portfolio Allocation Report\n"]

    for ticker in state["tickers"]:
        weight = state["weights"].get(ticker, 0)
        view = next((v for v in state["views"] if v["ticker"] == ticker), None)

        lines.append(f"## {ticker} — {weight:.2%}")

        if view:
            direction  = view.get("direction", "neutral")
            magnitude  = view.get("magnitude", 0.0)
            confidence = view.get("confidence", 0.0)
            reasoning  = view.get("reasoning", "No reasoning provided.")
            citation   = view.get("citation", "No citation provided.")

            lines.append(
                f"**{direction.title()}** "
                f"({magnitude:+.1%}, confidence {confidence:.0%})"
            )
            lines.append(f"{reasoning}")
            lines.append(f"*Source: {citation}*\n")
        else:
            lines.append("*No view available — held at market weight.*\n")

    report = "\n".join(lines)
    console.print("[green]  ✓ Report generated[/green]")

    return {**state, "report": report}