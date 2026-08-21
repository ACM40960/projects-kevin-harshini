"""
Phase 3 — LangGraph agent nodes.

Four nodes forming the QuantRAG pipeline:
  A. Filing Retriever   - pulls cited evidence per ticker (Phase 2 retriever)
  B. View Extractor      - Claude tool-use forms a structured view per ticker
  C. Portfolio Optimiser - Black-Litterman turns views into weights
  D. Report Generator    - assembles a readable, cited final report

PHASE 4 COST-EFFICIENCY UPDATE:
  - Model switched from claude-sonnet-4-6 to claude-haiku-4-5-20251001
    for view extraction (~10x cheaper per token, sufficient quality for
    this well-scoped, schema-constrained task - verify with spot checks
    if reasoning depth ever seems insufficient)
  - Added a system-level reminder to fill ALL required tool fields on
    the first attempt, reducing wasted resubmission round-trips
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

# Cost-efficient model for view extraction — well-scoped, schema-constrained
# task where Haiku's quality is generally sufficient. ~10x cheaper than Sonnet.
VIEW_EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

# used only as a backup when Haiku genuinely can't complete the schema —
# see _run_tool_loop below
FALLBACK_MODEL = "claude-sonnet-4-6"


class AgentState(TypedDict, total=False):
    # this is the shared "clipboard" passed between all 4 nodes — each
    # node reads what it needs off it and writes its own result back on
    tickers: List[str]
    market_caps: Dict[str, float]
    price_history: object
    evidence: Dict[str, List[dict]]
    views: List[dict]
    weights: Dict[str, float]
    report: str


# Node A — Filing Retriever

def node_retriever(state: AgentState) -> AgentState:
    """For each ticker, retrieve top-5 cited passages on outlook/risk."""
    console.print("\n[bold blue]Node A — Filing Retriever[/bold blue]")

    model = load_embedder()
    client = get_qdrant_client()

    evidence = {}
    # same fixed query for every ticker — keeps the evidence basis
    # consistent/comparable across companies rather than tailoring
    # the search per company
    query = "business outlook, risk factors, and forward-looking guidance"

    for ticker in state["tickers"]:
        results = retrieve(query, model, client, ticker=ticker, top_k=5)
        evidence[ticker] = results
        console.print(f"[green]  ✓ {ticker}: {len(results)} passages retrieved[/green]")

    return {**state, "evidence": evidence}


# Node B — View Extractor (Claude tool-use) 

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


# System-level reminder — sent with EVERY tool-use call to reduce the
# frequency of incomplete first-attempt submissions (each incomplete
# submission costs one extra, avoidable API round-trip).
_SYSTEM_REMINDER = (
    "When calling extract_investment_view, you MUST include ALL FIVE "
    "required fields in a single call: direction, magnitude, confidence, "
    "reasoning, AND citation. Do not omit any field — incomplete "
    "submissions require a costly resubmission round-trip."
)


"""
Phase 4 — Patch to src/agent/nodes.py

Adds a Sonnet fallback for tickers that exhaust max_turns on Haiku.
This targets the actual failure mode observed: Haiku is cheap and
usually sufficient, but occasionally cannot complete the structured
schema within 6 turns. Rather than accepting "no view at all" for
that ticker, or reverting entirely to expensive Sonnet for everything,
we escalate ONLY the tickers that genuinely need it.

Cost impact: adds at most 1 extra (Sonnet) call per ticker that would
otherwise have failed entirely — far cheaper than repeatedly retrying
on a model that's already demonstrated it can't complete the task,
and MUCH cheaper than using Sonnet for all tickers by default.
"""

# ── Replace _run_tool_loop in src/agent/nodes.py with this version ────────

FALLBACK_MODEL = "claude-sonnet-4-6"


def _run_tool_loop(client, ticker: str, messages: list, tools: list, max_turns: int = 6) -> "Optional[dict]":
    """
    Runs the Claude tool-use conversation loop on the primary model
    (VIEW_EXTRACTION_MODEL, i.e. Haiku). If max_turns is exhausted
    without a complete view, makes ONE additional attempt on Sonnet
    (FALLBACK_MODEL) with a fresh, short conversation — cheaper than
    continuing to retry on a model that has already proven insufficient
    for this particular ticker's evidence complexity.
    """
    import json as _json
    from src.agent.tools import get_financial_metric

    required_fields = {"direction", "magnitude", "confidence", "reasoning", "citation"}

    def attempt(model_name: str, msgs: list, turns: int) -> "Optional[dict]":
        for _ in range(turns):
            response = client.messages.create(
                model=model_name,
                max_tokens=1024,
                system=_SYSTEM_REMINDER,
                tools=tools,
                messages=msgs,
            )

            for block in response.content:
                if block.type == "tool_use" and block.name == "extract_investment_view":
                    view = dict(block.input)
                    missing = required_fields - set(view.keys())

                    if not missing:
                        view["ticker"] = ticker
                        return view

                    console.print(
                        f"[yellow]  ⚠ {ticker} ({model_name}): view missing {missing}, "
                        f"requesting resubmission[/yellow]"
                    )
                    msgs.append({"role": "assistant", "content": response.content})
                    msgs.append({
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
                msgs.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use" and block.name == "get_financial_metric":
                        result = get_financial_metric(**block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _json.dumps(result),
                        })

                if tool_results:
                    msgs.append({"role": "user", "content": tool_results})
                else:
                    msgs.append({
                        "role": "user",
                        "content": "Please submit your view using extract_investment_view "
                                   "with ALL required fields.",
                    })

        return None

    # Attempt 1 -  primary model (Haiku, cheap)
    result = attempt(VIEW_EXTRACTION_MODEL, messages, max_turns)
    if result:
        return result

    # Attempt 2 -fallback to Sonnet with a FRESH short conversation
    # (fresh, not continuing the failed Haiku thread — avoids carrying
    # over any confusion from the exhausted attempt)
    console.print(
        f"[yellow]  ⚠ {ticker}: exhausted {max_turns} turns on {VIEW_EXTRACTION_MODEL}, "
        f"escalating to {FALLBACK_MODEL}[/yellow]"
    )

    original_prompt = messages[0]["content"] if messages else ""
    fresh_messages = [{"role": "user", "content": original_prompt}]

    result = attempt(FALLBACK_MODEL, fresh_messages, max_turns=3)
    if result:
        console.print(f"[green]  ✓ {ticker}: recovered via {FALLBACK_MODEL}[/green]")
        return result

    console.print(f"[red]  ✗ {ticker}: failed on both {VIEW_EXTRACTION_MODEL} and {FALLBACK_MODEL}[/red]")
    return None


#Node C - Portfolio Optimiser

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
    """Assembles a readable, cited final report. Defensive against missing fields."""
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