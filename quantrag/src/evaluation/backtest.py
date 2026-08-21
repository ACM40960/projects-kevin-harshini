"""
Phase 4 — Quarterly rebalancing backtest.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from src.optimizer.black_litterman import run_black_litterman
from src.evaluation.momentum_baseline import generate_momentum_views
from src.evaluation.performance_metrics import full_report

console = Console()

BACKTEST_CHECKPOINT = "logs/backtest_checkpoint.jsonl"
VIEW_CACHE_FILE = "logs/backtest_view_cache.jsonl"
LOOKBACK_BUFFER_DAYS = 280


class ApiCallLimitExceeded(Exception):
    """Raised when the backtest's Claude API call count exceeds the safety ceiling."""
    pass


# ── API call counter — a simple global tracked across the whole backtest ──

class ApiCallCounter:
    """
    Tracks every real Claude API call made during a backtest run.
    Passed through to _run_tool_loop via a wrapped client so every
    call — including resubmission retries — is counted accurately.
    """
    def __init__(self, max_calls: int):
        self.count = 0
        self.max_calls = max_calls

    def increment(self):
        self.count += 1
        if self.count > self.max_calls:
            raise ApiCallLimitExceeded(
                f"API call limit exceeded: {self.count} calls made, "
                f"limit was {self.max_calls}. Stopping to protect credits — "
                f"this likely indicates unexpected re-computation (a caching "
                f"regression). Check logs/backtest_view_cache.jsonl for what "
                f"was already completed before this run."
            )


class CountedAnthropicClient:
    """
    Thin wrapper around the real Anthropic client that increments a
    shared counter on every .messages.create() call, then delegates
    to the real client. Used so _run_tool_loop's existing code needs
    zero changes to be tracked.
    """
    def __init__(self, real_client, counter: ApiCallCounter):
        self._real_client = real_client
        self._counter = counter
        self.messages = self

    def create(self, *args, **kwargs):
        self._counter.increment()
        return self._real_client.messages.create(*args, **kwargs)
# Point in time filing year resolution
# this decides which fiscal year's 10-K would actually have been
# public knowledge by a given simulated date, so the backtest never
# accidentally "sees the future"


# Point-in-time filing year resolution 

def filing_year_available_at(quarter_date: pd.Timestamp) -> int:
    if quarter_date.month >= 4:
        return quarter_date.year - 1
    else:
        return quarter_date.year - 2


def get_quarter_dates(start: str, end: str) -> List[pd.Timestamp]:
    return list(pd.date_range(start=start, end=end, freq="QE"))


def estimate_max_api_calls(tickers: List[str], start: str, end: str, calls_per_ticker: int = 3) -> int:
    """
    Estimate the maximum reasonable number of Claude API calls for a
    backtest, used as the default safety ceiling. calls_per_ticker=3
    generously covers 1 base call + up to 2 resubmission retries.
    """
    quarters = get_quarter_dates(start, end)
    fiscal_years = set(filing_year_available_at(q) for q in quarters)
    return len(tickers) * len(fiscal_years) * calls_per_ticker


# Backtest checkpointing, quarter level, weights
# this is what lets you stop and resume a multi hour backtest without
# losing anything already computed

def load_checkpoint() -> Dict[str, dict]:
    completed = {}
    if os.path.exists(BACKTEST_CHECKPOINT):
        with open(BACKTEST_CHECKPOINT) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    completed[row["key"]] = row
    return completed


def append_checkpoint(row: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(BACKTEST_CHECKPOINT, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


#Disk-backed per-ticker view cache (fine-grained, crash-proof)

def _load_disk_view_cache() -> dict:
    cache = {}
    if os.path.exists(VIEW_CACHE_FILE):
        with open(VIEW_CACHE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    key = (row["universe_key"], row["fiscal_year"], row["ticker"])
                    cache[key] = row["view"]
    return cache


def _append_disk_view_cache(universe_key: str, fiscal_year: int, ticker: str, view: dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(VIEW_CACHE_FILE, "a") as f:
        f.write(json.dumps({
            "universe_key": universe_key,
            "fiscal_year": fiscal_year,
            "ticker": ticker,
            "view": view,
        }) + "\n")


# Market data 

def fetch_full_history(tickers: List[str], start: str, end: str) -> tuple:
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    console.print(f"[blue]Fetching price history from {fetch_start} (includes lookback buffer)...[/blue]")

    price_data = {}
    market_caps = {}
    for ticker in tickers:
        t = yf.Ticker(ticker)
        hist = t.history(start=fetch_start, end=end)["Close"]
        price_data[ticker] = hist
        market_caps[ticker] = t.info.get("marketCap", 1e9)

    price_history = pd.DataFrame(price_data).dropna(how="all")
    if price_history.index.tz is not None:
        price_history.index = price_history.index.tz_localize(None)

    console.print(f"[green]  ✓ Loaded prices for {len(price_history.columns)} tickers, "
                  f"{len(price_history)} trading days (incl. buffer)[/green]")
    return price_history, market_caps


def price_history_up_to(price_history: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int = 252) -> pd.DataFrame:
    window_start = as_of - pd.Timedelta(days=lookback_days)
    return price_history[(price_history.index >= window_start) & (price_history.index <= as_of)]


def realised_return_between(price_history: pd.DataFrame, weights: Dict[str, float],
                              start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
    start_slice = price_history[price_history.index <= start_date]
    end_slice = price_history[price_history.index <= end_date]

    if start_slice.empty or end_slice.empty:
        console.print(f"[yellow]  ⚠ No price data available for {start_date.date()} -> {end_date.date()}[/yellow]")
        return 0.0

    start_prices = start_slice.iloc[-1]
    end_prices = end_slice.iloc[-1]

    portfolio_return = 0.0
    for ticker, weight in weights.items():
        if ticker in start_prices.index and ticker in end_prices.index:
            p0, p1 = start_prices[ticker], end_prices[ticker]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                portfolio_return += weight * ((p1 / p0) - 1)

    return portfolio_return


# View generation per fiscal year 

_view_cache: Dict[tuple, List[dict]] = {}


def get_quantrag_views_for_year(
    tickers: List[str], fiscal_year: int, model, qdrant_client, anthropic_client
) -> List[dict]:
    
    universe_key = ",".join(sorted(tickers))
    cache_key = (tuple(sorted(tickers)), fiscal_year)

    # in memory cache first, cheapest possible check, avoids even
    # touching the disk cache if we already did this exact year in
    # this exact run
    if cache_key in _view_cache:
        return _view_cache[cache_key]

    from src.retrieval.retriever import retrieve
    from src.agent.nodes import _run_tool_loop
    from src.agent.tools import FINANCIAL_DATA_TOOL_SCHEMA, VIEW_EXTRACTION_TOOL_SCHEMA

    disk_cache = _load_disk_view_cache()

    console.print(f"[blue]  Generating QuantRAG views for FY{fiscal_year}...[/blue]")

    views = []
    query = "business outlook, risk factors, and forward-looking guidance"
    already_cached_count = 0

    for ticker in tickers:
        disk_key = (universe_key, fiscal_year, ticker)

        if disk_key in disk_cache:
            views.append(disk_cache[disk_key])
            already_cached_count += 1
            continue

        results = retrieve(query, model, qdrant_client, ticker=ticker, top_k=5)
        if not results:
            continue

        context = "\n\n---\n\n".join(f"SOURCE: {r['citation']}\n{r['text']}" for r in results)
        prompt = f"""You are a financial analyst forming an investment view on {ticker}
based on their fiscal year {fiscal_year} filing.

FILING EVIDENCE:
{context}

Form a view: is {ticker} likely to outperform, underperform, or be
neutral relative to the market over the following year?"""

        messages = [{"role": "user", "content": prompt}]
        tools = [FINANCIAL_DATA_TOOL_SCHEMA, VIEW_EXTRACTION_TOOL_SCHEMA]

        view = _run_tool_loop(anthropic_client, ticker, messages, tools, max_turns=6)
        if view:
            views.append(view)
            _append_disk_view_cache(universe_key, fiscal_year, ticker, view)

    if already_cached_count:
        console.print(f"[dim]    ({already_cached_count}/{len(tickers)} tickers reused from disk cache)[/dim]")

    _view_cache[cache_key] = views
    return views


#  Main backtest loop

def run_backtest(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    resume: bool = True,
    max_api_calls: int = None,
) -> Dict:
    """
    Args:
        max_api_calls: hard ceiling on total Claude API calls for this
            run. If None, auto-estimated as (tickers × fiscal_years × 3)
            — generously covering 1 base call + 2 resubmission retries
            per ticker-year. Raises ApiCallLimitExceeded and stops
            cleanly if exceeded, protecting against any caching
            regression silently burning through credits.
    """
    console.print(f"\n[bold blue]Phase 4 — Backtest: {start} to {end}[/bold blue]")
    console.print(f"  Universe: {len(tickers)} tickers\n")

    if max_api_calls is None:
        max_api_calls = estimate_max_api_calls(tickers, start, end)
    console.print(f"[dim]API call safety ceiling: {max_api_calls}[/dim]\n")

    price_history, market_caps = fetch_full_history(tickers, start, end)
    quarters = get_quarter_dates(start, end)

    if len(quarters) < 2:
        raise ValueError(
            f"Only {len(quarters)} quarter(s) in range — need at least 2 "
            f"to compute any return. Widen the start/end range."
        )

    from src.retrieval.embedder import load_embedder, get_qdrant_client
    import anthropic

    console.print("[blue]Loading embedding model (once for entire backtest)...[/blue]")
    shared_model = load_embedder()
    shared_qdrant_client = get_qdrant_client()

    real_anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    call_counter = ApiCallCounter(max_calls=max_api_calls)
    shared_anthropic_client = CountedAnthropicClient(real_anthropic_client, call_counter)

    already_done = load_checkpoint() if resume else {}
    all_weights = {"quantrag": {}, "momentum": {}}

    try:
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Computing weights per quarter", total=len(quarters))

            for q_date in quarters:
                fiscal_year = filing_year_available_at(q_date)
                hist_window = price_history_up_to(price_history, q_date)

                if hist_window.shape[0] < 30:
                    console.print(f"[yellow]  ⚠ {q_date.date()}: insufficient price history "
                                  f"({hist_window.shape[0]} days), using equal weight[/yellow]")
                    all_weights["quantrag"][q_date] = {t: 1.0 / len(tickers) for t in tickers}
                    all_weights["momentum"][q_date] = {t: 1.0 / len(tickers) for t in tickers}
                    progress.advance(task)
                    continue

                qr_key = f"quantrag_{q_date.date()}"
                if qr_key in already_done:
                    qr_weights = already_done[qr_key]["weights"]
                else:
                    try:
                        views = get_quantrag_views_for_year(
                            tickers, fiscal_year, shared_model, shared_qdrant_client, shared_anthropic_client
                        )
                        qr_weights = run_black_litterman(tickers, views, hist_window, market_caps, max_weight=0.20)
                    except ApiCallLimitExceeded:
                        raise  # propagate — this must stop the whole run
                    except Exception as e:
                        console.print(f"[red]  ✗ QuantRAG failed at {q_date.date()}: {str(e)[:200]}[/red]")
                        qr_weights = {t: 1.0 / len(tickers) for t in tickers}
                    append_checkpoint({"key": qr_key, "date": q_date, "strategy": "quantrag", "weights": qr_weights})
                all_weights["quantrag"][q_date] = qr_weights

                mo_key = f"momentum_{q_date.date()}"
                if mo_key in already_done:
                    mo_weights = already_done[mo_key]["weights"]
                else:
                    try:
                        mo_views = generate_momentum_views(tickers, hist_window, as_of_date=q_date)
                        mo_weights = run_black_litterman(tickers, mo_views, hist_window, market_caps, max_weight=0.20)
                    except Exception as e:
                        console.print(f"[red]  ✗ Momentum failed at {q_date.date()}: {str(e)[:200]}[/red]")
                        mo_weights = {t: 1.0 / len(tickers) for t in tickers}
                    append_checkpoint({"key": mo_key, "date": q_date, "strategy": "momentum", "weights": mo_weights})
                all_weights["momentum"][q_date] = mo_weights

                progress.advance(task)

    except ApiCallLimitExceeded as e:
        console.print(f"\n[bold red]STOPPED — {e}[/bold red]")
        console.print(
            f"[yellow]Progress so far is safely saved in {BACKTEST_CHECKPOINT} "
            f"and {VIEW_CACHE_FILE} — re-run with resume=True to continue "
            f"once you've investigated.[/yellow]"
        )
        raise

    console.print(f"\n[dim]Total Claude API calls made this run: {call_counter.count}[/dim]")

    # Build equity curves - walk consecutive quarter pairs (no off-by-one)
    quantrag_curve = {quarters[0]: 100.0}
    momentum_curve = {quarters[0]: 100.0}

    for i in range(len(quarters) - 1):
        q_from, q_to = quarters[i], quarters[i + 1]

        qr_return = realised_return_between(price_history, all_weights["quantrag"][q_from], q_from, q_to)
        mo_return = realised_return_between(price_history, all_weights["momentum"][q_from], q_from, q_to)

        quantrag_curve[q_to] = quantrag_curve[q_from] * (1 + qr_return)
        momentum_curve[q_to] = momentum_curve[q_from] * (1 + mo_return)

    spy_fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    spy = yf.Ticker("SPY").history(start=spy_fetch_start, end=end)["Close"]
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)

    spy_curve = {}
    spy_base_slice = spy[spy.index <= quarters[0]]
    base_value = float(spy_base_slice.iloc[-1]) if not spy_base_slice.empty else float(spy.iloc[0])
    for q_date in quarters:
        s = spy[spy.index <= q_date]
        if not s.empty:
            spy_curve[q_date] = float(s.iloc[-1] / base_value * 100)

    quantrag_series = pd.Series(quantrag_curve).sort_index()
    momentum_series = pd.Series(momentum_curve).sort_index()
    spy_series = pd.Series(spy_curve).sort_index()

    quantrag_report = full_report(quantrag_series, spy_series, label="QuantRAG")
    momentum_report = full_report(momentum_series, spy_series, label="Momentum Baseline")
    spy_report = full_report(spy_series, label="SPY Benchmark")

    console.print("\n[bold green]Backtest complete[/bold green]")

    return {
        "quantrag_curve": quantrag_series,
        "momentum_curve": momentum_series,
        "spy_curve": spy_series,
        "quantrag_report": quantrag_report,
        "momentum_report": momentum_report,
        "spy_report": spy_report,
        "total_api_calls": call_counter.count,
    }