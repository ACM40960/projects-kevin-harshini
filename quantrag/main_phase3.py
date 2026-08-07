"""
Phase 3 — Agent runner.
Give it a list of tickers, get back a full portfolio with citations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
import pandas as pd
from rich.console import Console
from rich.rule import Rule

from src.agent.graph import build_agent_graph

console = Console()


def fetch_market_data(tickers: list) -> tuple:
    """Fetch market caps and 1-year price history for the given tickers."""
    console.print("[blue]Fetching market data...[/blue]")

    market_caps = {}
    price_data = {}

    for ticker in tickers:
        t = yf.Ticker(ticker)
        info = t.info
        market_caps[ticker] = info.get("marketCap", 1e9)

        hist = t.history(period="1y")["Close"]
        price_data[ticker] = hist

    price_history = pd.DataFrame(price_data).dropna()
    return market_caps, price_history


def main():
    console.print(Rule("[bold blue]QuantRAG Phase 3 — Agent Run[/bold blue]"))

    # Start small — 3 tickers for the first end-to-end test
    tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",      # Tech/Consumer Disc.
    "META", "TSLA", "IBM", "V", "MA",              # Comm/Financials
    "UNH", "JNJ", "PFE", "XOM", "CVX",             # Healthcare/Energy
    "PG", "KO", "WMT", "HD", "DIS",                # Staples/Consumer
]

    market_caps, price_history = fetch_market_data(tickers)

    agent = build_agent_graph()

    initial_state = {
        "tickers": tickers,
        "market_caps": market_caps,
        "price_history": price_history,
    }

    final_state = agent.invoke(initial_state)

    console.print(Rule("[bold green]Final Report[/bold green]"))
    console.print(final_state["report"])


if __name__ == "__main__":
    main()