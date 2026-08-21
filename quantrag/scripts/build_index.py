"""
Phase 2 — Building the full S&P 500 Qdrant index.

Agenda of this :
  1. Load 500 S&P 500 tickers
  2. Fetch 10-K filings for 2020-2024 (async, parallel)
  3. Embed and store ~360,000 chunks in Qdrant
"""

import asyncio
import sys
import os

# Fetching the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ingestion.bulk_loader import load_tickers, run_bulk_ingestion
from rich.console import Console
from rich.rule import Rule

console = Console()


async def main():
    console.print(Rule("[bold blue]QuantRAG Phase 2 — Build Index[/bold blue]"))

    # Loading the tickers
    tickers = load_tickers("data/sp500_tickers.txt")
    console.print(f"[green]Loaded {len(tickers)} tickers[/green]")

    # Run bulk ingestion
    # Set resume=True to pick up from where it stops
  
    await run_bulk_ingestion(
        tickers=tickers,
        years=[2020, 2021, 2022, 2023, 2024], #years being considered
        resume=True,
    )

    console.print(Rule("[bold green]Index build complete[/bold green]"))


if __name__ == "__main__":
    asyncio.run(main())