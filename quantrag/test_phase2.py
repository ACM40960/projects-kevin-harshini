"""
Phase 2 — 5-company test.
Run:  python test_phase2.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.bulk_loader import run_bulk_ingestion
from rich.console import Console
from rich.rule import Rule

console = Console()


async def main():
    console.print(Rule("[bold blue]Phase 2 — 5 Company Test[/bold blue]"))

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    await run_bulk_ingestion(
        tickers=test_tickers,
        years=[2023],
        resume=False,
    )

    console.print(Rule("[bold green]Test complete[/bold green]"))


if __name__ == "__main__":
    asyncio.run(main())