"""
Phase 2 - Exporting chunks for Colab embedding.

Code Description : 
Fetches all S&P 500 filings and chunks them EXACTLY like the local
pipeline already does — but does NOT embed. Writes plain text chunks
with metadata to a JSONL file that Colab will embed on GPU.
"""




import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import json
import uuid
from typing import List, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn
)

from src.ingestion.edgar_loader import fetch_10k_sections
from src.ingestion.chunker import chunk_sections, Chunk

load_dotenv()
console = Console()

YEARS         = [2020, 2021, 2022, 2023, 2024]
FETCH_WORKERS = 8  # how many companies we fetch at the same time (async)
OUTPUT_DIR    = "data/chunks_export"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "chunks.jsonl")
PROGRESS_LOG  = os.path.join(OUTPUT_DIR, "export_progress.json")
FAILED_LOG    = os.path.join(OUTPUT_DIR, "export_failed.json")


def load_tickers(path: str = "data/sp500_tickers.txt") -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress() -> set:
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_progress(done: set) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # write to a temp file first, then rename — avoids a half-written,
    # corrupted progress file if the process dies mid-save
    tmp = PROGRESS_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump([list(x) for x in done], f)
    os.replace(tmp, PROGRESS_LOG)


def save_failed(failed: list) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FAILED_LOG, "w") as f:
        json.dump(failed, f, indent=2)


def chunk_to_record(c: Chunk) -> dict:
    #Convert a Chunk to a JSON-serialisable dict with a DETERMINISTIC id.
    citation = c.citation()
    # same citation always produces the same id — so re-running this
    # script (or re-importing later) never creates duplicate entries
    deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, citation))

    return {
        "id":           deterministic_id,
        "text":         c.text,
        "ticker":       c.ticker,
        "company_name": c.company_name,
        "year":         c.year,
        "section":      c.section,
        "chunk_index":  c.chunk_index,
        "total_chunks": c.total_chunks,
        "filing_date":  c.filing_date,
        "citation":     citation,
    }


async def fetch_worker(work_queue, result_queue, semaphore):
    # one of several workers pulling jobs off the shared queue and
    # fetching them concurrently, instead of one company at a time
    while True:
        item = await work_queue.get()
        if item is None:
            # None is our "no more work, shut down" signal for this worker
            work_queue.task_done()
            break
        ticker, year = item
        async with semaphore:
            try:
                sections = await asyncio.to_thread(fetch_10k_sections, ticker, year)
                total = sum(
                    len(sections.get(k, ""))
                    for k in ["risk_factors", "mda", "market_risk"]
                )
                if total < 500:
                    # extraction basically failed (empty/near-empty sections) —
                    # log it as a failure rather than storing junk chunks
                    await result_queue.put((ticker, year, [], "too_short"))
                else:
                    chunks = await asyncio.to_thread(chunk_sections, sections)
                    await result_queue.put((ticker, year, chunks, None))
            except Exception as e:
                await result_queue.put((ticker, year, [], str(e)[:100]))
        work_queue.task_done()


async def run_export(tickers: List[str], years: List[int] = None, resume: bool = True):
    if years is None:
        years = YEARS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    already_done = load_progress() if resume else set()
    failed       = []

    # skip anything we've already successfully exported in a prior run

    work = [
        (t, y) for t in tickers for y in years
        if (t, y) not in already_done
    ]

    console.print(
        f"\n[bold blue]Phase 2 — Export Chunks for Colab[/bold blue]\n"
        f"  Total filings:  {len(tickers) * len(years):,}\n"
        f"  Already done:   {len(already_done):,}\n"
        f"  Remaining:      {len(work):,}\n"
        f"  Output:         {OUTPUT_FILE}\n"
    )

    if not work:
        console.print("[green]All filings already exported[/green]")
        return
    # "a" (append) mode — safe to resume, we never overwrite past chunks
    out_f = open(OUTPUT_FILE, "a", encoding="utf-8")

    work_queue   = asyncio.Queue()
    result_queue = asyncio.Queue(maxsize=FETCH_WORKERS * 2)
    semaphore    = asyncio.Semaphore(FETCH_WORKERS)

    for item in work:
        work_queue.put_nowait(item)
    for _ in range(FETCH_WORKERS):
        work_queue.put_nowait(None)

    fetchers = [
        asyncio.create_task(fetch_worker(work_queue, result_queue, semaphore))
        for _ in range(FETCH_WORKERS)
    ]

    processed = 0
    total     = len(work)
    total_chunks_written = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(), TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Exporting filings", total=total)

        while processed < total:
            ticker, year, chunks, error = await result_queue.get()
            processed += 1

            if error:
                failed.append({"ticker": ticker, "year": year, "reason": error})
                progress.advance(task)
                continue

            if chunks:
                for c in chunks:
                    record = chunk_to_record(c)
                    out_f.write(json.dumps(record) + "\n")
                    total_chunks_written += 1
                out_f.flush()  # ensure data hits disk immediately, crash-safe

                already_done.add((ticker, year))
                save_progress(already_done)
                # saving progress after EVERY filing (not just at the end)
                # is what makes this safe to interrupt at any point
                console.print(
                    f"[green]  ✓ {ticker} {year}: {len(chunks)} chunks written[/green]"
                )

            progress.advance(task)

    await asyncio.gather(*fetchers, return_exceptions=True)
    out_f.close()

    save_progress(already_done)
    save_failed(failed)

    console.print(
        f"\n[bold green]Export complete![/bold green]\n"
        f"  Filings exported: {len(already_done):,}\n"
        f"  Failed:           {len(failed):,}\n"
        f"  Total chunks:     {total_chunks_written:,}\n"
        f"  File:             {OUTPUT_FILE}\n"
        f"\n[bold]Next step:[/bold] upload {OUTPUT_FILE} to Colab and run "
        f"quantrag_embed_colab.ipynb"
    )


if __name__ == "__main__":
    tickers = load_tickers()
    asyncio.run(run_export(tickers))