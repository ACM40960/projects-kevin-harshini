"""
Phase 2 — Bulk ingestion of S&P 500 10-K filings.

Override parallelism with the PARALLEL_MODE env var or the parameter:
  PARALLEL_MODE=auto   (default — recommended)
  PARALLEL_MODE=on     (force parallel, e.g. Windows GPU laptop)
  PARALLEL_MODE=off    (force single-stream, most conservative)
"""

import asyncio
import os
import json
import uuid
import concurrent.futures
from typing import List, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
)

from src.ingestion.edgar_loader import fetch_10k_sections
from src.ingestion.chunker import chunk_sections, Chunk
from src.retrieval.embedder import get_qdrant_client, ensure_collection
from src.device import setup_compute

load_dotenv()
console = Console()

YEARS         = [2020, 2021, 2022, 2023, 2024]
FETCH_WORKERS = 8
COLLECTION    = os.getenv("QDRANT_COLLECTION", "quantrag_phase2")
PARALLEL_MODE = os.getenv("PARALLEL_MODE", "auto")   # auto | on | off
FAILED_LOG    = "logs/failed_tickers.json"
PROGRESS_LOG  = "logs/ingestion_progress.json"


# ── Helpers ────────────────────────────────────────────────────────────────

def load_tickers(path: str = "data/sp500_tickers.txt") -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_progress() -> set:
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def save_progress(done: set) -> None:
    """Atomic write — crash mid-save never corrupts the file."""
    os.makedirs("logs", exist_ok=True)
    tmp = PROGRESS_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump([list(x) for x in done], f)
    os.replace(tmp, PROGRESS_LOG)


def load_failed() -> list:
    if os.path.exists(FAILED_LOG):
        with open(FAILED_LOG) as f:
            return json.load(f)
    return []


def save_failed(failed: list) -> None:
    os.makedirs("logs", exist_ok=True)
    tmp = FAILED_LOG + ".tmp"
    with open(tmp, "w") as f:
        json.dump(failed, f, indent=2)
    os.replace(tmp, FAILED_LOG)


# ── Model loading ──────────────────────────────────────────────────────────

def load_embedder_on_device(device: str):
    from sentence_transformers import SentenceTransformer
    console.print(f"[blue]Loading BGE-large on {device}...[/blue]")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    console.print(f"[green]  ✓ Model loaded on {device} (dim=1024)[/green]")
    return model


# ── Serialise a Chunk to a plain dict (for parallel worker IPC) ────────────

def chunk_to_dict(c: Chunk) -> dict:
    return {
        "text":         c.text,
        "ticker":       c.ticker,
        "company_name": c.company_name,
        "year":         c.year,
        "section":      c.section,
        "chunk_index":  c.chunk_index,
        "total_chunks": c.total_chunks,
        "filing_date":  c.filing_date,
        "citation":     c.citation(),
    }


# ── Parallel embed worker — runs in a SEPARATE PROCESS (CUDA/CPU) ──────────

def _parallel_embed_worker(args: tuple) -> list:
    """
    Runs in a separate process (ProcessPoolExecutor).
    Each process loads its own model on the given device and embeds a batch.
    Used only when device.py decides parallel is safe (CUDA / multi-core CPU).
    Never used on MPS.
    """
    chunk_dicts, device, batch_size = args

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

    texts = [
        f"Represent this financial document for retrieval: {c['text']}"
        for c in chunk_dicts
    ]
    vectors = model.encode(
        texts, normalize_embeddings=True,
        batch_size=batch_size, show_progress_bar=False,
    )

    return [
        {
            "id":     str(uuid.uuid4()),
            "vector": vector.tolist(),
            "payload": cd,
        }
        for cd, vector in zip(chunk_dicts, vectors)
    ]


# ── Single-stream embed (MPS / default) ────────────────────────────────────

def embed_filing_single(
    chunks: List[Chunk],
    model,
    batch_size: int,
) -> list:
    """
    Embed one filing's chunks with the shared pre-loaded model.
    Used on MPS and as the safe default. Returns list of Qdrant point dicts.
    """
    points = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i: i + batch_size]
        texts = [
            f"Represent this financial document for retrieval: {c.text}"
            for c in batch
        ]
        vectors = model.encode(
            texts, normalize_embeddings=True,
            batch_size=batch_size, show_progress_bar=False,
        )
        for chunk, vector in zip(batch, vectors):
            points.append({
                "id":     str(uuid.uuid4()),
                "vector": vector.tolist(),
                "payload": chunk_to_dict(chunk),
            })
    return points


def store_points(client, point_dicts: list) -> int:
    """Upsert a list of point dicts into Qdrant."""
    from qdrant_client.models import PointStruct
    if not point_dicts:
        return 0
    points = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in point_dicts
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    return len(points)


# ── Async fetch workers ────────────────────────────────────────────────────

async def fetch_worker(work_queue, result_queue, semaphore):
    """Fetch + chunk one filing; put result on the result queue."""
    while True:
        item = await work_queue.get()
        if item is None:
            work_queue.task_done()
            break
        ticker, year = item
        async with semaphore:
            try:
                sections = await asyncio.to_thread(
                    fetch_10k_sections, ticker, year
                )
                total = sum(
                    len(sections.get(k, ""))
                    for k in ["risk_factors", "mda", "market_risk"]
                )
                if total < 500:
                    await result_queue.put((ticker, year, [], "too_short"))
                else:
                    chunks = await asyncio.to_thread(chunk_sections, sections)
                    await result_queue.put((ticker, year, chunks, None))
            except Exception as e:
                await result_queue.put((ticker, year, [], str(e)[:100]))
        work_queue.task_done()


# ── Main ───────────────────────────────────────────────────────────────────

async def run_bulk_ingestion(
    tickers: List[str],
    years: List[int] = None,
    resume: bool = True,
    parallel_mode: str = None,
) -> None:
    """
    Universal, crash-resilient, parallel-capable bulk ingestion.

    Args:
        tickers:       tickers to index
        years:         fiscal years per ticker
        resume:        skip already-completed (ticker, year) pairs
        parallel_mode: "auto" | "on" | "off"  (overrides PARALLEL_MODE env)
    """
    if years is None:
        years = YEARS
    mode = parallel_mode or PARALLEL_MODE

    console.print("[bold]Detecting compute hardware...[/bold]")
    cfg = setup_compute(parallel_mode=mode)
    device      = cfg["device"]
    batch_size  = cfg["batch_size"]
    parallel    = cfg["parallel"]
    num_workers = cfg["num_workers"]

    already_done = load_progress() if resume else set()
    failed       = load_failed()  if resume else []

    work = [
        (t, y) for t in tickers for y in years
        if (t, y) not in already_done
    ]

    console.print(
        f"\n[bold blue]QuantRAG Phase 2 — Bulk Ingestion[/bold blue]\n"
        f"  Total filings:  {len(tickers) * len(years):,}\n"
        f"  Already done:   {len(already_done):,}\n"
        f"  Remaining:      {len(work):,}\n"
        f"  Device:         {device}  |  Batch: {batch_size}\n"
        f"  Parallel embed: {parallel}  ({num_workers} workers)\n"
    )

    if not work:
        console.print("[green]All filings already indexed — nothing to do[/green]")
        return

    client = get_qdrant_client()
    ensure_collection(client)

    # ── Set up embedding strategy ──
    # Single-stream: load one shared model (MPS / default)
    # Parallel: use a ProcessPoolExecutor of embed workers (CUDA / CPU)
    shared_model = None
    executor     = None
    if parallel:
        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
        console.print(
            f"[dim]  Parallel embedding via {num_workers} processes on {device}[/dim]"
        )
    else:
        shared_model = load_embedder_on_device(device)

    # ── Async fetch + streaming embed ──
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
    loop      = asyncio.get_event_loop()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("[cyan]Filings", total=total)

        while processed < total:
            ticker, year, chunks, error = await result_queue.get()
            processed += 1

            if error:
                failed.append({"ticker": ticker, "year": year, "reason": error})
                if processed % 20 == 0:
                    save_failed(failed)
                progress.advance(task)
                continue

            if not chunks:
                progress.advance(task)
                continue

            try:
                if parallel:
                    # Embed this filing in a worker process
                    chunk_dicts = [chunk_to_dict(c) for c in chunks]
                    point_dicts = await loop.run_in_executor(
                        executor,
                        _parallel_embed_worker,
                        (chunk_dicts, device, batch_size),
                    )
                else:
                    # Single-stream embed with shared model
                    point_dicts = await asyncio.to_thread(
                        embed_filing_single, chunks, shared_model, batch_size
                    )

                # Store and checkpoint immediately
                n = store_points(client, point_dicts)
                already_done.add((ticker, year))
                save_progress(already_done)

                console.print(
                    f"[green]  ✓ {ticker} {year}: {n} chunks stored[/green]"
                )

            except Exception as e:
                failed.append({"ticker": ticker, "year": year,
                               "reason": f"embed_error: {str(e)[:80]}"})
                console.print(
                    f"[red]  ✗ {ticker} {year} embed failed: {str(e)[:60]}[/red]"
                )

            progress.advance(task)

    await asyncio.gather(*fetchers, return_exceptions=True)

    if executor:
        executor.shutdown(wait=True)

    save_progress(already_done)
    save_failed(failed)

    info = client.get_collection(COLLECTION)
    console.print(
        f"\n[bold green]Ingestion complete![/bold green]\n"
        f"  Filings done:   {len(already_done):,}\n"
        f"  Failed:         {len(failed):,}\n"
        f"  Qdrant points:  {info.points_count:,}\n"
    )