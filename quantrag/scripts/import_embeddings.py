"""
Phase 2 — Step 3 (LOCAL): Import Colab-embedded vectors into local Qdrant.

Reads the .parquet shard files produced by quantrag_embed_colab.ipynb
and bulk-upserts them into your LOCAL Qdrant (running in Docker, same
as always). No embedding happens here — just fast bulk insertion of
already-computed vectors.

Setup before running:
    1. Download quantrag_embeddings.zip from Colab
    2. Unzip it into: data/embeddings_from_colab/
       (should contain embeddings_part_0000.parquet, _0001.parquet, ...)

Usage:
    python scripts/import_embeddings.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TaskProgressColumn, TimeRemainingColumn
)

from src.retrieval.embedder import get_qdrant_client, ensure_collection

load_dotenv()
console = Console()

INPUT_DIR   = "data/embeddings_from_colab"
COLLECTION  = os.getenv("QDRANT_COLLECTION", "quantrag_phase2")
BATCH_SIZE  = 64   # points per Qdrant upsert call — fast, local network only


def find_shards(input_dir: str) -> list:
    """Find all .parquet shard files, sorted by shard number."""
    pattern = os.path.join(input_dir, "embeddings_part_*.parquet")
    shards  = sorted(glob.glob(pattern))
    return shards


def import_shard(shard_path: str, client, collection: str) -> int:
    """
    Load one parquet shard and upsert its rows into Qdrant.
    Returns number of points imported.

    Uses the deterministic 'id' column from export_chunks.py — so
    re-running this script (e.g. after adding more shards) never
    creates duplicate points. Qdrant upsert with the same id overwrites.
    """
    from qdrant_client.models import PointStruct

    df = pd.read_parquet(shard_path)

    points = []
    for _, row in df.iterrows():
        points.append(PointStruct(
            id=row["id"],
            vector=list(row["vector"]),
            payload={
                "text":         row["text"],
                "ticker":       row["ticker"],
                "company_name": row["company_name"],
                "year":         int(row["year"]),
                "section":      row["section"],
                "chunk_index":  int(row["chunk_index"]),
                "total_chunks": int(row["total_chunks"]),
                "filing_date":  row["filing_date"],
                "citation":     row["citation"],
            }
        ))

    # Upsert in batches — avoids one giant request for large shards
    imported = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i: i + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)
        imported += len(batch)

    return imported


def run_import():
    console.print(f"\n[bold blue]QuantRAG — Import Colab Embeddings[/bold blue]")

    shards = find_shards(INPUT_DIR)

    if not shards:
        console.print(
            f"[red]No shard files found in {INPUT_DIR}/[/red]\n"
            f"[dim]Expected files like: embeddings_part_0000.parquet[/dim]\n"
            f"[dim]Did you unzip quantrag_embeddings.zip into {INPUT_DIR}/ ?[/dim]"
        )
        return

    console.print(f"  Found {len(shards)} shard file(s) in {INPUT_DIR}/\n")

    client = get_qdrant_client()
    ensure_collection(client)

    total_imported = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), MofNCompleteColumn(), TaskProgressColumn(), TimeRemainingColumn(),
    ) as progress:

        task = progress.add_task("[cyan]Importing shards", total=len(shards))

        for shard_path in shards:
            n = import_shard(shard_path, client, COLLECTION)
            total_imported += n
            console.print(
                f"[green]  ✓ {os.path.basename(shard_path)}: "
                f"{n:,} points imported[/green]"
            )
            progress.advance(task)

    info = client.get_collection(COLLECTION)
    console.print(
        f"\n[bold green]Import complete![/bold green]\n"
        f"  Points imported this run: {total_imported:,}\n"
        f"  Total points in Qdrant:   {info.points_count:,}\n"
        f"\n[bold]Your local Qdrant is now fully populated.[/bold]\n"
        f"Continue with Phase 2 evaluation or Phase 3 as normal."
    )


if __name__ == "__main__":
    run_import()