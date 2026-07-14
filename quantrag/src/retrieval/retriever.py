"""
Phase 1 — Retriever.
Uses qdrant-client >= 1.7 API (query_points instead of search).
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.table import Table

console = Console()
COLLECTION_NAME = "quantrag_phase1"


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    ticker: Optional[str] = None,
    top_k: int = 5,
) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.

    What changed from old code:
    - client.search()       → removed in qdrant-client >= 1.7
    - client.query_points() → new method, returns QueryResponse object
    - results.points        → list of ScoredPoint objects
    """

    # Embed query — same model and prefix used during indexing
    query_vector = model.encode(
        f"Represent this financial document for retrieval: {query}",
        normalize_embeddings=True
    ).tolist()

    # Optional filter — restrict search to one company
    search_filter = None
    if ticker:
        search_filter = Filter(
            must=[FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker.upper())
            )]
        )

    # NEW API — query_points (qdrant-client >= 1.7)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k,
        with_payload=True,
    )

    # response.points is a list of ScoredPoint objects
    retrieved = []
    for hit in response.points:
        retrieved.append({
            "text":     hit.payload["text"],
            "citation": hit.payload["citation"],
            "score":    round(hit.score, 4),
            "ticker":   hit.payload["ticker"],
            "section":  hit.payload["section"],
            "year":     hit.payload["year"],
        })

    return retrieved


def display_results(query: str, results: List[dict]) -> None:
    """Pretty-print retrieval results."""
    console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Rank",     style="dim", width=5)
    table.add_column("Score",    width=7)
    table.add_column("Citation", width=45)
    table.add_column("Preview")

    for i, r in enumerate(results):
        table.add_row(
            str(i + 1),
            f"{r['score']:.4f}",
            r["citation"],
            r["text"][:120].replace("\n", " ") + "..."
        )

    console.print(table)