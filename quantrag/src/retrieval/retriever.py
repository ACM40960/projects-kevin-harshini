"""
Phase 2 — Two-stage retriever.
Stage 1: embedding search (top-20 from Qdrant)
Stage 2: cross-encoder reranking (top-5 final)
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rich.console import Console
from rich.table import Table

from src.retrieval.reranker import rerank

console = Console()
COLLECTION_NAME = "quantrag_phase2"


def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    ticker: Optional[str] = None,
    top_k: int = 5,
    use_reranker: bool = True,
    stage1_k: int = 20,
    max_retries: int = 2,
) -> List[dict]:
    """Two-stage retrieval with automatic retry on transient Qdrant timeouts."""

    for attempt in range(max_retries + 1):
        try:
            query_vector = model.encode(
                f"Represent this financial document for retrieval: {query}",
                normalize_embeddings=True
            ).tolist()

            search_filter = None
            if ticker:
                search_filter = Filter(must=[
                    FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
                ])

            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=search_filter,
                limit=stage1_k if use_reranker else top_k,
                with_payload=True,
            )

            candidates = [
                {
                    "text":     hit.payload["text"],
                    "citation": hit.payload["citation"],
                    "score":    round(hit.score, 4),
                    "ticker":   hit.payload["ticker"],
                    "section":  hit.payload["section"],
                    "year":     hit.payload["year"],
                }
                for hit in response.points
            ]

            if not use_reranker:
                return candidates

            return rerank(query, candidates, top_k=top_k)

        except Exception as e:
            if attempt < max_retries:
                console.print(
                    f"[yellow]  ⚠ Qdrant query failed (attempt {attempt+1}), "
                    f"retrying: {str(e)[:60]}[/yellow]"
                )
                continue
            console.print(f"[red]  ✗ Qdrant query failed after {max_retries+1} attempts[/red]")
            return []  # graceful degradation — empty result, not a crash

def display_results(query: str, results: List[dict]) -> None:
    """Pretty-print retrieval results with both scores."""
    console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Rank",     style="dim", width=5)
    table.add_column("Embed",    width=7)
    table.add_column("Rerank",   width=7)
    table.add_column("Citation", width=42)
    table.add_column("Preview")

    for i, r in enumerate(results):
        embed  = f"{r.get('embed_score', r.get('score', 0)):.4f}"
        rerank_score = f"{r.get('rerank_score', 0):.2f}" if "rerank_score" in r else "—"
        table.add_row(
            str(i + 1), embed, rerank_score,
            r["citation"],
            r["text"][:100].replace("\n", " ") + "..."
        )

    console.print(table)