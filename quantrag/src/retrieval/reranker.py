"""
Phase 2 — Cross-encoder reranker.

Two-stage retrieval:
  Stage 1 — embedding search (fast): top-20 candidates from Qdrant
  Stage 2 — cross-encoder reranking (precise): top-5 final

The cross-encoder reads (query, chunk) together as a pair — far more
accurate than the embedding model which encodes them separately.
"""

from typing import List
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder (downloads ~85MB on first run)."""
    global _reranker
    if _reranker is None:
        console.print(f"[blue]Loading reranker: {_RERANKER_MODEL}[/blue]")
        _reranker = CrossEncoder(_RERANKER_MODEL)
        console.print("[green]  ✓ Reranker loaded[/green]")
    return _reranker


def rerank(query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
    """
    Rerank retrieved chunks using the cross-encoder.

    Args:
        query:      the original search query
        candidates: chunk dicts from Stage 1 retrieval
        top_k:      how many to return after reranking

    Returns:
        Top-k reranked chunks, each with a new 'rerank_score' field
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Build (query, passage) pairs and score them together
    pairs  = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        candidate["embed_score"]  = candidate.pop("score")

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]