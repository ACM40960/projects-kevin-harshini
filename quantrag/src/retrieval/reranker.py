"""
Phase 2 — Cross-encoder reranker with diversity filtering.
"""

from typing import List
from sentence_transformers import CrossEncoder
from rich.console import Console

console = Console()

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

DIVERSITY_THRESHOLD = 0.75   # Jaccard similarity above this = "too similar"


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        console.print(f"[blue]Loading reranker: {_RERANKER_MODEL}[/blue]")
        _reranker = CrossEncoder(_RERANKER_MODEL)
        console.print("[green]  ✓ Reranker loaded[/green]")
    return _reranker


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Generic word-overlap similarity between two passages.
    Works on any text — no company, phrase, or domain hardcoding.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union else 0.0


def _diversify(candidates: List[dict], top_k: int) -> List[dict]:
    """
    Greedily select up to top_k candidates from a relevance-sorted list,
    skipping any candidate that is near-duplicate (by word overlap) of
    something already selected.

    This is a generic technique (similar in spirit to Maximal Marginal
    Relevance) — no domain knowledge, works for any query/company/topic.
    """
    selected: List[dict] = []

    for candidate in candidates:
        is_duplicate = any(
            _jaccard_similarity(candidate["text"][:300], chosen["text"][:300])
            > DIVERSITY_THRESHOLD
            for chosen in selected
        )
        if not is_duplicate:
            selected.append(candidate)
        if len(selected) >= top_k:
            break

    # If diversity filtering left us short (e.g. all candidates were
    # near-duplicates), backfill with the next-best remaining ones
    # rather than returning fewer than requested.
    if len(selected) < top_k:
        selected_texts = {s["text"] for s in selected}
        for candidate in candidates:
            if candidate["text"] not in selected_texts:
                selected.append(candidate)
                selected_texts.add(candidate["text"])
            if len(selected) >= top_k:
                break

    return selected


def rerank(query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
    """
    Rerank retrieved chunks using the cross-encoder, then apply a
    diversity filter so the final result set isn't dominated by
    near-duplicate boilerplate passages.
    """
    if not candidates:
        return []

    reranker = get_reranker()

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        candidate["embed_score"] = candidate.pop("score")

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    return _diversify(reranked, top_k)