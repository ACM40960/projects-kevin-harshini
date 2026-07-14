"""
Phase 1 — End-to-end test.
Run this to verify the full pipeline works:
EDGAR → chunks → embeddings → Qdrant → retrieval → Claude answer

Usage:
    python main_phase1.py
"""

import os
from dotenv import load_dotenv
#import anthropic
from src.llm import get_llm_response
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.ingestion.edgar_loader import fetch_10k_sections
from src.ingestion.chunker import chunk_sections
from src.retrieval.embedder import load_embedder, get_qdrant_client, embed_and_store
from src.retrieval.retriever import retrieve, display_results

load_dotenv()
console = Console()


def ask_claude(query: str, context_chunks: list) -> str:
    """
    Step 5 — Generate a cited answer.
    Provider is controlled by LLM_PROVIDER in .env
    Currently: groq | switch to anthropic in Phase 3
    """
    context = "\n\n---\n\n".join([
        f"SOURCE: {c['citation']}\n{c['text']}"
        for c in context_chunks
    ])

    prompt = f"""You are a financial analyst assistant.
Answer the following question based ONLY on the provided SEC filing excerpts.
Always cite the source of every claim using the SOURCE labels provided.
If the answer is not in the excerpts, say so — do not make up information.

QUESTION: {query}

FILING EXCERPTS:
{context}

ANSWER:"""

    return get_llm_response(prompt)


def main():
    console.print(Rule("[bold blue]QuantRAG Phase 1 — End-to-End Test[/bold blue]"))
    
    # ── Step 1: Fetch 10-K ──────────────────────────────────────────────────
    console.print("\n[bold]Step 1: Fetching AAPL 10-K from SEC EDGAR[/bold]")
    sections = fetch_10k_sections("AAPL", 2023)
    
    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    console.print("\n[bold]Step 2: Chunking into section-aware pieces[/bold]")
    chunks = chunk_sections(sections)
    
    # ── Step 3: Embed + store ───────────────────────────────────────────────
    console.print("\n[bold]Step 3: Embedding + storing in Qdrant[/bold]")
    model  = load_embedder()
    client = get_qdrant_client()
    embed_and_store(chunks, model, client)
    
    # ── Step 4: Retrieve ────────────────────────────────────────────────────
    console.print("\n[bold]Step 4: Retrieval test[/bold]")
    query = "What are Apple's main supply chain and geopolitical risks?"
    results = retrieve(query, model, client, ticker="AAPL", top_k=5)
    display_results(query, results)
    
    # ── Step 5: Claude answer ───────────────────────────────────────────────
    console.print("\n[bold]Step 5: Claude generates a cited answer[/bold]")
    answer = ask_claude(query, results)
    
    console.print(Panel(
        answer,
        title="[bold green]Claude's cited answer[/bold green]",
        border_style="green"
    ))
    
    console.print(Rule("[bold green]Phase 1 complete ✓[/bold green]"))
    console.print("\n[dim]Next: Phase 2 — ingest all S&P 500 10-Ks with parallel fetching[/dim]")


if __name__ == "__main__":
    main()