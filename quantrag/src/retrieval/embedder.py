"""
Phase 1 — Embedding + Qdrant storage.
Loads BGE-large-en-v1.5 (runs locally, no API needed)
and stores chunk vectors in Qdrant with metadata.

Why BGE-large?
- 1024-dimensional vectors (richer than OpenAI ada-002's 1536 but more efficient)
- Specifically trained on financial and technical English
- Free, runs on CPU (slow but works), much faster on GPU
- Top of MTEB benchmark for retrieval tasks as of 2024
"""

import os
import uuid
from typing import List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter,
    FieldCondition, MatchValue
)
from rich.console import Console
from rich.progress import track

from src.ingestion.chunker import Chunk

load_dotenv()
console = Console()

# ── Constants ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTOR_DIM      = 1024          # BGE-large output dimension
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "quantrag_phase1")
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")


def load_embedder() -> SentenceTransformer:
    """
    Load BGE-large-en-v1.5 from HuggingFace (downloads on first run ~1.3GB).
    Subsequent runs load from local cache — fast.
    """
    console.print(f"[blue]Loading embedding model: {EMBEDDING_MODEL}[/blue]")
    console.print("[dim]  First run downloads ~1.3GB — subsequent runs use cache[/dim]")
    model = SentenceTransformer(EMBEDDING_MODEL)
    console.print(f"[green]  ✓ Model loaded (dim={VECTOR_DIM})[/green]")
    return model


def get_qdrant_client() -> QdrantClient:
    """Connect to local Qdrant instance."""
    client = QdrantClient(url=QDRANT_URL)
    console.print(f"[green]  ✓ Connected to Qdrant at {QDRANT_URL}[/green]")
    return client


def ensure_collection(client: QdrantClient) -> None:
    """
    Create the Qdrant collection if it doesn't exist.
    A collection is like a table — it stores vectors + metadata (payload).
    We use cosine distance because BGE embeddings are normalised.
    """
    existing = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE   # cosine similarity for normalised vectors
            )
        )
        console.print(f"[green]  ✓ Created Qdrant collection: {COLLECTION_NAME}[/green]")
    else:
        console.print(f"[dim]  Collection already exists: {COLLECTION_NAME}[/dim]")


def embed_and_store(chunks: List[Chunk], 
                    model: SentenceTransformer,
                    client: QdrantClient,
                    batch_size: int = 32) -> int:
    """
    Embed each chunk and store it in Qdrant with its metadata as payload.
    
    The payload (metadata) stored alongside each vector lets us:
    - Filter searches by ticker, year, or section
    - Return citations with every retrieved chunk
    - Reconstruct the original passage for the LLM
    
    Args:
        chunks:     List of Chunk objects from the chunker
        model:      Loaded SentenceTransformer model
        client:     Connected QdrantClient
        batch_size: Number of chunks to embed at once (higher = faster if GPU)
    
    Returns:
        Number of points stored
    """
    ensure_collection(client)
    
    console.print(f"\n[blue]Embedding {len(chunks)} chunks (batch_size={batch_size})...[/blue]")
    
    points = []
    
    # Process in batches for memory efficiency
    for i in track(range(0, len(chunks), batch_size), 
                   description="Embedding chunks..."):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        
        # BGE-large works best with this instruction prefix for retrieval
        # (from the BGE paper — it improves retrieval quality)
        prefixed = [f"Represent this financial document for retrieval: {t}" 
                    for t in texts]
        
        # Embed — returns numpy array of shape (batch_size, 1024)
        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,   # cosine similarity needs normalised vectors
            show_progress_bar=False
        )
        
        # Build Qdrant points — each point = vector + payload (metadata)
        for chunk, vector in zip(batch, vectors):
            point = PointStruct(
                id=str(uuid.uuid4()),    # unique ID for this chunk
                vector=vector.tolist(),
                payload={
                    # This payload comes back with every search result
                    "text":         chunk.text,
                    "ticker":       chunk.ticker,
                    "company_name": chunk.company_name,
                    "year":         chunk.year,
                    "section":      chunk.section,
                    "chunk_index":  chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "filing_date":  chunk.filing_date,
                    "citation":     chunk.citation(),
                }
            )
            points.append(point)
    
    # Upload all points to Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    info = client.get_collection(COLLECTION_NAME)
    console.print(
        f"[bold green]✓ Stored {len(points)} chunks. "
        f"Collection now has {info.points_count} total points.[/bold green]"
    )
    return len(points)