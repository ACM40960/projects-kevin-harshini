"""
Phase 1 — Section-aware chunker.
Splits each 10-K section into overlapping chunks
with metadata attached to every chunk.
"""

from dataclasses import dataclass
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

console = Console()


@dataclass
class Chunk:
    """
    A single chunk of text with full metadata.
    This metadata travels with the chunk into Qdrant
    and comes back with every retrieval result.
    """
    text: str
    ticker: str
    company_name: str
    year: int
    section: str          # "risk_factors" | "mda" | "market_risk"
    chunk_index: int
    total_chunks: int
    filing_date: str
    
    def citation(self) -> str:
        """Human-readable citation string for this chunk."""
        section_map = {
            "risk_factors": "Item 1A (Risk Factors)",
            "mda":          "Item 7 (MD&A)",
            "market_risk":  "Item 7A (Market Risk)",
        }
        return (
            f"{self.company_name} ({self.ticker}) "
            f"10-K FY{self.year}, "
            f"{section_map.get(self.section, self.section)}, "
            f"chunk {self.chunk_index + 1}/{self.total_chunks}"
        )


def chunk_sections(sections: dict, 
                   chunk_size: int = 400,
                   chunk_overlap: int = 50) -> List[Chunk]:
    """
    Take the output of fetch_10k_sections() and split each section
    into overlapping chunks with metadata.
    
    Why 400 words with 50-word overlap?
    - 400 words ≈ 1 paragraph to 2 paragraphs — enough context for BGE-large
    - 50-word overlap ensures sentences at chunk boundaries are not lost
    - At 400 words, a 200-page 10-K becomes ~300 chunks — manageable in Qdrant
    
    Args:
        sections:      Output from fetch_10k_sections()
        chunk_size:    Target chunk size in characters (~400 words ≈ 2000 chars)
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of Chunk objects, each with full metadata
    """
    
    # RecursiveCharacterTextSplitter tries to split at paragraph boundaries first,
    # then sentences, then words — preserving semantic units as much as possible
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 5,   # ~5 chars per word on average
        chunk_overlap=chunk_overlap * 5,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    section_keys = ["risk_factors", "mda", "market_risk"]
    all_chunks: List[Chunk] = []
    
    for section_key in section_keys:
        text = sections.get(section_key, "")
        
        if not text or len(text) < 100:
            console.print(f"[yellow]  ⚠ Skipping {section_key} — too short[/yellow]")
            continue
        
        # Split this section into raw text chunks
        raw_chunks = splitter.split_text(text)
        
        console.print(
            f"[green]  ✓ {section_key}: {len(text)} chars → "
            f"{len(raw_chunks)} chunks[/green]"
        )
        
        # Wrap each raw chunk in a Chunk dataclass with metadata
        for i, raw_text in enumerate(raw_chunks):
            chunk = Chunk(
                text=raw_text,
                ticker=sections["ticker"],
                company_name=sections["company_name"],
                year=sections["year"],
                section=section_key,
                chunk_index=i,
                total_chunks=len(raw_chunks),
                filing_date=sections["filing_date"],
            )
            all_chunks.append(chunk)
    
    console.print(f"\n[bold green]Total chunks created: {len(all_chunks)}[/bold green]")
    return all_chunks


if __name__ == "__main__":
    # Test with mock data
    mock_sections = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "year": 2023,
        "filing_date": "2024-01-01",
        "risk_factors": "Global supply chain risks " * 200,
        "mda": "Revenue grew 15% driven by iPhone sales " * 200,
        "market_risk": "The company is exposed to foreign exchange risk " * 100,
    }
    
    chunks = chunk_sections(mock_sections)
    print(f"\nFirst chunk text:\n{chunks[0].text[:200]}")
    print(f"\nCitation: {chunks[0].citation()}")