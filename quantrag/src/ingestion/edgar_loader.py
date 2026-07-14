"""
Phase 1 — EDGAR document loader.
Distinguishes real sections from TOC entries using double-newline detection.
"""

import re
import os
from dotenv import load_dotenv
from edgar import Company, set_identity
from rich.console import Console
from rich.panel import Panel

load_dotenv()
console = Console()
set_identity("Harshini Student harshini@email.com")


SECTION_PATTERNS = {
    "risk_factors": [
        r"Item\s+1A\.\s+Risk\s+Factors",
        r"Item\s+1A\.",
    ],
    "mda": [
        r"Item\s+7\.\s+Management",
        r"Item\s+7\.",
    ],
    "market_risk": [
        r"Item\s+7A\.\s+Quantitative",
        r"Item\s+7A\.",
    ],
}

STOP_PATTERNS = {
    "risk_factors": r"Item\s+1[BC]\.",
    "mda":          r"Item\s+7A\.",
    "market_risk":  r"Item\s+8\.",
}


def is_real_section(full_text: str, match_end: int) -> tuple:
    """
    Determine if a regex match is the real section or just a TOC entry.

    How we tell the difference:
    ┌─────────────────────────────────────────────────────┐
    │ TOC entry:                                          │
    │ "Item 7.  Management's Discussion...      21        │
    │  Item 7A. Quantitative..."                          │
    │  → single newlines, page number, next TOC item      │
    │  → NO double newline (\n\n) in first 300 chars      │
    │                                                     │
    │ Real section:                                       │
    │ "Item 7. Management's Discussion...                 │
    │                                                     │
    │  Fiscal 2025 Highlights..."                         │
    │  → double newline before paragraph text             │
    │  → many letters 100+ chars in (past the title)     │
    └─────────────────────────────────────────────────────┘

    Returns (is_real: bool, reason: str)
    """
    after = full_text[match_end: match_end + 500]

    # Signal 1 — double newline within first 300 chars
    # Real sections always have \n\n before first paragraph
    # TOC entries only have \n before the next TOC line
    has_double_newline = '\n\n' in after[:300]

    # Signal 2 — substantial letters in chars 100-400
    # Goes PAST the title line into real content
    # TOC: chars 100-400 contain more short TOC lines
    # Real section: chars 100-400 contain paragraph sentences
    letters_deep = len(re.findall(r'[a-zA-Z]', after[100:400]))

    is_real = has_double_newline and letters_deep >= 80

    reason = (
        f"double_newline={has_double_newline}, "
        f"deep_letters={letters_deep} → "
        f"{'REAL SECTION ✓' if is_real else 'TOC/skip ✗'}"
    )
    return is_real, reason


def find_real_section_start(full_text: str, section_key: str) -> int:
    """
    Find character position of the REAL section header (not TOC).
    Iterates all pattern matches and applies is_real_section() to each.
    """
    for pattern in SECTION_PATTERNS[section_key]:
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

        for match in matches:
            real, reason = is_real_section(full_text, match.end())
            console.print(
                f"[dim]  pos {match.start():>8,} → {reason}[/dim]"
            )
            if real:
                return match.start()

    return -1


def extract_section(full_text: str, section_key: str) -> str:
    """
    Extract a complete section — from its real header to the next section's header.
    """
    start = find_real_section_start(full_text, section_key)
    if start == -1:
        return ""

    # Find next section — search only AFTER our start + 100 chars
    stop_match = re.search(
        STOP_PATTERNS[section_key],
        full_text[start + 100:],
        re.IGNORECASE
    )

    if stop_match:
        end = start + 100 + stop_match.start()
    else:
        end = start + 80000   # fallback — take up to 80K chars

    return clean_text(full_text[start:end])


def clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'20\d\d Form 10-K \| \d+', '', text)
    return text.strip()


def get_raw_text(filing) -> str:
    """Try multiple strategies to get raw text — handles all edgartools versions."""

    # Strategy 1 — filing.text()
    try:
        text = filing.text()
        if text and len(str(text)) > 1000:
            return str(text)
    except Exception:
        pass

    # Strategy 2 — str(filing.obj())
    try:
        text = str(filing.obj())
        if len(text) > 1000:
            return text
    except Exception:
        pass

    # Strategy 3 — filing.html() + strip tags
    try:
        html = filing.html()
        if html and len(str(html)) > 1000:
            return re.sub(r'<[^>]+>', ' ', str(html))
    except Exception:
        pass

    return ""


def fetch_10k_sections(ticker: str, year: int = 2023) -> dict:
    """
    Fetch 10-K and extract Item 1A, Item 7, Item 7A sections.
    """
    console.print(
        f"\n[bold blue]Fetching 10-K for {ticker} ({year})...[/bold blue]"
    )

    company = Company(ticker)
    filings = company.get_filings(form="10-K")

    # Find the right filing
    target = None
    for f in filings:
        if f.filing_date and f.filing_date.year in (year, year + 1):
            target = f
            break

    if not target:
        console.print("[yellow]  ⚠ Exact year not found — using most recent[/yellow]")
        target = filings[0]

    console.print(f"[green]  ✓ Filing: {target.filing_date}[/green]")

    full_text = get_raw_text(target)
    if not full_text or len(full_text) < 1000:
        raise ValueError(f"Could not get text for {ticker}")

    console.print(f"[green]  ✓ Raw text: {len(full_text):,} chars[/green]")

    result = {
        "ticker":       ticker.upper(),
        "company_name": company.name,
        "year":         year,
        "filing_date":  str(target.filing_date),
    }

    section_labels = {
        "risk_factors": "Item 1A (Risk Factors)",
        "mda":          "Item 7  (MD&A)",
        "market_risk":  "Item 7A (Market Risk)",
    }

    for key, label in section_labels.items():
        console.print(f"\n[blue]  Extracting {label}...[/blue]")
        text = extract_section(full_text, key)

        if text and len(text) > 200:
            result[key] = text
            console.print(
                f"[bold green]  ✓ {label}: {len(text):,} chars[/bold green]"
            )
        else:
            result[key] = ""
            console.print(f"[yellow]  ⚠ {label}: not found[/yellow]")

    return result


if __name__ == "__main__":
    result = fetch_10k_sections("AAPL", 2023)
    console.print(Panel(
        f"[bold]Company:[/bold]      {result['company_name']}\n"
        f"[bold]Filing date:[/bold]  {result['filing_date']}\n"
        f"[bold]Risk Factors:[/bold] {len(result['risk_factors']):,} chars\n"
        f"[bold]MD&A:[/bold]         {len(result['mda']):,} chars\n"
        f"[bold]Market Risk:[/bold]  {len(result['market_risk']):,} chars",
        title="[green]10-K extraction result[/green]"
    ))