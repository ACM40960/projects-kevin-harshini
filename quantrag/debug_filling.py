"""
Diagnostic script — shows us exactly what the raw 10-K text looks like
so we can write patterns that actually match it.
Run: python debug_filing.py
"""

import re
from edgar import Company, set_identity

set_identity("Harshini Student harshini@email.com")

company = Company("AAPL")
filings = company.get_filings(form="10-K")
filing  = filings[0]   # most recent

print(f"Filing date: {filing.filing_date}")

# Get raw text
text = filing.text()
if not text:
    text = str(filing.obj())

full = str(text)
print(f"Total characters: {len(full):,}")
print(f"Total lines: {full.count(chr(10)):,}")

# ── Show us the first 3000 chars so we see the structure ──
print("\n" + "="*60)
print("FIRST 3000 CHARACTERS:")
print("="*60)
print(full[:3000])

# ── Search for any line containing "item" ──
print("\n" + "="*60)
print("ALL LINES CONTAINING 'item' (first 60):")
print("="*60)
item_lines = [
    (i, line.strip())
    for i, line in enumerate(full.splitlines())
    if "item" in line.lower() and len(line.strip()) < 200
]
for i, (lineno, line) in enumerate(item_lines[:60]):
    print(f"  line {lineno:5d}: {line}")

# ── Search specifically for item 1a patterns ──
print("\n" + "="*60)
print("SEARCHING FOR ITEM 1A / RISK FACTORS:")
print("="*60)
for pattern in [r"item\s*1a", r"risk factor", r"ITEM 1A", r"Item 1A"]:
    matches = list(re.finditer(pattern, full, re.IGNORECASE))
    print(f"  Pattern '{pattern}': {len(matches)} matches")
    for m in matches[:3]:
        start = max(0, m.start() - 20)
        end   = min(len(full), m.end() + 80)
        print(f"    pos {m.start()}: ...{repr(full[start:end])}...")

# ── Search for item 7 ──
print("\n" + "="*60)
print("SEARCHING FOR ITEM 7 / MD&A:")
print("="*60)
for pattern in [r"item\s*7[^a8]", r"management.*discussion", r"Item 7\b"]:
    matches = list(re.finditer(pattern, full, re.IGNORECASE))
    print(f"  Pattern '{pattern}': {len(matches)} matches")
    for m in matches[:3]:
        start = max(0, m.start() - 20)
        end   = min(len(full), m.end() + 80)
        print(f"    pos {m.start()}: ...{repr(full[start:end])}...")