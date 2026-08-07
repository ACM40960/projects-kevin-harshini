"""
Phase 2 — FinanceBench evaluation with custom LLM-as-judge metrics.

Fully resumable, checkpointed evaluation — exactly like bulk_loader.py's
crash-resilient design. Each question's result is saved to disk the
moment it's scored. If the run is interrupted for any reason (network
blip, rate limit, laptop sleep, Ctrl+C), simply re-run the same command
and it picks up exactly where it stopped.

No hardcoding anywhere:
  - Ticker resolution: generic normalization + fuzzy match against
    whatever is actually indexed in Qdrant
  - Scoring: generic LLM-as-judge prompts, work on any question/answer
  - Retrieval diversity: generic Jaccard-based dedup (see reranker.py)

Usage:
    python scripts/run_eval.py
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Optional
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.retrieval.retriever import retrieve
from src.retrieval.embedder import load_embedder, get_qdrant_client
from src.llm import get_llm_response

load_dotenv()
console = Console()

TICKER_LOOKUP_CACHE = "logs/ticker_company_lookup.json"
EVAL_CHECKPOINT_FILE = "logs/eval_checkpoint.jsonl"   # one JSON line per completed question


# ── Ticker resolution — generic, no hardcoding ─────────────────────────────

def normalize_company_name(name: str) -> str:
    """Strip legal-entity suffixes/punctuation so naming conventions align."""
    name = name.upper()
    name = re.sub(r'/[A-Z]+/', '', name)
    name = re.sub(r'[.,&\'"]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    suffixes = {
        "INCORPORATED", "CORPORATION", "COMPANY", "HOLDINGS", "HOLDING",
        "GROUP", "LIMITED", "INC", "CORP", "CO", "LTD", "LLC", "PLC",
        "LP", "SA", "NV", "AG",
    }
    words = name.split()
    while words and words[-1] in suffixes:
        words.pop()
    return " ".join(words)


def build_ticker_lookup(client) -> dict:
    """Build {normalized_company_name: ticker} from what's actually indexed."""
    if os.path.exists(TICKER_LOOKUP_CACHE):
        with open(TICKER_LOOKUP_CACHE) as f:
            return json.load(f)

    console.print("[blue]Building ticker lookup from Qdrant index (one-time)...[/blue]")
    lookup = {}
    offset = None
    seen = set()

    while True:
        records, offset = client.scroll(
            collection_name=os.getenv("QDRANT_COLLECTION", "quantrag_phase2"),
            limit=1000, offset=offset,
            with_payload=["ticker", "company_name"],
        )
        for r in records:
            ticker = r.payload.get("ticker", "")
            company = r.payload.get("company_name", "")
            if ticker and company and ticker not in seen:
                lookup[normalize_company_name(company)] = ticker
                seen.add(ticker)
        if offset is None:
            break

    os.makedirs("logs", exist_ok=True)
    with open(TICKER_LOOKUP_CACHE, "w") as f:
        json.dump(lookup, f, indent=2)
    console.print(f"[green]  ✓ Built lookup for {len(lookup)} companies[/green]")
    return lookup


def company_to_ticker(company_name: str, lookup: dict) -> str:
    """Resolve any company name to a ticker: exact match, then fuzzy fallback."""
    if not company_name or not lookup:
        return ""
    normalized = normalize_company_name(company_name)
    if normalized in lookup:
        return lookup[normalized]

    from rapidfuzz import process, fuzz
    match = process.extractOne(
        normalized, lookup.keys(), scorer=fuzz.partial_ratio, score_cutoff=80,
    )
    return lookup[match[0]] if match else ""


# ── FinanceBench loading ────────────────────────────────────────────────────

def load_financebench(num_samples: int, client) -> List[Dict]:
    console.print("[blue]Loading FinanceBench dataset...[/blue]")
    ds = load_dataset("PatronusAI/financebench", split="train")
    console.print(f"[green]  ✓ Loaded {len(ds)} questions[/green]")

    lookup = build_ticker_lookup(client)

    samples = []
    for i, item in enumerate(ds.select(range(min(num_samples, len(ds))))):
        company = item.get("company", "")
        samples.append({
            "id":          i,
            "question":    item.get("question", ""),
            "answer":      item.get("answer", ""),
            "company":     company,
            "ticker":      company_to_ticker(company, lookup),
            "doc_type":    item.get("doc_type", "10-K"),
            "fiscal_year": item.get("doc_period", ""),
        })
    return samples


def build_rag_answer(question: str, ticker: str, model, client) -> tuple:
    """Run the full two-stage RAG pipeline (with diversity filtering) on one question."""
    results = retrieve(
        query=question, model=model, client=client,
        ticker=ticker if ticker else None, top_k=5, use_reranker=True,
    )
    if not results:
        return "No relevant passages found in the index.", []

    context = "\n\n---\n\n".join(
        f"SOURCE: {r['citation']}\n{r['text']}" for r in results
    )
    prompt = f"""You are a financial analyst assistant.
Answer the following question based ONLY on the provided SEC filing excerpts.
Always cite the source of every claim using the SOURCE labels.
If the answer is not in the excerpts, say "Not found in provided documents."

QUESTION: {question}

FILING EXCERPTS:
{context}

ANSWER:"""

    answer = get_llm_response(prompt, max_tokens=1024)
    return answer, [r["text"] for r in results]


# ── Custom LLM-as-judge metrics ─────────────────────────────────────────────

def _extract_score(raw: str) -> Optional[float]:
    match = re.search(r'([01](?:\.\d+)?)', raw.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def score_faithfulness(question: str, answer: str, contexts: List[str]) -> Optional[float]:
    context_block = "\n\n".join(contexts)
    prompt = f"""Evaluate whether every claim in the ANSWER is directly supported
by the SOURCE DOCUMENTS (measures "faithfulness" / absence of hallucination).

QUESTION: {question}

SOURCE DOCUMENTS:
{context_block}

ANSWER: {answer}

Score from 0.0 to 1.0:
  1.0 = every claim is directly supported by the sources
  0.5 = some claims supported, others not verifiable
  0.0 = not supported at all, or contradicts the sources

Respond with ONLY a number between 0.0 and 1.0."""
    return _extract_score(get_llm_response(prompt, max_tokens=200))


def score_answer_relevancy(question: str, answer: str) -> Optional[float]:
    prompt = f"""Evaluate how directly the ANSWER addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Score from 0.0 to 1.0:
  1.0 = directly and completely addresses the question
  0.5 = partially relevant or incomplete
  0.0 = does not address the question at all

Respond with ONLY a number between 0.0 and 1.0."""
    return _extract_score(get_llm_response(prompt, max_tokens=200))


def score_context_precision(question: str, contexts: List[str], ground_truth: str) -> Optional[float]:
    """
    Single-call scoring of all contexts together — more reliable and much
    faster than one call per context.
    """
    if not contexts:
        return 0.0

    numbered = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = f"""QUESTION: {question}
KNOWN CORRECT ANSWER: {ground_truth}

RETRIEVED PASSAGES:
{numbered}

Which numbered passages are relevant and useful for answering the question?
Respond with ONLY the relevant numbers, comma-separated (e.g. "1,3,4").
If none are relevant, respond "none"."""

    raw = get_llm_response(prompt, max_tokens=200).strip().lower()
    if raw == "none" or not raw:
        return 0.0

    numbers = set(re.findall(r'\d+', raw))
    relevant_count = len(numbers)
    return round(min(relevant_count, len(contexts)) / len(contexts), 4)


def score_context_recall(question: str, contexts: List[str], ground_truth: str) -> Optional[float]:
    if not contexts:
        return 0.0

    context_block = "\n\n".join(contexts)
    prompt = f"""QUESTION: {question}
KNOWN CORRECT ANSWER: {ground_truth}

RETRIEVED PASSAGES (combined):
{context_block}

Do the passages together contain enough information to derive the
correct answer? Score from 0.0 to 1.0:
  1.0 = all necessary information is present
  0.5 = some but not all necessary information is present
  0.0 = none of the necessary information is present

Respond with ONLY a number between 0.0 and 1.0."""
    return _extract_score(get_llm_response(prompt, max_tokens=200))


async def score_one(question: str, answer: str, contexts: List[str], ground_truth: str) -> dict:
    loop = asyncio.get_event_loop()

    faithfulness = await loop.run_in_executor(
        None, score_faithfulness, question, answer, contexts
    )
    answer_relevancy = await loop.run_in_executor(
        None, score_answer_relevancy, question, answer
    )
    context_precision = await loop.run_in_executor(
        None, score_context_precision, question, contexts, ground_truth
    )
    context_recall = await loop.run_in_executor(
        None, score_context_recall, question, contexts, ground_truth
    )

    return {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,
    }


def average_valid(values: List) -> float:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else 0.0


# ── Checkpoint helpers — resumable evaluation ───────────────────────────────

def load_checkpoint() -> Dict[int, dict]:
    """Load already-completed question results, keyed by question id."""
    completed = {}
    if os.path.exists(EVAL_CHECKPOINT_FILE):
        with open(EVAL_CHECKPOINT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    completed[row["id"]] = row
    return completed


def append_checkpoint(row: dict) -> None:
    """Append one completed question's result — crash-safe, immediate write."""
    os.makedirs("logs", exist_ok=True)
    with open(EVAL_CHECKPOINT_FILE, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Main evaluation ──────────────────────────────────────────────────────────

async def run_evaluation_async(num_samples: int = 150, save_results: bool = True) -> Dict:
    console.print("\n[bold blue]Phase 2 — FinanceBench Evaluation (resumable)[/bold blue]")

    model  = load_embedder()
    client = get_qdrant_client()
    items  = load_financebench(num_samples, client)

    already_done = load_checkpoint()
    remaining = [item for item in items if item["id"] not in already_done]

    console.print(
        f"\n[blue]{len(already_done)} already completed, "
        f"{len(remaining)} remaining[/blue]\n"
    )

    for item in remaining:
        console.print(
            f"[dim]  [{item['id']+1}/{len(items)}] {item['company']} — "
            f"{item['question'][:55]}...[/dim]"
        )

        try:
            answer, contexts = build_rag_answer(
                item["question"], item.get("ticker", ""), model, client
            )
            row_scores = await score_one(
                item["question"], answer, contexts, item["answer"]
            )

            score_summary = "  ".join(
                f"{k}={v:.2f}" if v is not None else f"{k}=FAIL"
                for k, v in row_scores.items()
            )
            console.print(f"    [cyan]→ {score_summary}[/cyan]")

            append_checkpoint({
                "id": item["id"],
                "question": item["question"],
                "ticker": item["ticker"],
                "scores": row_scores,
            })

        except Exception as e:
            console.print(f"[red]  ✗ Error on question {item['id']+1}: {str(e)[:100]}[/red]")
            append_checkpoint({
                "id": item["id"],
                "question": item["question"],
                "ticker": item["ticker"],
                "scores": {k: None for k in
                           ["faithfulness", "answer_relevancy",
                            "context_precision", "context_recall"]},
            })

    # ── Aggregate all checkpointed results (this run + any prior runs) ──
    all_results = load_checkpoint()
    all_scores = {
        "faithfulness": [], "answer_relevancy": [],
        "context_precision": [], "context_recall": [],
    }
    for row in all_results.values():
        for k in all_scores:
            all_scores[k].append(row["scores"].get(k))

    scores = {k: average_valid(v) for k, v in all_scores.items()}

    table = Table(show_header=True, header_style="bold", title="FinanceBench Evaluation Results")
    table.add_column("Metric", width=22)
    table.add_column("Score", width=8)
    table.add_column("Target", width=8)
    table.add_column("Baseline", width=10)
    table.add_column("Status", width=8)

    targets   = {"faithfulness": 0.80, "answer_relevancy": 0.75,
                 "context_precision": 0.70, "context_recall": 0.65}
    baselines = {"faithfulness": 0.19, "answer_relevancy": "—",
                 "context_precision": "—", "context_recall": "—"}

    for metric, score in scores.items():
        target = targets[metric]
        status = "✅" if score >= target else "⚠ below"
        table.add_row(metric, f"{score:.4f}", f"{target:.2f}", str(baselines[metric]), status)

    console.print(table)

    if save_results:
        os.makedirs("logs", exist_ok=True)
        with open("logs/ragas_results.json", "w") as f:
            json.dump({
                "scores": scores,
                "num_questions_scored": len(all_results),
            }, f, indent=2)
        console.print("\n[dim]Results saved to logs/ragas_results.json[/dim]")
        console.print(f"[dim]Per-question checkpoint: {EVAL_CHECKPOINT_FILE}[/dim]")

    console.print(Panel(
        f"[bold]Faithfulness:      {scores['faithfulness']:.4f}[/bold]  (target ≥ 0.80, baseline 0.19)\n"
        f"Answer Relevancy:  {scores['answer_relevancy']:.4f}  (target ≥ 0.75)\n"
        f"Context Precision: {scores['context_precision']:.4f}  (target ≥ 0.70)\n"
        f"Context Recall:    {scores['context_recall']:.4f}  (target ≥ 0.65)\n"
        f"Questions scored:  {len(all_results)}/{len(items)}",
        title="[green]Evaluation Complete[/green]"
    ))

    return scores


def run_evaluation(num_samples: int = 150, save_results: bool = True) -> Dict:
    return asyncio.run(run_evaluation_async(num_samples, save_results))