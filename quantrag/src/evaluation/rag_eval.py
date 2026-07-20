"""
Phase 2 — FinanceBench evaluation using RAGAS.

Runs our full RAG pipeline on 150 FinanceBench questions
and scores: faithfulness, answer_relevancy,
            context_precision, context_recall

Target scores:
    faithfulness        >= 0.80
    context_precision   >= 0.70
    context_recall      >= 0.65
    answer_relevancy    >= 0.75

Baseline (GPT-4 no RAG, from FinanceBench paper):
    faithfulness = 0.19

Usage:
    python scripts/run_eval.py
"""
# Patch missing VertexAI module — ragas imports it but we never use it
import sys
import types
_dummy = types.ModuleType('langchain_community.chat_models.vertexai')
_dummy.ChatVertexAI = None
sys.modules['langchain_community.chat_models.vertexai'] = _dummy

import os
import json
from typing import List, Dict
from datasets import load_dataset
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.retrieval.retriever import retrieve
from src.llm import get_llm_response
from src.retrieval.embedder import load_embedder, get_qdrant_client

load_dotenv()
console = Console()


def load_financebench(num_samples: int = 150) -> List[Dict]:
    """
    Load the open-source FinanceBench questions.

    Each item has:
        question:     the financial question
        answer:       human gold answer
        evidence:     list of supporting passage strings
        company:      ticker / company name
        doc_type:     "10-K", "10-Q", etc.

    Args:
        num_samples: how many questions to evaluate (max 150 open-source)
    """
    console.print("[blue]Loading FinanceBench dataset...[/blue]")

    ds = load_dataset(
        "PatronusAI/financebench",
        split="train",
        trust_remote_code=True
    )

    console.print(f"[green]  ✓ Loaded {len(ds)} questions[/green]")
    console.print(f"[dim]  Columns: {ds.column_names}[/dim]")

    # Take first num_samples
    samples = []
    for item in ds.select(range(min(num_samples, len(ds)))):
        samples.append({
            "question":    item.get("question", ""),
            "answer":      item.get("answer", ""),
            "evidence":    item.get("evidence", []),
            "company":     item.get("company_name", ""),
            "ticker":      item.get("ticker", ""),
            "doc_type":    item.get("doc_type", "10-K"),
            "fiscal_year": item.get("fiscal_year", ""),
        })

    return samples


def build_rag_answer(
    question: str,
    ticker: str,
    model,
    client,
) -> tuple:
    """
    Run our full two-stage RAG pipeline on one question.

    Returns:
        (answer_text, list_of_context_strings)
    """

    # Retrieve top-5 chunks using two-stage retrieval
    results = retrieve(
        query=question,
        model=model,
        client=client,
        ticker=ticker if ticker else None,
        top_k=5,
        use_reranker=True,
    )

    if not results:
        return "No relevant passages found in the index.", []

    # Build context string for the LLM
    context = "\n\n---\n\n".join([
        f"SOURCE: {r['citation']}\n{r['text']}"
        for r in results
    ])

    # Build prompt — same structure as Phase 1
    prompt = f"""You are a financial analyst assistant.
Answer the following question based ONLY on the provided SEC filing excerpts.
Always cite the source of every claim using the SOURCE labels.
If the answer is not in the excerpts, say "Not found in provided documents."

QUESTION: {question}

FILING EXCERPTS:
{context}

ANSWER:"""

    answer = get_llm_response(prompt, max_tokens=512)
    contexts = [r["text"] for r in results]

    return answer, contexts


def run_evaluation(
    num_samples: int = 150,
    save_results: bool = True,
) -> Dict:
    """
    Run the full FinanceBench RAGAS evaluation.

    Steps:
    1. Load 150 FinanceBench questions
    2. Run our RAG pipeline on each question
    3. Score with RAGAS metrics
    4. Save results to logs/ and print summary table

    Args:
        num_samples:  How many questions to evaluate
        save_results: Whether to save detailed results to JSON

    Returns:
        Dict of metric name → score
    """

    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset

    # ── Load resources ─────────────────────────────────────────────
    console.print("\n[bold blue]Phase 2 — FinanceBench Evaluation[/bold blue]")

    model  = load_embedder()
    client = get_qdrant_client()
    items  = load_financebench(num_samples)

    console.print(
        f"\n[blue]Running RAG pipeline on {len(items)} questions...[/blue]"
    )

    # ── Run pipeline on every question ────────────────────────────
    eval_data = {
        "question":    [],
        "answer":      [],
        "contexts":    [],
        "ground_truth": [],
    }

    for i, item in enumerate(items):
        console.print(
            f"[dim]  [{i+1}/{len(items)}] {item['company']} — "
            f"{item['question'][:60]}...[/dim]"
        )

        try:
            answer, contexts = build_rag_answer(
                question=item["question"],
                ticker=item.get("ticker", ""),
                model=model,
                client=client,
            )

            eval_data["question"].append(item["question"])
            eval_data["answer"].append(answer)
            eval_data["contexts"].append(contexts)
            eval_data["ground_truth"].append(item["answer"])

        except Exception as e:
            console.print(f"[red]  ✗ Error on question {i+1}: {e}[/red]")
            # Add placeholder so RAGAS dataset stays aligned
            eval_data["question"].append(item["question"])
            eval_data["answer"].append("Error")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(item["answer"])

    # ── Run RAGAS ─────────────────────────────────────────────────
    console.print("\n[blue]Running RAGAS scoring...[/blue]")

    dataset = Dataset.from_dict(eval_data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness":       round(float(result["faithfulness"]),       4),
        "answer_relevancy":   round(float(result["answer_relevancy"]),   4),
        "context_precision":  round(float(result["context_precision"]),  4),
        "context_recall":     round(float(result["context_recall"]),     4),
    }

    # ── Print results table ───────────────────────────────────────
    table = Table(
        show_header=True,
        header_style="bold",
        title="FinanceBench RAGAS Results"
    )
    table.add_column("Metric",    width=22)
    table.add_column("Score",     width=8)
    table.add_column("Target",    width=8)
    table.add_column("Baseline",  width=10)
    table.add_column("Status",    width=8)

    targets   = [0.80, 0.75, 0.70, 0.65]
    baselines = [0.19, "—",  "—",  "—" ]

    for (metric, score), target, baseline in zip(
        scores.items(), targets, baselines
    ):
        status = "✅" if score >= target else "⚠ below"
        table.add_row(
            metric,
            f"{score:.4f}",
            f"{target:.2f}",
            str(baseline),
            status,
        )

    console.print(table)

    # ── Save results ──────────────────────────────────────────────
    if save_results:
        os.makedirs("logs", exist_ok=True)
        with open("logs/ragas_results.json", "w") as f:
            json.dump(scores, f, indent=2)
        console.print("\n[dim]Results saved to logs/ragas_results.json[/dim]")

    console.print(Panel(
        f"[bold]Faithfulness:      {scores['faithfulness']:.4f}[/bold]  "
        f"(target ≥ 0.80, baseline 0.19)\n"
        f"Answer Relevancy:  {scores['answer_relevancy']:.4f}  (target ≥ 0.75)\n"
        f"Context Precision: {scores['context_precision']:.4f}  (target ≥ 0.70)\n"
        f"Context Recall:    {scores['context_recall']:.4f}  (target ≥ 0.65)",
        title="[green]RAGAS Evaluation Complete[/green]"
    ))

    return scores