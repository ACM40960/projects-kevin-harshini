"""
Phase 2 — Run FinanceBench RAGAS evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.rag_eval import run_evaluation

if __name__ == "__main__":
    scores = run_evaluation(num_samples=150, save_results=True)