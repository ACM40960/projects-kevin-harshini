"""
Phase 4 — Price-momentum baseline (the ablation).
Generates views using ONLY price history — no LLM, no filings, no RAG.
Everything else (Black-Litterman, optimizer, constraints) stays identical
to QuantRAG's real pipeline, isolating the effect of RAG-grounded views.
"""

import pandas as pd
from typing import List, Dict


def generate_momentum_views(
    tickers: List[str],
    price_history: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: int = 180,
) -> List[Dict]:
    """
    For each ticker, compute trailing N-day momentum and convert it
    directly into a view — same {ticker, magnitude, confidence} shape
    that Black-Litterman expects from the LLM pipeline, but derived
    purely from price arithmetic.

    Magnitude: trailing return, capped to a sane range
    Confidence: scaled by how strong/consistent the momentum is
                (stronger, more consistent moves = higher confidence,
                 mirroring how the LLM assigns higher confidence to
                 stronger evidence)
    """
    views = []
    window_start = as_of_date - pd.Timedelta(days=lookback_days)

    for ticker in tickers:
        if ticker not in price_history.columns:
            continue

        series = price_history[ticker]
        window = series[(series.index >= window_start) & (series.index <= as_of_date)]

        if len(window) < 20:  # not enough data to form a view
            continue

        momentum = (window.iloc[-1] / window.iloc[0]) - 1

        # Cap magnitude to the same reasonable range the LLM uses (-15% to +15%)
        magnitude = max(min(momentum, 0.15), -0.15)

        # Confidence scales with |momentum| — stronger moves = more confidence
        # (capped between 0.3 and 0.8, mirroring realistic LLM confidence range)
        confidence = max(min(0.3 + abs(momentum) * 2, 0.8), 0.3)

        direction = "bullish" if momentum > 0.02 else ("bearish" if momentum < -0.02 else "neutral")

        views.append({
            "ticker": ticker,
            "direction": direction,
            "magnitude": round(float(magnitude), 4),
            "confidence": round(float(confidence), 4),
            "reasoning": f"{lookback_days}-day price momentum: {momentum:+.1%}",
            "citation": f"Price history {window_start.date()} to {as_of_date.date()}",
        })

    return views