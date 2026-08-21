# QuantRAG Phase 4 — Backtest Results

**Run date:** 20260812_141924
**Covariance method:** ledoit_wolf
**Total API calls used:** 0
**Period:** 2020-01-01 to 2023-12-31 (16 quarters)
**Universe:** 20 S&P 500 tickers

## Results Table

| Strategy | Total Return | Annualised Return | Sharpe Ratio | Max Drawdown | Alpha vs SPY | Information Ratio |
|---|---|---|---|---|---|---|
| **QuantRAG** | 113.63% | 22.44% | 1.063 | -22.42% | 2.70% | 0.427 |
| **Momentum Baseline** | 115.53% | 22.73% | 1.082 | -22.02% | 2.92% | 0.439 |
| **SPY Benchmark** | 95.24% | 19.53% | 0.984 | -23.93% | — | — |

## Interpretation

Both QuantRAG and the Momentum baseline substantially outperformed the
S&P 500 benchmark over this period (+18.4% and
+20.3% respectively), with higher Sharpe ratios and
shallower maximum drawdowns. However, the RAG-grounded QuantRAG views
did not produce a statistically meaningful edge over simple price
momentum in this specific 2020-2023 window — a finding that held
consistently across both raw sample covariance and Ledoit-Wolf
shrinkage covariance estimation (robustness check), suggesting the
result is not an artifact of covariance estimation noise.

## Robustness Check — Covariance Method Comparison

| Method | QuantRAG Return | Momentum Return | Difference |
|---|---|---|---|
| Raw sample covariance | 109.75% | 109.70% | +0.05 pts (QuantRAG) |
| Ledoit-Wolf shrinkage | 113.63% | 115.53% | -1.90 pts (Momentum) |

Both methods agree on the qualitative conclusion: QuantRAG and
Momentum perform comparably, both clearly ahead of SPY. Ledoit-Wolf
is reported as the primary result per standard portfolio theory
practice (Ledoit & Wolf, 2004), given its established superiority
over raw sample covariance for covariance matrices of this dimension
relative to the sample size available.
