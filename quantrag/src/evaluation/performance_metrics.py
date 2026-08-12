"""
Phase 4 — Portfolio performance metrics.
Sharpe ratio, max drawdown, alpha, information ratio — computed
the same way for any equity curve (QuantRAG, momentum, or SPY).
"""

import numpy as np
import pandas as pd


def compute_returns(equity_curve: pd.Series) -> pd.Series:
    """Period-over-period returns from a cumulative equity curve."""
    return equity_curve.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 4, risk_free_rate: float = 0.02) -> float:
    """
    Annualised Sharpe ratio. periods_per_year=4 for quarterly rebalancing.
    risk_free_rate is annual (e.g. 0.02 = 2%).
    """
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction (-0.25 = -25%)."""
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return float(drawdown.min())


def annualised_return(equity_curve: pd.Series, periods_per_year: int = 4) -> float:
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_periods / periods_per_year
    return float(total_return ** (1 / years) - 1) if years > 0 else 0.0


def alpha_vs_benchmark(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Simple alpha: mean strategy return minus mean benchmark return, annualised."""
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if aligned.empty:
        return 0.0
    diff = aligned["strategy"] - aligned["benchmark"]
    return float(diff.mean() * 4)  # annualise quarterly diff


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Consistency of outperformance: mean excess return / tracking error."""
    aligned = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if aligned.empty:
        return 0.0
    excess = aligned["strategy"] - aligned["benchmark"]
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(4))


def full_report(equity_curve: pd.Series, benchmark_curve: pd.Series = None, label: str = "Strategy") -> dict:
    """One-call summary of all metrics for a given equity curve."""
    returns = compute_returns(equity_curve)

    report = {
        "label": label,
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1),
        "annualised_return": annualised_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
    }

    if benchmark_curve is not None:
        bench_returns = compute_returns(benchmark_curve)
        report["alpha"] = alpha_vs_benchmark(returns, bench_returns)
        report["information_ratio"] = information_ratio(returns, bench_returns)

    return report