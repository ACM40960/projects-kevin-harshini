"""
Phase 4 — Equity curve plot for the paper.

Reads the three saved equity curve CSVs and produces a single,
publication - quality figure showing QuantRAG, Momentum, and SPY
all starting at 100 and diverging over the 16 quarters.

Usage:
    python plot_equity_curves.py

Output:
    outputs/phase4_backtest/equity_curves.png 
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = "outputs/phase4_backtest"

#Load the three saved curves
quantrag = pd.read_csv(f"{DATA_DIR}/quantrag_equity_curve_latest.csv", index_col=0, parse_dates=True)
momentum = pd.read_csv(f"{DATA_DIR}/momentum_equity_curve_latest.csv", index_col=0, parse_dates=True)
spy      = pd.read_csv(f"{DATA_DIR}/spy_equity_curve_latest.csv", index_col=0, parse_dates=True)

#Build the figure
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

ax.plot(quantrag.index, quantrag["value"], label="QuantRAG",
        color="#2a78d6", linewidth=2.2, zorder=3)
ax.plot(momentum.index, momentum["value"], label="Momentum Baseline",
        color="#eb6834", linewidth=2.2, linestyle="--", zorder=3)
ax.plot(spy.index, spy["value"], label="S&P 500 (SPY)",
        color="#898781", linewidth=1.8, linestyle=":", zorder=2)

# End-of-line value labels
label_offsets = {"QuantRAG": -8, "Momentum": 8, "SPY": 0}

for series, color, name in [
    (quantrag, "#2a78d6", "QuantRAG"),
    (momentum, "#eb6834", "Momentum"),
    (spy,      "#52514e", "SPY"),
]:
    last_date = series.index[-1]
    last_val  = series["value"].iloc[-1]
    ax.annotate(
        f"{name}: {last_val:.0f}",
        xy=(last_date, last_val),
        xytext=(8, label_offsets[name]), textcoords="offset points",
        fontsize=9, color=color, va="center", fontweight="medium",
    )

#Styling
ax.set_title("QuantRAG vs. Momentum Baseline vs. S&P 500 (2020\u20132023)",
              fontsize=13, fontweight="medium", pad=14)
ax.set_ylabel("Portfolio Value (Base = 100)", fontsize=10.5)
ax.set_xlabel("")

ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.grid(False, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="both", labelsize=9.5, colors="#52514e")

ax.legend(
    loc="upper left", frameon=False, fontsize=9.5,
    handlelength=2.2, labelspacing=0.6,
)

plt.tight_layout()

#Save
out_path = f"{DATA_DIR}/equity_curves.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")

plt.show()