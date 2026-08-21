import matplotlib.pyplot as plt

NAVY  = "#12345A"
TEAL  = "#0F6E56"
CORAL = "#D85A30"
GRAY  = "#5F5E5A"

labels = ["S&P 500\n(SPY)", "Momentum\nBaseline", "QuantRAG"]
values = [0.9524, 1.1553, 1.1363]
colors = [GRAY, CORAL, TEAL]

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, val + 0.02,
        f"{val:.0%}", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color=NAVY,
    )

ax.set_ylim(0, 1.3)
ax.set_ylabel("Total Return", fontsize=13)
ax.set_title("16-Quarter Backtest: Total Return (2020\u20132023)", fontsize=15, fontweight="medium", pad=14)

ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.grid(False, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="both", labelsize=12, colors=GRAY)

# Format y-axis as percentages
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

plt.tight_layout()
plt.savefig("chart2_backtest.png", dpi=300, bbox_inches="tight")
print("Saved: chart2_backtest.png")
plt.show()