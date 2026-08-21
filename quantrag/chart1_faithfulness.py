import matplotlib.pyplot as plt

# ── Colors matching the poster palette ──
NAVY  = "#12345A"
TEAL  = "#0F6E56"
CORAL = "#D85A30"
GRAY  = "#5F5E5A"

labels = ["GPT-4\n(no RAG)", "QuantRAG\n(RAG-grounded)"]
values = [0.19, 0.6864]
colors = [CORAL, TEAL]

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)

bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)

# Value labels on top of each bar
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, val + 0.02,
        f"{val:.2f}", ha="center", va="bottom",
        fontsize=18, fontweight="bold", color=NAVY,
    )

ax.set_ylim(0, 0.8)
ax.set_ylabel("Faithfulness Score", fontsize=13)
ax.set_title("RAG Faithfulness vs. Ungrounded LLM Baseline", fontsize=15, fontweight="medium", pad=14)

ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
ax.grid(False, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="both", labelsize=12, colors=GRAY)

plt.tight_layout()
plt.savefig("chart1_faithfulness.png", dpi=300, bbox_inches="tight")
print("Saved: chart1_faithfulness.png")
plt.show()