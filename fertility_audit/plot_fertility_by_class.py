"""
Generate publication-ready visualizations for the fertility audit results.

Reads: results/fertility_by_class.json
Outputs (all 300 DPI, in results/):
  1. fertility_lambda_bar.png          — Bar chart of λ per morpheme class
  2. fertility_word_distribution.png   — Pie chart of word counts by class
  3. fertility_morpheme_histogram.png  — Grouped histogram of morpheme-count distribution
  4. fertility_summary_table.png       — Rendered table figure for paper/slides
"""

import json
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Load results ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "fertility_by_class.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

classes_data = data["classes"]
global_lambda = data["global_fertility"]
total_words = data["total_words"]
num_sentences = data["num_sentences"]

# Ordered class keys (exclude "other" from main 4, show it separately)
MAIN_CLASSES = ["noun_root", "verb_inflection", "compound", "postposition"]
ALL_CLASSES = MAIN_CLASSES + ["other"]

# Colors — distinct, colorblind-friendly palette
COLORS = {
    "noun_root":       "#2563eb",   # blue
    "verb_inflection": "#16a34a",   # green
    "compound":        "#dc2626",   # red
    "postposition":    "#9333ea",   # purple
    "other":           "#6b7280",   # gray
}

DISPLAY = {k: classes_data[k]["display_name"] for k in ALL_CLASSES}

# ── Figure 1: Bar chart of λ per class ───────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(8, 5))

labels = [DISPLAY[c] for c in ALL_CLASSES]
lambdas = [classes_data[c]["fertility_lambda"] for c in ALL_CLASSES]
stds = [classes_data[c]["std"] for c in ALL_CLASSES]
colors = [COLORS[c] for c in ALL_CLASSES]

bars = ax1.bar(labels, lambdas, color=colors, edgecolor="white", linewidth=0.8,
               yerr=stds, capsize=5, error_kw={"lw": 1.2, "color": "#374151"})

# Global λ reference line
ax1.axhline(y=global_lambda, color="#f59e0b", linewidth=2, linestyle="--",
            label=f"Global λ = {global_lambda:.4f}", zorder=2)

# Value labels on bars
for bar, lam, std in zip(bars, lambdas, stds):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.03,
             f"λ = {lam:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax1.set_ylabel("Fertility (λ) — avg morphemes per word", fontsize=12)
ax1.set_title(f"Fertility by Morpheme Class — Marathi\n"
              f"({num_sentences:,} sentences, {total_words:,} words, BhashaSetu corpus)",
              fontsize=13, fontweight="bold")
ax1.set_ylim(0, max(lambdas) + max(stds) + 0.4)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.xticks(rotation=15, ha="right", fontsize=10)
plt.tight_layout()

out1 = os.path.join(SCRIPT_DIR, "fertility_lambda_bar.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ── Figure 2: Pie chart of word count distribution ───────────────────────────

fig2, ax2 = plt.subplots(figsize=(7, 7))

word_counts = [classes_data[c]["num_words"] for c in ALL_CLASSES]
pie_labels = [f"{DISPLAY[c]}\n({classes_data[c]['num_words']:,})" for c in ALL_CLASSES]
pie_colors = [COLORS[c] for c in ALL_CLASSES]

wedges, texts, autotexts = ax2.pie(
    word_counts, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
    startangle=140, pctdistance=0.78,
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
    textprops=dict(fontsize=10),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")

ax2.set_title(f"Word Distribution by Morpheme Class\n"
              f"({total_words:,} words total)",
              fontsize=13, fontweight="bold")
plt.tight_layout()

out2 = os.path.join(SCRIPT_DIR, "fertility_word_distribution.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig2)

# ── Figure 3: Morpheme count histogram per class ────────────────────────────
# Read detail JSONL to build per-class morpheme count distributions

detail_path = os.path.join(SCRIPT_DIR, "fertility_by_class_detail.jsonl")
class_morph_counts = {c: [] for c in ALL_CLASSES}

print("Reading detail JSONL for histogram...")
with open(detail_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        cls = row["class"]
        if cls in class_morph_counts:
            class_morph_counts[cls].append(row["n_morphemes"])

fig3, ax3 = plt.subplots(figsize=(9, 5))

max_morphemes = 6  # cap display at 6 for readability
bins = list(range(1, max_morphemes + 2))  # [1, 2, 3, 4, 5, 6, 7]
bin_centers = np.array(range(1, max_morphemes + 1))
bar_width = 0.15

for i, cls in enumerate(ALL_CLASSES):
    counts = class_morph_counts[cls]
    # Build histogram: clamp values > max_morphemes into last bin
    hist = [0] * max_morphemes
    for c in counts:
        idx = min(c, max_morphemes) - 1
        hist[idx] += 1
    # Normalize to percentages
    total = sum(hist)
    pcts = [h / total * 100 if total > 0 else 0 for h in hist]

    offset = (i - len(ALL_CLASSES) / 2 + 0.5) * bar_width
    ax3.bar(bin_centers + offset, pcts, width=bar_width, color=COLORS[cls],
            label=DISPLAY[cls], edgecolor="white", linewidth=0.5)

ax3.set_xlabel("Morphemes per word", fontsize=12)
ax3.set_ylabel("Percentage of words (%)", fontsize=12)
ax3.set_title(f"Morpheme Count Distribution by Class\n"
              f"({total_words:,} words)", fontsize=13, fontweight="bold")
ax3.set_xticks(bin_centers)
ax3.set_xticklabels([str(b) if b < max_morphemes else f"{b}+" for b in bin_centers])
ax3.legend(fontsize=9, loc="upper right")
ax3.grid(axis="y", alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
plt.tight_layout()

out3 = os.path.join(SCRIPT_DIR, "fertility_morpheme_histogram.png")
fig3.savefig(out3, dpi=300, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

# ── Figure 4: Rendered summary table ────────────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(10, 4))
ax4.axis("off")

col_labels = ["Morpheme Class", "Words", "Total\nMorphemes", "λ (avg)", "σ", "Min", "Max"]
table_data = []
row_colors = []

for cls in ALL_CLASSES:
    d = classes_data[cls]
    table_data.append([
        d["display_name"],
        f"{d['num_words']:,}",
        f"{d['total_morphemes']:,}",
        f"{d['fertility_lambda']:.4f}",
        f"{d['std']:.4f}",
        str(d["min_morphemes"]),
        str(d["max_morphemes"]),
    ])
    row_colors.append(COLORS[cls] + "18")  # very light tint

# Global row
table_data.append([
    "GLOBAL",
    f"{total_words:,}",
    f"{data['total_morphemes']:,}",
    f"{global_lambda:.4f}",
    "—",
    "—",
    "—",
])
row_colors.append("#f59e0b22")

table = ax4.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#1e293b")
    cell.set_text_props(color="white", fontweight="bold")

# Style rows
for i in range(len(table_data)):
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(row_colors[i])
        if i == len(table_data) - 1:  # global row
            cell.set_text_props(fontweight="bold")
        # Bold the λ column
        if j == 3:
            cell.set_text_props(fontweight="bold")

ax4.set_title(f"Fertility (λ) by Morpheme Class — Marathi (BhashaSetu)\n"
              f"{num_sentences:,} sentences | {total_words:,} words",
              fontsize=13, fontweight="bold", pad=20)
plt.tight_layout()

out4 = os.path.join(SCRIPT_DIR, "fertility_summary_table.png")
fig4.savefig(out4, dpi=300, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close(fig4)

print("\nAll fertility visualizations generated successfully.")
