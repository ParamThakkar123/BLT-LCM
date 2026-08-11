"""
Visualize error analysis results: Phase 2 (BLT) vs Phase 1 tokenizer failures.
Reads from failure_sentences_100.jsonl and category_summary.json.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lcm_scripts.device_utils import report_cpu_only

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["figure.dpi"] = 150

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

report_cpu_only("matplotlib rendering from existing result files")

with open(os.path.join(SCRIPT_DIR, "failure_sentences_100.jsonl"), "r", encoding="utf-8") as f:
    failures = [json.loads(line) for line in f if line.strip()]

with open(os.path.join(SCRIPT_DIR, "category_summary.json"), "r", encoding="utf-8") as f:
    summary = json.load(f)

# ── 1. Category distribution (horizontal bar) ────────────────

fig, ax = plt.subplots(figsize=(9, 4))
cats = summary["category_counts"]
labels = list(cats.keys())
counts = list(cats.values())
colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6", "#9b59b6", "#f39c12"]
bars = ax.barh(labels[::-1], counts[::-1], color=colors[:len(labels)][::-1], edgecolor="white", linewidth=0.5)
for bar, cnt in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", fontweight="bold", fontsize=11)
ax.set_xlabel("Count (out of 100 failure sentences)")
ax.set_title("Failure Category Distribution — Phase 2 BLT vs Phase 1 Tokenizers")
ax.set_xlim(0, max(counts) + 10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig1_category_distribution.png"))
plt.close()
print("Saved fig1_category_distribution.png")

# ── 2. Delta regression distribution (histogram) ─────────────

fig, ax = plt.subplots(figsize=(8, 4.5))
deltas = [f["delta_regression"] for f in failures]
ax.hist(deltas, bins=20, color="#3498db", edgecolor="white", linewidth=0.8, alpha=0.9)
ax.axvline(np.mean(deltas), color="#e74c3c", linestyle="--", linewidth=1.5,
           label=f"Mean = {np.mean(deltas):.3f}")
ax.axvline(np.median(deltas), color="#2ecc71", linestyle="--", linewidth=1.5,
           label=f"Median = {np.median(deltas):.3f}")
ax.set_xlabel("Fertility Delta (BLT − Phase1 Best)")
ax.set_ylabel("Number of Sentences")
ax.set_title("Distribution of Fertility Regression (Top 100 Failures)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig2_delta_distribution.png"))
plt.close()
print("Saved fig2_delta_distribution.png")

# ── 3. BLT fertility vs Phase1 best fertility (scatter) ──────

fig, ax = plt.subplots(figsize=(6, 6))
p1 = [f["phase1_best_fertility"] for f in failures]
blt = [f["blt_fertility"] for f in failures]
nw = [f["n_words"] for f in failures]

scatter = ax.scatter(p1, blt, c=nw, cmap="viridis", s=50, alpha=0.75, edgecolors="white", linewidth=0.4)
lims = [min(min(p1), min(blt)) - 0.1, max(max(p1), max(blt)) + 0.1]
ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, label="y = x (no regression)")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Phase 1 Best Fertility (tokens/word)")
ax.set_ylabel("Phase 2 BLT Fertility (patches/word)")
ax.set_title("BLT vs Phase 1 Fertility\n(points above diagonal = BLT regression)")
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label("Word Count")
ax.legend(loc="upper left", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig3_fertility_scatter.png"))
plt.close()
print("Saved fig3_fertility_scatter.png")

# ── 4. Delta by category (box plot) ──────────────────────────

cat_deltas = {}
for f in failures:
    for c in f["categories"]:
        cat_deltas.setdefault(c, []).append(f["delta_regression"])

sorted_cats = sorted(cat_deltas.keys(), key=lambda c: -np.median(cat_deltas[c]))
box_data = [cat_deltas[c] for c in sorted_cats]
short_labels = [c.replace("Domain-specific: ", "Domain: ").replace("Rare Unicode / uncommon Devanagari", "Rare Unicode / Devanagari").replace("Very short sentence (< 5 words)", "Very short (< 5 words)") for c in sorted_cats]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(box_data, vert=False, patch_artist=True, widths=0.5,
                boxprops=dict(linewidth=0.8),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker="o", markersize=4, alpha=0.5))
colors_box = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#95a5a6"]
for patch, color in zip(bp["boxes"], colors_box[:len(bp["boxes"])]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_yticklabels(short_labels)
ax.set_xlabel("Fertility Delta (BLT − Phase1 Best)")
ax.set_title("Regression Severity by Failure Category")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig4_delta_by_category.png"))
plt.close()
print("Saved fig4_delta_by_category.png")

# ── 5. Word count vs delta (scatter with category color) ─────

cat_color_map = {
    "Very short sentence (< 5 words)": "#e74c3c",
    "Rare Unicode / uncommon Devanagari": "#3498db",
    "Long compound words": "#2ecc71",
    "Domain-specific: medical": "#9b59b6",
    "Domain-specific: legal": "#f39c12",
    "Other": "#95a5a6",
}

fig, ax = plt.subplots(figsize=(9, 5))
for f in failures:
    primary_cat = f["categories"][0]
    color = cat_color_map.get(primary_cat, "#95a5a6")
    ax.scatter(f["n_words"], f["delta_regression"], c=color, s=45, alpha=0.7,
               edgecolors="white", linewidth=0.3)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=8, label=k.replace("Very short sentence (< 5 words)", "Very short (< 5 words)").replace("Rare Unicode / uncommon Devanagari", "Rare Unicode / Devanagari"))
                   for k, v in cat_color_map.items() if k in {f["categories"][0] for f in failures}]
ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9)
ax.set_xlabel("Word Count")
ax.set_ylabel("Fertility Delta (BLT − Phase1 Best)")
ax.set_title("Regression Delta vs Sentence Length (colored by primary category)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig5_wordcount_vs_delta.png"))
plt.close()
print("Saved fig5_wordcount_vs_delta.png")

# ── 6. Stacked summary: Phase1 aug vs ret vs BLT (grouped bar) ─

fig, ax = plt.subplots(figsize=(10, 4.5))
ranks = np.arange(1, len(failures) + 1)
width = 0.28
ax.bar(ranks - width, [f["fertility_augmented"] for f in failures], width, label="Phase1 Augmented", color="#3498db", alpha=0.8)
ax.bar(ranks, [f["fertility_retrained"] for f in failures], width, label="Phase1 Retrained", color="#2ecc71", alpha=0.8)
ax.bar(ranks + width, [f["blt_fertility"] for f in failures], width, label="Phase2 BLT", color="#e74c3c", alpha=0.8)
ax.set_xlabel("Failure Rank (sorted by regression severity)")
ax.set_ylabel("Fertility (tokens or patches / word)")
ax.set_title("Fertility Comparison Across All 100 Failure Sentences")
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(0, 101)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "fig6_fertility_comparison.png"))
plt.close()
print("Saved fig6_fertility_comparison.png")

print("\nAll 6 figures saved.")
