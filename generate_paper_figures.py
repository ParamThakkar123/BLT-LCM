"""
Generate all publication-ready figures for the research paper.
Reads existing result files only -- does NOT recompute anything.

Outputs (all 300 DPI, tight layout):
  - fig1_patches_per_sentence.png      : Bar chart of mean patches/sentence vs tau
  - fig2_compression_ratio.png         : Compression ratio & avg patch size vs tau
  - fig3_patch_distribution_boxplot.png : Box plot of patch count distributions
  - fig4_morpheme_alignment_f1.png     : P/R/F1 vs tau with best-tau annotation
  - fig5_marginal_effect.png           : Marginal % change between consecutive taus
  - fig6_summary_table.png             : Combined results table as image
"""

import sys
import io
import json
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────

with open("sweep_summary.json", "r", encoding="utf-8") as f:
    sweep = json.load(f)

with open("morpheme_alignment_results.json", "r", encoding="utf-8") as f:
    morph = json.load(f)

with open("sweep_results.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

TAUS = [0.5, 1.0, 1.5, 2.0, 2.5]
TAU_LABELS = ["0.5", "1.0", "1.5", "2.0", "2.5"]
N = sweep["num_sentences"]
total_bytes = sum(r["byte_length"] for r in rows)

# Shared style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6"]

# ── Figure 1: Patches per Sentence (bar + line) ─────────────────────────────

fig, ax1 = plt.subplots(figsize=(7, 4.5))

means = [sweep["thresholds"][t]["patches_per_sentence"]["mean"] for t in TAU_LABELS]
stds = [sweep["thresholds"][t]["patches_per_sentence"]["std"] for t in TAU_LABELS]
x = np.arange(len(TAUS))

bars = ax1.bar(x, means, width=0.5, color=COLORS, edgecolor="white", linewidth=1.2,
               yerr=stds, capsize=5, error_kw={"lw": 1.2, "capthick": 1.2})

# Value labels on bars
for i, (m, s) in enumerate(zip(means, stds)):
    ax1.text(i, m + s + 12, f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax1.set_xlabel(r"Entropy Threshold ($\tau$)")
ax1.set_ylabel("Mean Patches per Sentence")
ax1.set_title(r"Effect of Entropy Threshold $\tau$ on Patch Count" + f"\n({N:,} Marathi sentences, BhashaSetu)")
ax1.set_xticks(x)
ax1.set_xticklabels([str(t) for t in TAUS])
ax1.set_ylim(0, max(means) * 1.25)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("fig1_patches_per_sentence.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig1_patches_per_sentence.png")

# ── Figure 2: Compression Ratio & Avg Patch Size ────────────────────────────

fig, ax1 = plt.subplots(figsize=(7, 4.5))

comp_ratios = [total_bytes / sweep["thresholds"][t]["total_patches"] for t in TAU_LABELS]
avg_sizes = [sweep["thresholds"][t]["avg_patch_size"]["mean"] for t in TAU_LABELS]

color1 = "#2563eb"
color2 = "#dc2626"

ax1.plot(TAUS, comp_ratios, "o-", color=color1, linewidth=2.5, markersize=9, label="Compression Ratio", zorder=3)
ax1.set_xlabel(r"Entropy Threshold ($\tau$)")
ax1.set_ylabel("Compression Ratio (bytes / patches)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_ylim(0, max(comp_ratios) * 1.3)

ax2 = ax1.twinx()
ax2.plot(TAUS, avg_sizes, "s--", color=color2, linewidth=2, markersize=7, label="Avg Patch Size (bytes)", zorder=2)
ax2.set_ylabel("Avg Patch Size (bytes)", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax2.set_ylim(0, max(avg_sizes) * 1.5)

# Add value labels
for i, (cr, aps) in enumerate(zip(comp_ratios, avg_sizes)):
    ax1.annotate(f"{cr:.2f}x", (TAUS[i], cr), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=9, color=color1, fontweight="bold")
    ax2.annotate(f"{aps:.2f}B", (TAUS[i], aps), textcoords="offset points",
                 xytext=(0, -16), ha="center", fontsize=9, color=color2, fontweight="bold")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

ax1.set_title(r"Compression Ratio and Patch Granularity vs. $\tau$" + f"\n({N:,} Marathi sentences)")
ax1.set_xticks(TAUS)
ax1.spines["top"].set_visible(False)
ax1.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("fig2_compression_ratio.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig2_compression_ratio.png")

# ── Figure 3: Patch Count Distribution Box Plot ─────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))

box_data = []
for t in TAU_LABELS:
    key = f"tau_{t}_num_patches"
    box_data.append([r[key] for r in rows])

bp = ax.boxplot(box_data, positions=range(len(TAUS)), widths=0.45, patch_artist=True,
                showfliers=False, showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
                medianprops=dict(color="black", linewidth=1.5))

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel(r"Entropy Threshold ($\tau$)")
ax.set_ylabel("Patches per Sentence")
ax.set_title(r"Distribution of Patch Counts Across $\tau$" + f"\n({N:,} Marathi sentences, outliers hidden)")
ax.set_xticks(range(len(TAUS)))
ax.set_xticklabels([str(t) for t in TAUS])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Add median/mean annotations
for i, data in enumerate(box_data):
    med = sorted(data)[len(data) // 2]
    mn = sum(data) / len(data)
    ax.text(i + 0.28, med, f"med={med:.0f}", va="center", fontsize=8, color="black")
    ax.text(i + 0.28, mn, f"mean={mn:.0f}", va="center", fontsize=8, color="gray")

plt.tight_layout()
plt.savefig("fig3_patch_distribution_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig3_patch_distribution_boxplot.png")

# ── Figure 4: Morpheme Alignment P/R/F1 ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))

f1_vals = [morph["thresholds"][t]["f1"] for t in TAU_LABELS]
prec_vals = [morph["thresholds"][t]["precision"] for t in TAU_LABELS]
rec_vals = [morph["thresholds"][t]["recall"] for t in TAU_LABELS]

ax.plot(TAUS, f1_vals, "o-", color="#2563eb", linewidth=2.5, markersize=9, label="F1", zorder=3)
ax.plot(TAUS, prec_vals, "s--", color="#16a34a", linewidth=1.8, markersize=7, label="Precision", alpha=0.85)
ax.plot(TAUS, rec_vals, "^--", color="#dc2626", linewidth=1.8, markersize=7, label="Recall", alpha=0.85)

# Shade the best-tau region
best_tau = morph["best_tau"]
best_f1 = morph["best_f1"]
ax.axvline(x=best_tau, color="#f59e0b", linestyle=":", linewidth=1.5, alpha=0.7)
ax.axhspan(best_f1 - 0.005, best_f1 + 0.005, alpha=0.08, color="#2563eb")

ax.annotate(
    f"Best: $\\tau$={best_tau}\nF1={best_f1:.4f}",
    xy=(best_tau, best_f1),
    xytext=(best_tau + 0.45, best_f1 - 0.06),
    fontsize=11, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.8),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.95),
)

ax.set_xlabel(r"Entropy Threshold ($\tau$)")
ax.set_ylabel("Score")
ax.set_title(r"Patch Boundary vs. Morpheme Boundary Alignment" + f"\n({morph['num_sentences']:,} Marathi sentences, tolerance=\u00b1{morph['tolerance_bytes']}B)")
ax.set_xticks(TAUS)
ax.set_ylim(0, 1.08)
ax.legend(loc="right", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("fig4_morpheme_alignment_f1.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig4_morpheme_alignment_f1.png")

# ── Figure 5: Marginal Effect (% change between consecutive taus) ───────────

fig, ax = plt.subplots(figsize=(7, 4))

transitions = []
pct_changes = []
for i in range(len(TAUS) - 1):
    t1, t2 = TAU_LABELS[i], TAU_LABELS[i + 1]
    p1 = sweep["thresholds"][t1]["total_patches"]
    p2 = sweep["thresholds"][t2]["total_patches"]
    pct = ((p2 - p1) / p1) * 100
    transitions.append(f"{t1} -> {t2}")
    pct_changes.append(pct)

bar_colors = ["#ef4444" if p < -1 else "#f59e0b" if p < -0.01 else "#22c55e" for p in pct_changes]
bars = ax.barh(range(len(transitions)), pct_changes, color=bar_colors, edgecolor="white", height=0.5)

for i, (bar, pct) in enumerate(zip(bars, pct_changes)):
    xpos = pct - 2 if pct < -5 else pct + 0.3
    ax.text(xpos, i, f"{pct:+.1f}%", va="center", fontsize=11, fontweight="bold",
            color="white" if pct < -5 else "black")

ax.set_yticks(range(len(transitions)))
ax.set_yticklabels(transitions, fontsize=11)
ax.set_xlabel("% Change in Total Patches")
ax.set_title(r"Marginal Effect of Increasing $\tau$" + f"\n({N:,} Marathi sentences)")
ax.axvline(x=0, color="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("fig5_marginal_effect.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig5_marginal_effect.png")

# ── Figure 6: Combined Summary Table as Image ───────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis("off")

# Build table data
col_labels = [r"$\tau$", "Total\nPatches", "Mean\n/Sent", "Median\n/Sent", "Std", "Avg Patch\nSize (B)",
              "Compress.\nRatio", "Morph\nPrecision", "Morph\nRecall", "Morph\nF1"]

table_data = []
for i, t in enumerate(TAU_LABELS):
    s = sweep["thresholds"][t]
    ps = s["patches_per_sentence"]
    aps = s["avg_patch_size"]["mean"]
    cr = total_bytes / s["total_patches"]
    m = morph["thresholds"][t]
    row = [
        t,
        f"{s['total_patches']:,}",
        f"{ps['mean']:.2f}",
        f"{ps['median']:.1f}",
        f"{ps['std']:.2f}",
        f"{aps:.2f}",
        f"{cr:.2f}x",
        f"{m['precision']:.4f}",
        f"{m['recall']:.4f}",
        f"{m['f1']:.4f}",
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#1e3a5f")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)

# Highlight best F1 row (tau=1.0, row index 1)
best_row_idx = TAU_LABELS.index(str(morph["best_tau"]))
for j in range(len(col_labels)):
    cell = table[best_row_idx + 1, j]
    cell.set_facecolor("#fef3c7")
    cell.set_edgecolor("#f59e0b")

# Alternate row shading
for i in range(len(table_data)):
    if i == best_row_idx:
        continue
    bg = "#f8fafc" if i % 2 == 0 else "#e2e8f0"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(bg)

ax.set_title(f"Entropy Threshold Sweep: Combined Results ({N:,} Marathi sentences)",
             fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("fig6_summary_table.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig6_summary_table.png")

print("\nAll 6 figures generated successfully.")
