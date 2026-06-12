"""
Comprehensive comparison: Phase 1 Augmented vs Phase 1 Retrained vs Phase 2 BLT-LCM
Reads the full 50K-sentence fertility data and generates publication-quality charts.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(SCRIPT_DIR, "all_sentences_scores.jsonl")

MODEL_NAMES = [
    "Augmented Tokenizer",
    "Retrained Tokenizer",
    "BLT-LCM",
]
COLORS = ["#3498db", "#2ecc71", "#e74c3c"]

# ── Check if full data already exists, otherwise generate it ──

if not os.path.exists(DATA_PATH):
    print("Full scores file not found. Generating from 50K sentences...")
    print("(This will take ~20 min on GPU)")

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "patching_scratch"))
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "lcm_scripts"))

    import torch
    from tokenizers import Tokenizer
    from tqdm import tqdm
    import re

    from run_blt_patching import (
        text_to_byte_tokens, ByteEntropyModel,
        compute_entropies_for_tokens, entropy_patch_sentence, DEFAULT_THRESHOLD,
    )

    REPO_ROOT = os.path.join(PROJECT_ROOT, "..")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok_aug = Tokenizer.from_file(os.path.join(REPO_ROOT, "Tokeniser_Augmented", "tokenizer.json"))
    tok_ret = Tokenizer.from_file(os.path.join(REPO_ROOT, "Tokeniser_Retrained", "tokenizer.json"))

    ckpt = torch.load(os.path.join(PROJECT_ROOT, "patching_scratch", "entropy_model_marathi.pt"),
                       map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    entropy_model = ByteEntropyModel(
        vocab_size=cfg.get("vocab_size", 260), dim=cfg.get("dim", 256),
        n_heads=cfg.get("n_heads", 4), n_layers=cfg.get("n_layers", 4),
        max_seqlen=cfg.get("max_seqlen", 512),
        ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", 1.3),
    ).to(device)
    entropy_model.load_state_dict(ckpt["model_state_dict"])
    entropy_model.eval()

    with open(os.path.join(PROJECT_ROOT, "marathi_sentences.json"), "r", encoding="utf-8") as f:
        all_sentences = json.load(f)

    def split_words(t):
        return [w for w in re.split(r"\s+", t.strip()) if w]

    with open(DATA_PATH, "w", encoding="utf-8") as fout:
        for idx, text in enumerate(tqdm(all_sentences, desc="Scoring")):
            text = text.strip()
            if not text:
                continue
            words = split_words(text)
            nw = len(words)
            if nw == 0:
                continue

            n_aug = len(tok_aug.encode(text).ids)
            n_ret = len(tok_ret.encode(text).ids)

            byte_tokens = text_to_byte_tokens(text)
            if len(byte_tokens) == 0:
                continue
            tokens_tensor = torch.tensor([byte_tokens], dtype=torch.long).to(device)
            entropies = compute_entropies_for_tokens(tokens_tensor, entropy_model, device=device)
            boundaries, _ = entropy_patch_sentence(entropies[0].tolist(), DEFAULT_THRESHOLD)
            n_patches = len(boundaries)

            rec = {
                "idx": idx, "n_words": nw, "n_bytes": len(byte_tokens),
                "tok_aug": n_aug, "tok_ret": n_ret, "patches_blt": n_patches,
                "fert_aug": round(n_aug / nw, 4),
                "fert_ret": round(n_ret / nw, 4),
                "fert_blt": round(n_patches / nw, 4),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {DATA_PATH}")

# ── Load data ─────────────────────────────────────────────────

print("Loading scores...")
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

fert_aug = np.array([r["fert_aug"] for r in records])
fert_ret = np.array([r["fert_ret"] for r in records])
fert_blt = np.array([r["fert_blt"] for r in records])
n_words  = np.array([r["n_words"] for r in records])
n_bytes  = np.array([r["n_bytes"] for r in records])
tok_aug  = np.array([r["tok_aug"] for r in records])
tok_ret  = np.array([r["tok_ret"] for r in records])
patches  = np.array([r["patches_blt"] for r in records])

N = len(records)
print(f"Loaded {N} sentences.")

# =====================================================================
# FIGURE 1: Mean Fertility Comparison (Bar Chart)
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
means = [fert_aug.mean(), fert_ret.mean(), fert_blt.mean()]
stds  = [fert_aug.std(), fert_ret.std(), fert_blt.std()]
bars = ax.bar(MODEL_NAMES, means, yerr=stds, color=COLORS, edgecolor="white",
              linewidth=0.8, capsize=6, error_kw={"linewidth": 1.2})
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{m:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_ylabel("Mean Fertility (tokens or patches per word)")
ax.set_title("Mean Fertility: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM")
ax.set_ylim(0, max(means) + max(stds) + 0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp1_mean_fertility.png"))
plt.close()
print("Saved cmp1_mean_fertility.png")

# =====================================================================
# FIGURE 2: Fertility Distribution (Overlaid Histograms)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.linspace(0.8, 3.5, 60)
ax.hist(fert_aug, bins=bins, alpha=0.55, color=COLORS[0], label=MODEL_NAMES[0], density=True)
ax.hist(fert_ret, bins=bins, alpha=0.55, color=COLORS[1], label=MODEL_NAMES[1], density=True)
ax.hist(fert_blt, bins=bins, alpha=0.55, color=COLORS[2], label=MODEL_NAMES[2], density=True)
ax.axvline(fert_aug.mean(), color=COLORS[0], linestyle="--", linewidth=1.5)
ax.axvline(fert_ret.mean(), color=COLORS[1], linestyle="--", linewidth=1.5)
ax.axvline(fert_blt.mean(), color=COLORS[2], linestyle="--", linewidth=1.5)
ax.set_xlabel("Fertility (tokens or patches per word)")
ax.set_ylabel("Density")
ax.set_title("Fertility Distribution: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp2_fertility_distribution.png"))
plt.close()
print("Saved cmp2_fertility_distribution.png")

# =====================================================================
# FIGURE 3: Fertility by Sentence Length Bucket
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
buckets = [(1, 4, "<5"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 39, "20-39"), (40, 1000, "40+")]
bucket_labels = []
bucket_aug, bucket_ret, bucket_blt = [], [], []
for lo, hi, label in buckets:
    mask = (n_words >= lo) & (n_words <= hi)
    if mask.sum() == 0:
        continue
    bucket_labels.append(f"{label}\n(n={mask.sum()})")
    bucket_aug.append(fert_aug[mask].mean())
    bucket_ret.append(fert_ret[mask].mean())
    bucket_blt.append(fert_blt[mask].mean())

x = np.arange(len(bucket_labels))
w = 0.25
ax.bar(x - w, bucket_aug, w, color=COLORS[0], label=MODEL_NAMES[0], edgecolor="white")
ax.bar(x,     bucket_ret, w, color=COLORS[1], label=MODEL_NAMES[1], edgecolor="white")
ax.bar(x + w, bucket_blt, w, color=COLORS[2], label=MODEL_NAMES[2], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(bucket_labels)
ax.set_xlabel("Sentence Length (words)")
ax.set_ylabel("Mean Fertility")
ax.set_title("Mean Fertility by Sentence Length: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp3_fertility_by_length.png"))
plt.close()
print("Saved cmp3_fertility_by_length.png")

# =====================================================================
# FIGURE 4: Win/Tie/Loss Matrix (which model is best per sentence)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 6))

best_aug = ((fert_aug < fert_ret) & (fert_aug < fert_blt)).sum()
best_ret = ((fert_ret < fert_aug) & (fert_ret < fert_blt)).sum()
best_blt = ((fert_blt < fert_aug) & (fert_blt < fert_ret)).sum()
tied = N - best_aug - best_ret - best_blt

sizes = [best_blt, best_ret, best_aug, tied]
colors_pie = [COLORS[2], COLORS[1], COLORS[0], "#bdc3c7"]
legend_labels = [
    f"BLT-LCM  ({best_blt:,} = {100*best_blt/N:.1f}%)",
    f"Retrained Tokenizer  ({best_ret:,} = {100*best_ret/N:.1f}%)",
    f"Augmented Tokenizer  ({best_aug:,} = {100*best_aug/N:.1f}%)",
    f"Tied  ({tied:,} = {100*tied/N:.1f}%)",
]
wedges, _ = ax.pie(sizes, colors=colors_pie, startangle=140,
                    wedgeprops=dict(edgecolor="white", linewidth=1.5))
ax.legend(wedges, legend_labels, title="Winner", loc="center left",
          bbox_to_anchor=(1.0, 0.5), fontsize=11, title_fontsize=12,
          frameon=False)
ax.set_title("Per-Sentence Winner: Which Model Achieves Lowest Fertility?",
             fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp4_winner_pie.png"), bbox_inches="tight")
plt.close()
print("Saved cmp4_winner_pie.png")

# =====================================================================
# FIGURE 5: Cumulative Fertility CDF
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5))
for fert, name, color in zip([fert_aug, fert_ret, fert_blt], MODEL_NAMES, COLORS):
    sorted_f = np.sort(fert)
    cdf = np.arange(1, len(sorted_f) + 1) / len(sorted_f)
    ax.plot(sorted_f, cdf, color=color, linewidth=2, label=name)
ax.set_xlabel("Fertility (tokens or patches per word)")
ax.set_ylabel("Cumulative Proportion of Sentences")
ax.set_title("Cumulative Distribution: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM")
ax.set_xlim(0.8, 3.0)
ax.legend(frameon=False)
ax.grid(alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp5_fertility_cdf.png"))
plt.close()
print("Saved cmp5_fertility_cdf.png")

# =====================================================================
# FIGURE 6: Pairwise Scatter — BLT vs Augmented & BLT vs Retrained
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, fert_p1, name, color in zip(axes, [fert_aug, fert_ret],
                                      [MODEL_NAMES[0], MODEL_NAMES[1]],
                                      [COLORS[0], COLORS[1]]):
    subsample = np.random.default_rng(42).choice(N, size=min(3000, N), replace=False)
    ax.scatter(fert_p1[subsample], fert_blt[subsample],
               alpha=0.25, s=12, c=color, edgecolors="none")
    lims = [0.8, 3.5]
    ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{name}\nFertility")
    ax.set_ylabel("BLT-LCM Fertility")
    blt_wins = (fert_blt[subsample] < fert_p1[subsample]).sum()
    p1_wins = (fert_p1[subsample] < fert_blt[subsample]).sum()
    ties = len(subsample) - blt_wins - p1_wins
    ax.set_title(f"BLT-LCM vs {name}\nBLT wins: {blt_wins} | {name} wins: {p1_wins} | Tied: {ties}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("Pairwise Fertility Comparison: BLT-LCM vs Tokenizers",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp6_pairwise_scatter.png"), bbox_inches="tight")
plt.close()
print("Saved cmp6_pairwise_scatter.png")

# =====================================================================
# FIGURE 7: Compression Efficiency — Tokens/Patches per Byte
# =====================================================================
fig, ax = plt.subplots(figsize=(8, 5))
eff_aug = tok_aug / n_bytes
eff_ret = tok_ret / n_bytes
eff_blt = patches / n_bytes

means_eff = [eff_aug.mean(), eff_ret.mean(), eff_blt.mean()]
bars = ax.bar(MODEL_NAMES, means_eff, color=COLORS, edgecolor="white", linewidth=0.8)
for bar, m in zip(bars, means_eff):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{m:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.set_ylabel("Tokens (or Patches) per Byte")
ax.set_title("Compression Efficiency: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM\n(lower = more compressed)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp7_compression_efficiency.png"))
plt.close()
print("Saved cmp7_compression_efficiency.png")

# =====================================================================
# FIGURE 8: Box Plot — Fertility Distribution Side by Side
# =====================================================================
fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot([fert_aug, fert_ret, fert_blt], labels=MODEL_NAMES,
                patch_artist=True, widths=0.5,
                boxprops=dict(linewidth=0.8),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
                showfliers=True)
for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

stats_text = (
    f"Medians: {np.median(fert_aug):.3f} | {np.median(fert_ret):.3f} | {np.median(fert_blt):.3f}\n"
    f"Means:   {fert_aug.mean():.3f} | {fert_ret.mean():.3f} | {fert_blt.mean():.3f}"
)
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="right", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
ax.set_ylabel("Fertility (tokens or patches per word)")
ax.set_title("Fertility Distribution: Augmented Tokenizer vs Retrained Tokenizer vs BLT-LCM")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "cmp8_fertility_boxplot.png"))
plt.close()
print("Saved cmp8_fertility_boxplot.png")

# =====================================================================
# FIGURE 9: Patch Compression Ratio by Morpheme Class
# =====================================================================
morpheme_detail_path = os.path.join(PROJECT_ROOT, "results", "fertility_by_class_detail.jsonl")
if os.path.exists(morpheme_detail_path):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from tokenization_statistics.patch_compression_by_morpheme_class import (
        build_analysis,
        load_sentence_classes,
        plot_class_comparison,
        write_sentence_csv,
    )

    score_records = {
        int(r["idx"]): {
            "sentence_id": int(r["idx"]),
            "text": r.get("marathi_text", ""),
            "blt_patches": int(r["patches_blt"]),
            "bpe_tokens": int(r["tok_ret"]),
            "num_bytes": int(r["n_bytes"]),
        }
        for r in records
    }
    sentence_classes = load_sentence_classes(morpheme_detail_path)
    sentence_rows, class_summary = build_analysis(score_records, sentence_classes)
    class_summary["inputs"] = {
        "blt_or_scores_source": DATA_PATH,
        "bpe_source": f"{DATA_PATH}:tok_ret",
        "morpheme_detail": morpheme_detail_path,
    }

    cmp9_csv = os.path.join(SCRIPT_DIR, "cmp9_patch_bpe_sentence_comparison.csv")
    cmp9_json = os.path.join(SCRIPT_DIR, "cmp9_patch_compression_by_morpheme_class.json")
    cmp9_png = os.path.join(SCRIPT_DIR, "cmp9_patch_compression_by_morpheme_class.png")
    write_sentence_csv(sentence_rows, cmp9_csv)
    with open(cmp9_json, "w", encoding="utf-8") as f:
        json.dump(class_summary, f, indent=2, ensure_ascii=False)
    plot_class_comparison(class_summary, cmp9_png)
    print("Saved cmp9_patch_compression_by_morpheme_class.png")
else:
    print(f"Skipping Figure 9: morpheme detail file not found at {morpheme_detail_path}")

# ── Print summary table ──────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (50K sentences)")
print("=" * 70)
print(f"{'Metric':<30} {'Augmented':>12} {'Retrained':>12} {'BLT-LCM':>12}")
print("-" * 70)
print(f"{'Mean Fertility':<30} {fert_aug.mean():>12.4f} {fert_ret.mean():>12.4f} {fert_blt.mean():>12.4f}")
print(f"{'Median Fertility':<30} {np.median(fert_aug):>12.4f} {np.median(fert_ret):>12.4f} {np.median(fert_blt):>12.4f}")
print(f"{'Std Fertility':<30} {fert_aug.std():>12.4f} {fert_ret.std():>12.4f} {fert_blt.std():>12.4f}")
print(f"{'Tokens/Patches per Byte':<30} {eff_aug.mean():>12.4f} {eff_ret.mean():>12.4f} {eff_blt.mean():>12.4f}")
print(f"{'% sentences fertility ≤ 1.0':<30} {100*(fert_aug<=1.0).mean():>11.1f}% {100*(fert_ret<=1.0).mean():>11.1f}% {100*(fert_blt<=1.0).mean():>11.1f}%")
print(f"{'Best on N sentences':<30} {best_aug:>12,} {best_ret:>12,} {best_blt:>12,}")
print(f"{'Tied':<30} {tied:>12,}")
print("=" * 70)
print("\nAll comparison figures saved.")
