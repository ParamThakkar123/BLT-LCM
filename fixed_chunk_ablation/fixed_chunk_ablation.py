"""
Fixed-length chunking ablation study.

Tests whether entropy-adaptive boundary placement outperforms naive
fixed-length chunking at morpheme boundary alignment.

1. Loads all 50K Marathi sentences.
2. Creates fixed-length chunk boundaries at every K bytes (K = 4, 8).
3. Computes precision / recall / F1 against gold morpheme boundaries
   with the same +-3-byte tolerance used in the entropy-adaptive study.
4. Compares against the entropy-adaptive results (tau = 1.0, F1 = 0.6341).
5. Runs tolerance sensitivity analysis (tol = 0, 1, 2, 3 bytes).
6. Computes boundary efficiency (F1 per boundary).

Outputs:
  - fixed_chunk_ablation_results.json
  - fixed_chunk_ablation_per_sentence.jsonl
  - fixed_chunk_ablation_comparison.png
  - fixed_chunk_tolerance_sensitivity.png
"""

import argparse
import sys
import io
import json
import math
import os
import bisect
from collections import defaultdict, Counter

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_scripts.checkpoint_utils import (
    ResumableJsonl,
    StageTracker,
    config_fingerprint,
)

# ── Indic NLP setup ──────────────────────────────────────────────────────────

from lcm_scripts.device_utils import report_cpu_only
from lcm_scripts.indic_resources import configure_indic_resources
from indicnlp import loader

configure_indic_resources()
loader.load()

from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── CLI ──────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--resume",
    default=os.environ.get("ABLATION_RESUME", "auto"),
    metavar="auto|never",
    help="'auto' (default) continues from partial per-sentence output and any "
    "completed tolerance stages; 'never' discards them and starts over.",
)
_parser.add_argument("--corpus", default="marathi_sentences.json")
_parser.add_argument("--entropy_results", default="morpheme_alignment_results.json")
_parser.add_argument("--out_json", default="fixed_chunk_ablation_results.json")
_parser.add_argument("--out_jsonl", default="fixed_chunk_ablation_per_sentence.jsonl")
ARGS = _parser.parse_args()
report_cpu_only("fixed-length chunking and boundary scoring")

# ── Load corpus ──────────────────────────────────────────────────────────────

with open(ARGS.corpus, "r", encoding="utf-8") as f:
    sentences = json.load(f)

print(f"Loaded {len(sentences)} sentences for fixed-chunk ablation.")

# ── Build byte-level transition probabilities (for entropy-adaptive) ────────

print("Building bigram transition probabilities...")
byte_sequences_all = [s.encode("utf-8") for s in sentences]
transitions = defaultdict(Counter)
for seq in byte_sequences_all:
    for i in range(len(seq) - 1):
        transitions[seq[i]][seq[i + 1]] += 1

probs = {}
for prev, counter in transitions.items():
    total = sum(counter.values())
    probs[prev] = {nb: count / total for nb, count in counter.items()}


def compute_entropy(seq, default_entropy=8.0):
    """Per-position conditional entropy: H(next_byte | prev_byte)."""
    entropies = []
    for i in range(len(seq) - 1):
        prev = seq[i]
        if prev in probs:
            h = 0.0
            for p in probs[prev].values():
                if p > 0:
                    h -= p * math.log2(p)
            entropies.append(h)
        else:
            entropies.append(default_entropy)
    entropies.append(default_entropy)
    return entropies


def get_entropy_boundaries(seq, entropies, theta):
    """Return set of byte-offset positions where entropy exceeds threshold."""
    return {i for i, h in enumerate(entropies) if h > theta}


# ── Fixed-length boundary generation ─────────────────────────────────────────

CHUNK_SIZES = [4, 8]


def get_fixed_boundaries(byte_length, chunk_size):
    """Return set of byte-offset positions where fixed-length boundaries fall."""
    return {i for i in range(chunk_size, byte_length, chunk_size)}


# ── Gold morpheme boundary extraction ────────────────────────────────────────

def get_morpheme_boundaries(sentence):
    tokens = trivial_tokenize(sentence, "mr")
    boundaries = set()
    byte_offset = 0
    text_bytes = sentence.encode("utf-8")

    for token in tokens:
        token_bytes = token.encode("utf-8")
        pos = text_bytes.find(token_bytes, byte_offset)
        if pos == -1:
            byte_offset += len(token_bytes)
            continue

        try:
            morphemes = morph_analyzer.morph_analyze(token)
        except Exception:
            morphemes = [token]

        if len(morphemes) > 1:
            morph_offset = pos
            for morph in morphemes[:-1]:
                morph_bytes = morph.encode("utf-8")
                morph_offset += len(morph_bytes)
                if morph_offset < len(text_bytes):
                    boundaries.add(morph_offset)

        token_end = pos + len(token_bytes)
        if token_end < len(text_bytes):
            boundaries.add(token_end)

        byte_offset = token_end

    return boundaries


# ── Alignment metrics ────────────────────────────────────────────────────────

TOLERANCE = 3  # Marathi UTF-8 chars ~ 3 bytes


def _count_matched(query_sorted, ref_sorted, tolerance):
    matched = 0
    for q in query_sorted:
        lo = bisect.bisect_left(ref_sorted, q - tolerance)
        hi = bisect.bisect_right(ref_sorted, q + tolerance)
        if lo < hi:
            matched += 1
    return matched


def boundaries_match(pred_boundaries, gold_boundaries, tolerance=TOLERANCE):
    if len(pred_boundaries) == 0 and len(gold_boundaries) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_boundaries) == 0:
        return 0.0, 0.0, 0.0
    if len(gold_boundaries) == 0:
        return 0.0, 0.0, 0.0

    gold_list = sorted(gold_boundaries)
    pred_list = sorted(pred_boundaries)

    tp_prec = _count_matched(pred_list, gold_list, tolerance)
    tp_rec = _count_matched(gold_list, pred_list, tolerance)

    precision = tp_prec / len(pred_list)
    recall = tp_rec / len(gold_list)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


# ── Part 1: Fixed-chunk ablation at default tolerance +-3 ──────────────────

print("\nPart 1: Fixed-chunk boundary F1 (tolerance = +-3 bytes)...")

FINGERPRINT = config_fingerprint(
    {
        "corpus": ARGS.corpus,
        "num_sentences": len(sentences),
        "chunk_sizes": CHUNK_SIZES,
        "tolerance": TOLERANCE,
    }
)

# Gold morpheme segmentation dominates the runtime, so rows stream to disk and
# an interrupted pass resumes at the first sentence it had not written.
writer = ResumableJsonl(
    ARGS.out_jsonl,
    fingerprint=FINGERPRINT,
    resume=ARGS.resume != "never",
    key="sentence_id",
)
if writer.done:
    print(f"Resuming Part 1: {len(writer.done)}/{len(sentences)} sentences already done")

N = len(sentences)
for idx, sent in enumerate(sentences):
    if writer.is_done(idx):
        continue
    seq = sent.encode("utf-8")
    gold = get_morpheme_boundaries(sent)

    row = {
        "sentence_id": idx,
        "byte_length": len(seq),
        "num_gold_boundaries": len(gold),
    }

    for k in CHUNK_SIZES:
        pred = get_fixed_boundaries(len(seq), k)
        p, r, f1 = boundaries_match(pred, gold)

        key = f"fixed_{k}"
        row[key + "_num_pred"] = len(pred)
        row[key + "_precision"] = round(p, 4)
        row[key + "_recall"] = round(r, 4)
        row[key + "_f1"] = round(f1, 4)

    writer.append(row)

    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1}/{N} sentences...")

writer.close()
per_sentence = writer.all_records()

# Aggregate from the written rows so a run assembled from several attempts still
# averages over the whole corpus.
aggregate = {k: {"precisions": [], "recalls": [], "f1s": []} for k in CHUNK_SIZES}
for row in per_sentence:
    for k in CHUNK_SIZES:
        key = f"fixed_{k}"
        aggregate[k]["precisions"].append(row[key + "_precision"])
        aggregate[k]["recalls"].append(row[key + "_recall"])
        aggregate[k]["f1s"].append(row[key + "_f1"])

# Load entropy-adaptive baseline
with open(ARGS.entropy_results, "r", encoding="utf-8") as f:
    entropy_results = json.load(f)

entropy_best_tau = entropy_results["best_tau"]
entropy_best = entropy_results["thresholds"][str(entropy_best_tau)]

results = {
    "study": "Fixed-length chunking ablation vs. entropy-adaptive boundaries",
    "num_sentences": len(sentences),
    "tolerance_bytes": TOLERANCE,
    "fixed_chunk_results": {},
    "entropy_adaptive_baseline": {
        "tau": entropy_best_tau,
        "precision": entropy_best["precision"],
        "recall": entropy_best["recall"],
        "f1": entropy_best["f1"],
    },
}

for k in CHUNK_SIZES:
    avg_p = mean(aggregate[k]["precisions"])
    avg_r = mean(aggregate[k]["recalls"])
    avg_f1 = mean(aggregate[k]["f1s"])
    avg_num_pred = mean([r[f"fixed_{k}_num_pred"] for r in per_sentence])

    results["fixed_chunk_results"][str(k)] = {
        "chunk_size_bytes": k,
        "precision": round(avg_p, 4),
        "recall": round(avg_r, 4),
        "f1": round(avg_f1, 4),
        "avg_boundaries_per_sentence": round(avg_num_pred, 2),
        "delta_f1_vs_entropy": round(avg_f1 - entropy_best["f1"], 4),
    }

# Print comparison table
print("\n" + "=" * 78)
print("FIXED-LENGTH CHUNKING ABLATION vs. ENTROPY-ADAPTIVE BOUNDARIES")
print(f"{len(sentences):,} sentences | tolerance = +-{TOLERANCE} bytes")
print("=" * 78)
print(f"{'Method':>24} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Delta F1':>10}")
print("-" * 78)

print(f"{'Entropy (tau=' + str(entropy_best_tau) + ')':>24} | "
      f"{entropy_best['precision']:>10.4f} | "
      f"{entropy_best['recall']:>10.4f} | "
      f"{entropy_best['f1']:>10.4f} | "
      f"{'(baseline)':>10}")

for k in CHUNK_SIZES:
    d = results["fixed_chunk_results"][str(k)]
    print(f"{'Fixed ' + str(k) + '-byte':>24} | "
          f"{d['precision']:>10.4f} | "
          f"{d['recall']:>10.4f} | "
          f"{d['f1']:>10.4f} | "
          f"{d['delta_f1_vs_entropy']:>+10.4f}")

print("=" * 78)

# ── Part 2: Tolerance sensitivity analysis ──────────────────────────────────

print("\nPart 2: Tolerance sensitivity (tol = 0, 1, 2, 3 bytes)...")

TOLERANCES = [0, 1, 2, 3]
tolerance_results = {}

# Each tolerance is another full pass over the corpus with morphological
# analysis, so completed ones are memoized and a rerun starts at the first
# tolerance it had not finished.
tol_stages = StageTracker(
    os.path.splitext(ARGS.out_json)[0] + ".tolerance_state.json",
    fingerprint=FINGERPRINT,
    resume=ARGS.resume != "never",
)


def _run_tolerance(tol):
    tol_agg = {
        "entropy_adaptive": {"precisions": [], "recalls": [], "f1s": []},
    }
    for k in CHUNK_SIZES:
        tol_agg[f"fixed_{k}"] = {"precisions": [], "recalls": [], "f1s": []}

    for sent in sentences:
        seq = sent.encode("utf-8")
        gold = get_morpheme_boundaries(sent)
        entropies = compute_entropy(seq)

        # Entropy-adaptive boundaries
        pred_ent = get_entropy_boundaries(seq, entropies, 1.0)
        p, r, f1 = boundaries_match(pred_ent, gold, tolerance=tol)
        tol_agg["entropy_adaptive"]["precisions"].append(p)
        tol_agg["entropy_adaptive"]["recalls"].append(r)
        tol_agg["entropy_adaptive"]["f1s"].append(f1)

        # Fixed-length boundaries
        for k in CHUNK_SIZES:
            pred_fix = get_fixed_boundaries(len(seq), k)
            p, r, f1 = boundaries_match(pred_fix, gold, tolerance=tol)
            tol_agg[f"fixed_{k}"]["precisions"].append(p)
            tol_agg[f"fixed_{k}"]["recalls"].append(r)
            tol_agg[f"fixed_{k}"]["f1s"].append(f1)

    return {
        method_key: {
            "precision": round(mean(tol_agg[method_key]["precisions"]), 4),
            "recall": round(mean(tol_agg[method_key]["recalls"]), 4),
            "f1": round(mean(tol_agg[method_key]["f1s"]), 4),
        }
        for method_key in ["entropy_adaptive"] + [f"fixed_{k}" for k in CHUNK_SIZES]
    }


for tol in TOLERANCES:
    tolerance_results[tol] = tol_stages.run(
        f"tolerance={tol}", lambda tol=tol: _run_tolerance(tol)
    )
    print(f"  Tolerance = {tol} done.")

# Print tolerance sensitivity table
print("\n" + "=" * 78)
print("TOLERANCE SENSITIVITY ANALYSIS")
print("=" * 78)
print(f"{'Tol':>4} | {'Method':>20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 70)
for tol in TOLERANCES:
    for method_key, label in [("entropy_adaptive", "Entropy (tau=1.0)"),
                               ("fixed_4", "Fixed 4-byte"),
                               ("fixed_8", "Fixed 8-byte")]:
        d = tolerance_results[tol][method_key]
        print(f"{tol:>4} | {label:>20} | {d['precision']:>10.4f} | {d['recall']:>10.4f} | {d['f1']:>10.4f}")
    print("-" * 70)

results["tolerance_sensitivity"] = {str(t): v for t, v in tolerance_results.items()}

# ── Part 3: Boundary efficiency ─────────────────────────────────────────────

print("\n" + "=" * 78)
print("BOUNDARY EFFICIENCY (F1 per 100 boundaries)")
print("=" * 78)

entropy_boundaries = 332.86 - 1  # patches - 1 from sweep at tau=1.0; TODO: compute from actual sweep data instead of hardcoding
methods_eff = [
    ("Entropy (tau=1.0)", entropy_best["f1"], entropy_boundaries),
    ("Fixed 4-byte", results["fixed_chunk_results"]["4"]["f1"],
     results["fixed_chunk_results"]["4"]["avg_boundaries_per_sentence"]),
    ("Fixed 8-byte", results["fixed_chunk_results"]["8"]["f1"],
     results["fixed_chunk_results"]["8"]["avg_boundaries_per_sentence"]),
]

print(f"{'Method':>20} | {'F1':>8} | {'Avg Boundaries':>16} | {'F1 / 100 bnd':>14}")
print("-" * 68)
for name, f1, bnd in methods_eff:
    eff = (f1 / bnd) * 100 if bnd > 0 else 0
    print(f"{name:>20} | {f1:>8.4f} | {bnd:>16.1f} | {eff:>14.4f}")

results["boundary_efficiency"] = {}
for name, f1, bnd in methods_eff:
    results["boundary_efficiency"][name] = {
        "f1": f1,
        "avg_boundaries": round(bnd, 2),
        "f1_per_100_boundaries": round((f1 / bnd) * 100, 4) if bnd > 0 else 0,
    }

print("=" * 78)

# ── Save all results ────────────────────────────────────────────────────────

with open(ARGS.out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Per-sentence rows already live in ARGS.out_jsonl, written incrementally above.

# ── Plot 1: Bar chart comparison (tolerance = 3) ───────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

methods = [f"Entropy-adaptive\n(tau={entropy_best_tau})",
           "Fixed 4-byte", "Fixed 8-byte"]
f1_scores = [
    entropy_best["f1"],
    results["fixed_chunk_results"]["4"]["f1"],
    results["fixed_chunk_results"]["8"]["f1"],
]
prec_scores = [
    entropy_best["precision"],
    results["fixed_chunk_results"]["4"]["precision"],
    results["fixed_chunk_results"]["8"]["precision"],
]
rec_scores = [
    entropy_best["recall"],
    results["fixed_chunk_results"]["4"]["recall"],
    results["fixed_chunk_results"]["8"]["recall"],
]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))

bars_p = ax.bar(x - width, prec_scores, width, label="Precision",
                color="#16a34a", alpha=0.85, edgecolor="white", linewidth=0.8)
bars_r = ax.bar(x, rec_scores, width, label="Recall",
                color="#dc2626", alpha=0.85, edgecolor="white", linewidth=0.8)
bars_f = ax.bar(x + width, f1_scores, width, label="F1",
                color="#2563eb", alpha=0.85, edgecolor="white", linewidth=0.8)

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.axhline(y=entropy_best["f1"], color="#2563eb", linestyle="--",
           alpha=0.4, linewidth=1, label=f"Entropy F1 = {entropy_best['f1']:.4f}")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Fixed-Length Chunking Ablation vs. Entropy-Adaptive Boundaries\n"
             f"({len(sentences):,} Marathi sentences, tolerance = +-{TOLERANCE} bytes)",
             fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fixed_chunk_ablation_comparison.png", dpi=300, bbox_inches="tight")
print("\nPlot saved: fixed_chunk_ablation_comparison.png")

# ── Plot 2: Tolerance sensitivity ───────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))

for method_key, label, color, marker in [
    ("entropy_adaptive", "Entropy-adaptive (tau=1.0)", "#2563eb", "o"),
    ("fixed_4", "Fixed 4-byte", "#f59e0b", "s"),
    ("fixed_8", "Fixed 8-byte", "#9333ea", "^"),
]:
    f1_by_tol = [tolerance_results[t][method_key]["f1"] for t in TOLERANCES]
    ax2.plot(TOLERANCES, f1_by_tol, f"{marker}-", color=color,
             linewidth=2, markersize=8, label=label)

ax2.set_xlabel("Tolerance (bytes)", fontsize=12)
ax2.set_ylabel("F1 Score", fontsize=12)
ax2.set_title("Boundary F1 vs. Matching Tolerance\n"
              f"({len(sentences):,} Marathi sentences)", fontsize=13)
ax2.set_xticks(TOLERANCES)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fixed_chunk_tolerance_sensitivity.png", dpi=300, bbox_inches="tight")
print("Tolerance plot saved: fixed_chunk_tolerance_sensitivity.png")

print("\nAll results saved: fixed_chunk_ablation_results.json")
print("Per-sentence details: fixed_chunk_ablation_per_sentence.jsonl")
