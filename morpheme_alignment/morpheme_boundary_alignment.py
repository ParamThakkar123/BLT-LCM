"""
Patch boundary vs. morpheme boundary alignment study.

1. Uses all 50K Marathi sentences from the corpus.
2. Obtains gold morpheme segmentations via Indic NLP morphological analyzer.
3. For each tau in {0.5, 1.0, 1.5, 2.0, 2.5}: computes precision, recall, F1
   of entropy-based patch boundaries against morpheme boundaries.
4. Plots F1 vs. tau and annotates the best threshold.

Outputs:
  - morpheme_alignment_results.json   : per-tau P/R/F1 + per-sentence details
  - morpheme_f1_vs_tau.png            : publication-ready plot

Resumable: per-sentence rows stream to ``morpheme_alignment_per_sentence.jsonl``
as they are computed, so an interrupted run restarts at the first sentence it
had not written. Pass ``--resume never`` to force a clean run.
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

from lcm_scripts.checkpoint_utils import ResumableJsonl, config_fingerprint

# ── Indic NLP setup ──────────────────────────────────────────────────────────

from lcm_scripts.device_utils import report_cpu_only
from lcm_scripts.indic_resources import configure_indic_resources
from indicnlp import loader

configure_indic_resources()
loader.load()

from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── Load corpus ──────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--resume",
    default=os.environ.get("ALIGNMENT_RESUME", "auto"),
    metavar="auto|never",
    help="'auto' (default) continues from a partial per-sentence JSONL whose "
    "configuration matches; 'never' discards it and starts over.",
)
_parser.add_argument("--corpus", default="marathi_sentences.json")
_parser.add_argument("--out_jsonl", default="morpheme_alignment_per_sentence.jsonl")
_parser.add_argument("--out_json", default="morpheme_alignment_results.json")
_parser.add_argument("--out_plot", default="morpheme_f1_vs_tau.png")
ARGS = _parser.parse_args()
report_cpu_only("Indic NLP morphology and boundary scoring")

with open(ARGS.corpus, "r", encoding="utf-8") as f:
    all_sentences = json.load(f)

# Use all 50K sentences
sentences = all_sentences
print(f"Using all {len(sentences)} sentences for alignment study.")

# ── Build transition probabilities (same as sweep script) ────────────────────

byte_sequences_all = [s.encode("utf-8") for s in all_sentences]
transitions = defaultdict(Counter)
for seq in byte_sequences_all:
    for i in range(len(seq) - 1):
        transitions[seq[i]][seq[i + 1]] += 1

probs = {}
for prev, counter in transitions.items():
    total = sum(counter.values())
    probs[prev] = {nb: count / total for nb, count in counter.items()}


def compute_entropy(seq, default_entropy=8.0):
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


def get_patch_boundaries(seq, entropies, theta):
    """Return set of byte-offset positions where a patch boundary occurs."""
    boundaries = set()
    for i, h in enumerate(entropies):
        if h > theta:
            boundaries.add(i)
    return boundaries


# ── Get gold morpheme boundaries ─────────────────────────────────────────────

def get_morpheme_boundaries(sentence):
    """
    Tokenize sentence, morphologically segment each word, and return
    the set of byte offsets where morpheme boundaries fall.
    """
    tokens = trivial_tokenize(sentence, "mr")
    boundaries = set()

    byte_offset = 0
    text_bytes = sentence.encode("utf-8")

    for token in tokens:
        token_bytes = token.encode("utf-8")
        # Find token start in the byte stream
        pos = text_bytes.find(token_bytes, byte_offset)
        if pos == -1:
            byte_offset += len(token_bytes)
            continue

        # Get morpheme segmentation
        try:
            morphemes = morph_analyzer.morph_analyze(token)
        except Exception:
            morphemes = [token]

        if len(morphemes) > 1:
            # Mark boundaries between morphemes within this token
            morph_offset = pos
            for m_idx, morph in enumerate(morphemes[:-1]):
                morph_bytes = morph.encode("utf-8")
                morph_offset += len(morph_bytes)
                # Boundary is at the start of the next morpheme
                if morph_offset < len(text_bytes):
                    boundaries.add(morph_offset)

        # Also mark word boundaries (between tokens)
        token_end = pos + len(token_bytes)
        if token_end < len(text_bytes):
            boundaries.add(token_end)

        byte_offset = token_end

    return boundaries


# ── Compute alignment metrics ────────────────────────────────────────────────

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
TOLERANCE = 3  # byte tolerance for boundary match (Marathi UTF-8 chars = 3 bytes)


def _count_matched(query_sorted, ref_sorted, tolerance):
    """Count how many items in query have at least one ref within +-tolerance.
    Both lists must be sorted. Uses binary search for O(n log m)."""
    matched = 0
    for q in query_sorted:
        lo = bisect.bisect_left(ref_sorted, q - tolerance)
        hi = bisect.bisect_right(ref_sorted, q + tolerance)
        if lo < hi:
            matched += 1
    return matched


def boundaries_match(pred_boundaries, gold_boundaries, tolerance=TOLERANCE):
    """
    Compute precision, recall, F1 with a tolerance window.
    A predicted boundary counts as a true positive if any gold boundary
    is within +-tolerance bytes.
    """
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


# ── Run alignment study ─────────────────────────────────────────────────────

print("Computing morpheme boundaries and alignment metrics...")

FINGERPRINT = config_fingerprint(
    {
        "corpus": ARGS.corpus,
        "num_sentences": len(sentences),
        "thresholds": THRESHOLDS,
        "tolerance": TOLERANCE,
    }
)

# Morphological analysis is the slow part here; rows stream to disk so an
# interrupted run resumes at the first sentence it had not written.
writer = ResumableJsonl(
    ARGS.out_jsonl,
    fingerprint=FINGERPRINT,
    resume=ARGS.resume != "never",
    key="sentence_id",
)
if writer.done:
    print(f"Resuming: {len(writer.done)}/{len(sentences)} sentences already done")

N = len(sentences)
for idx, sent in enumerate(sentences):
    if writer.is_done(idx):
        continue
    seq = sent.encode("utf-8")
    entropies = compute_entropy(seq)
    gold = get_morpheme_boundaries(sent)

    row = {
        "sentence_id": idx,
        "byte_length": len(seq),
        "num_gold_boundaries": len(gold),
    }

    for tau in THRESHOLDS:
        pred = get_patch_boundaries(seq, entropies, tau)
        p, r, f1 = boundaries_match(pred, gold)

        key = f"tau_{tau}"
        row[key + "_num_pred"] = len(pred)
        row[key + "_precision"] = round(p, 4)
        row[key + "_recall"] = round(r, 4)
        row[key + "_f1"] = round(f1, 4)

    writer.append(row)

    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1}/{N} sentences...")

writer.close()
per_sentence = writer.all_records()

# Aggregate from the written rows rather than an in-memory accumulator, so a run
# stitched together from several attempts still averages over the full corpus.
aggregate = {tau: {"precisions": [], "recalls": [], "f1s": []} for tau in THRESHOLDS}
for row in per_sentence:
    for tau in THRESHOLDS:
        key = f"tau_{tau}"
        aggregate[tau]["precisions"].append(row[key + "_precision"])
        aggregate[tau]["recalls"].append(row[key + "_recall"])
        aggregate[tau]["f1s"].append(row[key + "_f1"])

# ── Aggregate results ────────────────────────────────────────────────────────

def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

results = {
    "study": "Patch boundary vs. morpheme boundary alignment",
    "num_sentences": len(sentences),
    "tolerance_bytes": TOLERANCE,
    "thresholds": {},
}

best_tau = None
best_f1 = -1

for tau in THRESHOLDS:
    avg_p = mean(aggregate[tau]["precisions"])
    avg_r = mean(aggregate[tau]["recalls"])
    avg_f1 = mean(aggregate[tau]["f1s"])

    results["thresholds"][str(tau)] = {
        "precision": round(avg_p, 4),
        "recall": round(avg_r, 4),
        "f1": round(avg_f1, 4),
    }

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_tau = tau

results["best_tau"] = best_tau
results["best_f1"] = round(best_f1, 4)
with open(ARGS.out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Per-sentence detail already lives in ARGS.out_jsonl, written incrementally by
# the resumable writer above.

# ── Print results table ─────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("PATCH BOUNDARY vs. MORPHEME BOUNDARY ALIGNMENT")
print(f"{len(sentences)} sentences | tolerance = +-{TOLERANCE} bytes")
print("=" * 64)
print(f"{'tau':>6} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Best':>6}")
print("-" * 64)
for tau in THRESHOLDS:
    d = results["thresholds"][str(tau)]
    marker = " <--" if tau == best_tau else ""
    print(f"{tau:>6.1f} | {d['precision']:>10.4f} | {d['recall']:>10.4f} | {d['f1']:>10.4f} |{marker}")
print("-" * 64)
print(f"Chosen tau = {best_tau} (F1 = {best_f1:.4f})")
print("=" * 64)

# ── Plot F1 vs. tau ─────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

taus = THRESHOLDS
f1_vals = [results["thresholds"][str(t)]["f1"] for t in taus]
prec_vals = [results["thresholds"][str(t)]["precision"] for t in taus]
rec_vals = [results["thresholds"][str(t)]["recall"] for t in taus]

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(taus, f1_vals, "o-", color="#2563eb", linewidth=2.2, markersize=8, label="F1", zorder=3)
ax.plot(taus, prec_vals, "s--", color="#16a34a", linewidth=1.5, markersize=6, label="Precision", alpha=0.8)
ax.plot(taus, rec_vals, "^--", color="#dc2626", linewidth=1.5, markersize=6, label="Recall", alpha=0.8)

# Annotate best tau
best_f1_val = results["thresholds"][str(best_tau)]["f1"]
ax.annotate(
    f"Best: tau={best_tau}\nF1={best_f1_val:.4f}",
    xy=(best_tau, best_f1_val),
    xytext=(best_tau + 0.35, best_f1_val - 0.04),
    fontsize=10,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1e293b", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.9),
)

ax.set_xlabel("Entropy Threshold (tau)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(f"Patch Boundary vs. Morpheme Boundary Alignment\n({len(sentences):,} Marathi sentences, BhashaSetu)", fontsize=13)
ax.set_xticks(taus)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(ARGS.out_plot, dpi=300, bbox_inches="tight")
print(f"\nPlot saved: {ARGS.out_plot}")
print(f"Results saved: {ARGS.out_json}")
print(f"Per-sentence detail: {ARGS.out_jsonl}")
