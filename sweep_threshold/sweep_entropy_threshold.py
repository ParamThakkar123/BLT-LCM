"""
Entropy threshold sweep script.

Sweeps τ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} over the full 50K Marathi sentence
corpus, recording per-sentence patch counts at each threshold.

Outputs:
  - sweep_results.jsonl   : one JSON object per sentence with patch counts
  - sweep_summary.json    : aggregate statistics per threshold
"""

import json
import math
import csv
from collections import defaultdict, Counter

# For morpheme boundary evaluation on a small sample
from lcm_scripts.indic_resources import configure_indic_resources
from indicnlp import loader

configure_indic_resources()
loader.load()
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

# Initialize morph analyzer for Marathi
morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── Load corpus ──────────────────────────────────────────────────────────────

with open("marathi_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

byte_sequences = [s.encode("utf-8") for s in sentences]
print(f"Loaded {len(sentences)} sentences.")

# ── Build byte-level transition probabilities ────────────────────────────────

transitions = defaultdict(Counter)
for seq in byte_sequences:
    for i in range(len(seq) - 1):
        transitions[seq[i]][seq[i + 1]] += 1

probs = {}
for prev, counter in transitions.items():
    total = sum(counter.values())
    probs[prev] = {nb: count / total for nb, count in counter.items()}


# ── Entropy helpers ──────────────────────────────────────────────────────────

def compute_entropy(seq, default_entropy=8.0):
    """Per-position conditional entropy: H(next_byte | prev_byte).
    
    NOTE: This uses a bigram co-occurrence model (empirical conditional entropy
    from corpus statistics), NOT the neural ByteEntropyModel used in
    run_blt_patching.py and blt_loader.py. Results from this sweep may differ
    systematically from the neural model's entropy estimates.
    """
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
    entropies.append(default_entropy)  # last byte
    return entropies


def entropy_patching(seq, entropies, theta):
    """Return list of patches (as byte strings) by splitting where H > θ."""
    patches = []
    current_patch = []
    for byte_val, h in zip(seq, entropies):
        current_patch.append(byte_val)
        if h > theta and len(current_patch) > 0:
            patches.append(bytes(current_patch))
            current_patch = []
    if current_patch:
        patches.append(bytes(current_patch))
    return patches


# ── Sweep configuration ─────────────────────────────────────────────────────

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
TOLERANCE = 3  # byte tolerance for boundary match (Marathi UTF-8 chars = 3 bytes)
EVAL_SAMPLE_SIZE = 200  # number of sentences to use when computing boundary P/R/F1

# ── Run sweep ────────────────────────────────────────────────────────────────

results = []
# Accumulators for summary statistics per threshold
summary = {tau: {"patch_counts": [], "avg_patch_sizes": []} for tau in THRESHOLDS}

for idx, seq in enumerate(byte_sequences):
    entropies = compute_entropy(seq)
    row = {
        "sentence_id": idx,
        "byte_length": len(seq),
    }
    for tau in THRESHOLDS:
        patches = entropy_patching(seq, entropies, tau)
        num_patches = len(patches)
        patch_lengths = [len(p) for p in patches]
        avg_size = sum(patch_lengths) / num_patches if num_patches else 0

        key = f"tau_{tau}"
        row[key + "_num_patches"] = num_patches
        row[key + "_avg_patch_size"] = round(avg_size, 2)

        summary[tau]["patch_counts"].append(num_patches)
        summary[tau]["avg_patch_sizes"].append(avg_size)

    results.append(row)

    if (idx + 1) % 2000 == 0:
        print(f"  Processed {idx + 1}/{len(sentences)} sentences...")

# ── Write per-sentence JSONL ─────────────────────────────────────────────────

with open("sweep_results.jsonl", "w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ── Write per-sentence CSV (easy to inspect) ─────────────────────────────────

csv_cols = ["sentence_id", "byte_length"]
for tau in THRESHOLDS:
    csv_cols.append(f"tau_{tau}_num_patches")
    csv_cols.append(f"tau_{tau}_avg_patch_size")

with open("sweep_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    writer.writerows(results)

# ── Compute and write summary ────────────────────────────────────────────────

def stats(values):
    n = len(values)
    mean = sum(values) / n
    sorted_v = sorted(values)
    median = sorted_v[n // 2] if n % 2 else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    return {
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


summary_out = {"num_sentences": len(sentences), "thresholds": {}}
for tau in THRESHOLDS:
    summary_out["thresholds"][str(tau)] = {
        "patches_per_sentence": stats(summary[tau]["patch_counts"]),
        "avg_patch_size": stats(summary[tau]["avg_patch_sizes"]),
        "total_patches": sum(summary[tau]["patch_counts"]),
    }

# --- Boundary evaluation on a small sample ---------------------------------

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


def get_patch_boundaries(seq, entropies, theta):
    """Return set of byte-offset positions where a patch boundary occurs."""
    boundaries = set()
    for i, h in enumerate(entropies):
        if h > theta:
            boundaries.add(i)
    return boundaries


def _count_matched(query_sorted, ref_sorted, tolerance):
    import bisect
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


# If we have enough sentences, evaluate on the first EVAL_SAMPLE_SIZE
eval_sample = sentences[:EVAL_SAMPLE_SIZE]
if eval_sample:
    print(f"Evaluating boundary alignment on {len(eval_sample)} sentences...")
    eval_aggregate = {tau: {"precisions": [], "recalls": [], "f1s": []} for tau in THRESHOLDS}
    for idx, sent in enumerate(eval_sample):
        seq = sent.encode("utf-8")
        entropies = compute_entropy(seq)
        gold = get_morpheme_boundaries(sent)

        for tau in THRESHOLDS:
            pred = get_patch_boundaries(seq, entropies, tau)
            p, r, f1 = boundaries_match(pred, gold)
            eval_aggregate[tau]["precisions"].append(p)
            eval_aggregate[tau]["recalls"].append(r)
            eval_aggregate[tau]["f1s"].append(f1)

    def mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    for tau in THRESHOLDS:
        avg_p = mean(eval_aggregate[tau]["precisions"])
        avg_r = mean(eval_aggregate[tau]["recalls"])
        avg_f1 = mean(eval_aggregate[tau]["f1s"])
        summary_out["thresholds"][str(tau)]["boundary_precision"] = round(avg_p, 4)
        summary_out["thresholds"][str(tau)]["boundary_recall"] = round(avg_r, 4)
        summary_out["thresholds"][str(tau)]["boundary_f1"] = round(avg_f1, 4)

with open("sweep_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

# ── Print summary table ──────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("ENTROPY THRESHOLD SWEEP RESULTS")
print(f"Corpus: {len(sentences)} sentences")
print("=" * 72)
print(f"{'tau':>6} | {'Total Patches':>14} | {'Mean/Sent':>10} | {'Median/Sent':>12} | {'Avg Patch Size':>15} | {'Bdy F1':>7}")
print("-" * 72)
for tau in THRESHOLDS:
    s = summary_out["thresholds"][str(tau)]
    ps = s["patches_per_sentence"]
    total = s["total_patches"]
    avg_sz = s["avg_patch_size"]["mean"]
    bdf1 = s.get("boundary_f1", None)
    bdf1_str = f"{bdf1:.4f}" if bdf1 is not None else "-"
    print(f"{tau:>6.1f} | {total:>14,} | {ps['mean']:>10.2f} | {ps['median']:>12.2f} | {avg_sz:>15.2f} | {bdf1_str:>7}")
print("=" * 72)
print(f"\nOutputs: sweep_results.jsonl, sweep_results.csv, sweep_summary.json")
