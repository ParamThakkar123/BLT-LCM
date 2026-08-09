"""
Entropy threshold sweep script.

Sweeps τ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} over the full 50K Marathi sentence
corpus, recording per-sentence patch counts at each threshold.

Outputs:
  - sweep_results.jsonl   : one JSON object per sentence with patch counts
  - sweep_summary.json    : aggregate statistics per threshold

Resumable: per-sentence rows stream to ``sweep_results.jsonl`` as they are
computed, so an interrupted sweep restarts at the first sentence it had not
written. Pass ``--resume never`` (or set ``SWEEP_RESUME=never``) to force a
clean run.
"""

import argparse
import json
import math
import csv
import os
import sys
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_scripts.checkpoint_utils import (
    ResumableJsonl,
    StageTracker,
    config_fingerprint,
)

# For morpheme boundary evaluation on a small sample
from lcm_scripts.indic_resources import configure_indic_resources
from indicnlp import loader

configure_indic_resources()
loader.load()
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

# Initialize morph analyzer for Marathi
morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── CLI ──────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--resume",
    default=os.environ.get("SWEEP_RESUME", "auto"),
    metavar="auto|never",
    help="'auto' (default) continues from a partial sweep_results.jsonl whose "
    "configuration matches; 'never' discards it and starts over.",
)
_parser.add_argument("--corpus", default="marathi_sentences.json")
_parser.add_argument("--out_jsonl", default="sweep_results.jsonl")
_parser.add_argument("--out_csv", default="sweep_results.csv")
_parser.add_argument("--out_summary", default="sweep_summary.json")
ARGS = _parser.parse_args()

# ── Load corpus ──────────────────────────────────────────────────────────────

with open(ARGS.corpus, "r", encoding="utf-8") as f:
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

FINGERPRINT = config_fingerprint(
    {
        "corpus": ARGS.corpus,
        "num_sentences": len(sentences),
        "thresholds": THRESHOLDS,
        "tolerance": TOLERANCE,
        "eval_sample_size": EVAL_SAMPLE_SIZE,
    }
)

# Rows stream to disk as they are produced, so an interrupted sweep resumes at
# the first sentence it had not written rather than rescanning the corpus.
writer = ResumableJsonl(
    ARGS.out_jsonl,
    fingerprint=FINGERPRINT,
    resume=ARGS.resume != "never",
    key="sentence_id",
)
if writer.done:
    print(f"Resuming sweep: {len(writer.done)}/{len(sentences)} sentences already done")

for idx, seq in enumerate(byte_sequences):
    if writer.is_done(idx):
        continue
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

    writer.append(row)

    if (idx + 1) % 2000 == 0:
        print(f"  Processed {idx + 1}/{len(sentences)} sentences...")

writer.close()
results = writer.all_records()

# Summary is derived from the written rows rather than accumulated in memory, so
# a run stitched together from several attempts aggregates over everything.
summary = {tau: {"patch_counts": [], "avg_patch_sizes": []} for tau in THRESHOLDS}
for row in results:
    for tau in THRESHOLDS:
        key = f"tau_{tau}"
        summary[tau]["patch_counts"].append(row[key + "_num_patches"])
        summary[tau]["avg_patch_sizes"].append(row[key + "_avg_patch_size"])

# ── Write per-sentence CSV (easy to inspect) ─────────────────────────────────

csv_cols = ["sentence_id", "byte_length"]
for tau in THRESHOLDS:
    csv_cols.append(f"tau_{tau}_num_patches")
    csv_cols.append(f"tau_{tau}_avg_patch_size")

with open(ARGS.out_csv, "w", newline="", encoding="utf-8") as f:
    csv_writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
    csv_writer.writeheader()
    csv_writer.writerows(results)

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

    def _boundary_eval():
        aggregate = {
            tau: {"precisions": [], "recalls": [], "f1s": []} for tau in THRESHOLDS
        }
        for sent in eval_sample:
            seq = sent.encode("utf-8")
            entropies = compute_entropy(seq)
            gold = get_morpheme_boundaries(sent)

            for tau in THRESHOLDS:
                pred = get_patch_boundaries(seq, entropies, tau)
                p, r, f1 = boundaries_match(pred, gold)
                aggregate[tau]["precisions"].append(p)
                aggregate[tau]["recalls"].append(r)
                aggregate[tau]["f1s"].append(f1)

        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        return {
            str(tau): {
                "boundary_precision": round(mean(aggregate[tau]["precisions"]), 4),
                "boundary_recall": round(mean(aggregate[tau]["recalls"]), 4),
                "boundary_f1": round(mean(aggregate[tau]["f1s"]), 4),
            }
            for tau in THRESHOLDS
        }

    # Morphological analysis is much slower per sentence than the byte scan, so
    # this stage is memoized separately from the sweep itself.
    boundary_stats = StageTracker(
        os.path.splitext(ARGS.out_summary)[0] + ".state.json",
        fingerprint=FINGERPRINT,
        resume=ARGS.resume != "never",
    ).run("boundary_eval", _boundary_eval)

    for tau in THRESHOLDS:
        summary_out["thresholds"][str(tau)].update(boundary_stats[str(tau)])

with open(ARGS.out_summary, "w", encoding="utf-8") as f:
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
