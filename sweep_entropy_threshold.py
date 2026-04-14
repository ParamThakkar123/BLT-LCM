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

with open("sweep_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

# ── Print summary table ──────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("ENTROPY THRESHOLD SWEEP RESULTS")
print(f"Corpus: {len(sentences)} sentences")
print("=" * 72)
print(f"{'tau':>6} | {'Total Patches':>14} | {'Mean/Sent':>10} | {'Median/Sent':>12} | {'Avg Patch Size':>15}")
print("-" * 72)
for tau in THRESHOLDS:
    s = summary_out["thresholds"][str(tau)]
    ps = s["patches_per_sentence"]
    total = s["total_patches"]
    avg_sz = s["avg_patch_size"]["mean"]
    print(f"{tau:>6.1f} | {total:>14,} | {ps['mean']:>10.2f} | {ps['median']:>12.2f} | {avg_sz:>15.2f}")
print("=" * 72)
print(f"\nOutputs: sweep_results.jsonl, sweep_results.csv, sweep_summary.json")
