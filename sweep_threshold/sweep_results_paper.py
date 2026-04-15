"""
Generate research-paper-ready results from the entropy threshold sweep.

Reads sweep_summary.json and sweep_results.jsonl (already produced by
sweep_entropy_threshold.py) and prints formatted tables + analysis.
Does NOT modify any existing files.
"""

import json
import math

# ── Load data ────────────────────────────────────────────────────────────────

with open("sweep_summary.json", "r", encoding="utf-8") as f:
    summary = json.load(f)

with open("sweep_results.jsonl", "r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

N = summary["num_sentences"]
taus = ["0.5", "1.0", "1.5", "2.0", "2.5"]

# ── Corpus overview ──────────────────────────────────────────────────────────

byte_lens = [r["byte_length"] for r in rows]
total_bytes = sum(byte_lens)
mean_bl = total_bytes / N
sorted_bl = sorted(byte_lens)
median_bl = sorted_bl[N // 2]
std_bl = math.sqrt(sum((x - mean_bl) ** 2 for x in byte_lens) / N)

print("=" * 80)
print("ENTROPY THRESHOLD SWEEP -- RESULTS FOR RESEARCH PAPER")
print("=" * 80)

print("\n1. CORPUS OVERVIEW")
print("-" * 80)
print(f"  Dataset          : ParamTh/BhashaSetu (Marathi)")
print(f"  Sentences         : {N:,}")
print(f"  Total bytes       : {total_bytes:,}")
print(f"  Bytes/sentence    : mean={mean_bl:.2f}, median={median_bl}, std={std_bl:.2f}")
print(f"                      min={min(byte_lens)}, max={max(byte_lens)}")

# ── Table 1: Main sweep results ─────────────────────────────────────────────

print("\n\n2. TABLE 1: Entropy Threshold Sweep Results")
print("-" * 80)
print(f"{'tau':>5} | {'Total':>12} | {'Mean':>8} | {'Median':>8} | {'Std':>8} | {'Min':>6} | {'Max':>6} | {'Avg Patch':>10}")
print(f"{'':>5} | {'Patches':>12} | {'/sent':>8} | {'/sent':>8} | {'':>8} | {'':>6} | {'':>6} | {'Size (B)':>10}")
print("-" * 80)
for tau in taus:
    d = summary["thresholds"][tau]
    ps = d["patches_per_sentence"]
    aps = d["avg_patch_size"]
    print(
        f"{tau:>5} | "
        f"{d['total_patches']:>12,} | "
        f"{ps['mean']:>8.2f} | "
        f"{ps['median']:>8.1f} | "
        f"{ps['std']:>8.2f} | "
        f"{ps['min']:>6} | "
        f"{ps['max']:>6} | "
        f"{aps['mean']:>10.2f}"
    )
print("-" * 80)

# ── Table 2: Compression ratio & sequence reduction ─────────────────────────

print("\n\n3. TABLE 2: Compression Ratio and Sequence Length Reduction")
print("-" * 80)
print(f"{'tau':>5} | {'Compression':>14} | {'Seq. Reduction':>16} | {'Patches Saved':>16}")
print(f"{'':>5} | {'Ratio':>14} | {'(%)':>16} | {'vs tau=0.5':>16}")
print("-" * 80)
for tau in taus:
    d = summary["thresholds"][tau]
    total_patches = d["total_patches"]
    compression = total_bytes / total_patches
    reduction = (1 - total_patches / total_bytes) * 100
    saved_vs_05 = summary["thresholds"]["0.5"]["total_patches"] - total_patches
    print(
        f"{tau:>5} | "
        f"{compression:>13.2f}x | "
        f"{reduction:>15.1f}% | "
        f"{saved_vs_05:>16,}"
    )
print("-" * 80)

# ── Table 3: Quartile distribution ──────────────────────────────────────────

print("\n\n4. TABLE 3: Patch Count Distribution (Quartiles)")
print("-" * 80)
print(f"{'tau':>5} | {'Q1':>8} | {'Median':>8} | {'Q3':>8} | {'IQR':>8} | {'P5':>8} | {'P95':>8}")
print("-" * 80)
for tau in taus:
    key = f"tau_{tau}_num_patches"
    vals = sorted([r[key] for r in rows])
    q1 = vals[N // 4]
    med = vals[N // 2]
    q3 = vals[3 * N // 4]
    p5 = vals[int(N * 0.05)]
    p95 = vals[int(N * 0.95)]
    print(
        f"{tau:>5} | "
        f"{q1:>8} | "
        f"{med:>8} | "
        f"{q3:>8} | "
        f"{q3 - q1:>8} | "
        f"{p5:>8} | "
        f"{p95:>8}"
    )
print("-" * 80)

# ── Table 4: Pairwise delta between consecutive thresholds ──────────────────

print("\n\n5. TABLE 4: Marginal Effect of Increasing tau")
print("-" * 80)
print(f"{'Transition':>14} | {'Delta Mean':>12} | {'Delta Total':>14} | {'% Change':>10}")
print(f"{'':>14} | {'Patches/Sent':>12} | {'Patches':>14} | {'':>10}")
print("-" * 80)
for i in range(len(taus) - 1):
    t1, t2 = taus[i], taus[i + 1]
    d1 = summary["thresholds"][t1]
    d2 = summary["thresholds"][t2]
    delta_mean = d2["patches_per_sentence"]["mean"] - d1["patches_per_sentence"]["mean"]
    delta_total = d2["total_patches"] - d1["total_patches"]
    pct = (delta_total / d1["total_patches"]) * 100
    print(
        f"{t1:>5} -> {t2:<5} | "
        f"{delta_mean:>+12.2f} | "
        f"{delta_total:>+14,} | "
        f"{pct:>+9.1f}%"
    )
print("-" * 80)

# ── Key findings ─────────────────────────────────────────────────────────────

# Compute the big drop
drop_05_10 = (
    (summary["thresholds"]["0.5"]["total_patches"] - summary["thresholds"]["1.0"]["total_patches"])
    / summary["thresholds"]["0.5"]["total_patches"]
) * 100

plateau_diff = (
    abs(summary["thresholds"]["1.5"]["total_patches"] - summary["thresholds"]["2.5"]["total_patches"])
    / summary["thresholds"]["1.5"]["total_patches"]
) * 100

print("\n\n6. KEY FINDINGS")
print("-" * 80)
print(f"  (a) Largest drop occurs between tau=0.5 and tau=1.0:")
print(f"      Patches per sentence decrease from {summary['thresholds']['0.5']['patches_per_sentence']['mean']:.2f}")
print(f"      to {summary['thresholds']['1.0']['patches_per_sentence']['mean']:.2f} ({drop_05_10:.1f}% reduction in total patches).")
print()
print(f"  (b) Diminishing returns beyond tau=1.5:")
print(f"      Between tau=1.5 and tau=2.5, total patches change by only {plateau_diff:.2f}%,")
print(f"      indicating a saturation point in Marathi byte-transition entropy.")
print()
print(f"  (c) Average patch size ranges from {summary['thresholds']['0.5']['avg_patch_size']['mean']:.2f} bytes (tau=0.5)")
print(f"      to {summary['thresholds']['2.5']['avg_patch_size']['mean']:.2f} bytes (tau=2.5), showing that most Marathi UTF-8")
print(f"      byte transitions have conditional entropy below 1.5 bits.")
print()
print(f"  (d) At tau=1.5 and above, mean patches/sentence stabilizes near ~305,")
print(f"      yielding a compression ratio of ~{total_bytes / summary['thresholds']['1.5']['total_patches']:.1f}x over raw bytes.")

# ── LaTeX table ──────────────────────────────────────────────────────────────

print("\n\n7. LATEX TABLE (copy-paste ready)")
print("-" * 80)
print(r"""\begin{table}[ht]
\centering
\caption{Effect of entropy threshold $\tau$ on byte-level patching for Marathi (50K sentences from BhashaSetu).}
\label{tab:entropy_sweep}
\begin{tabular}{c r r r r r}
\toprule
$\tau$ & Total Patches & Mean/Sent & Median/Sent & Std & Avg Patch Size (B) \\
\midrule""")
for tau in taus:
    d = summary["thresholds"][tau]
    ps = d["patches_per_sentence"]
    aps = d["avg_patch_size"]
    print(
        f"{tau} & "
        f"{d['total_patches']:,} & "
        f"{ps['mean']:.2f} & "
        f"{ps['median']:.1f} & "
        f"{ps['std']:.2f} & "
        f"{aps['mean']:.2f} \\\\"
    )
print(r"""\bottomrule
\end{tabular}
\end{table}""")

print("\n" + "=" * 80)
print("Results generated from sweep_summary.json and sweep_results.jsonl")
print("=" * 80)
