"""
Generate research paper results from BLT entropy-based patching output.

Reads blt_marathi_patched.jsonl and produces figures, tables, and stats
in a results/ folder.

Usage:
    python generate_results.py
    python generate_results.py --input my_output.jsonl --results_dir my_results
"""

import argparse
import csv
import json
import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


def load_results(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_statistics(results):
    num_sentences = len(results)
    all_num_bytes = [r["num_bytes"] for r in results]
    all_num_patches = [r["num_patches"] for r in results]
    all_avg_patch_len = [r["avg_patch_length"] for r in results]

    all_entropies = []
    for r in results:
        all_entropies.extend(r["entropy_per_byte"])

    all_patch_lengths = []
    all_boundary_entropies = []
    all_mean_patch_entropies = []
    for r in results:
        all_patch_lengths.extend(r["patch_lengths"])
        for p in r["patches"]:
            all_boundary_entropies.append(p["entropy_at_boundary"])
            all_mean_patch_entropies.append(p["mean_entropy_in_patch"])

    ent = np.array(all_entropies)
    pl = np.array(all_patch_lengths)
    be = np.array(all_boundary_entropies)

    stats = {
        "dataset": "ParamTh/BhashaSetu (Marathi)",
        "num_sentences": num_sentences,
        "threshold": results[0]["threshold_used"],
        "total_bytes": sum(all_num_bytes),
        "total_patches": sum(all_num_patches),
        "bytes_per_sentence": {
            "mean": round(np.mean(all_num_bytes), 2),
            "median": round(float(np.median(all_num_bytes)), 2),
            "std": round(np.std(all_num_bytes), 2),
            "min": int(np.min(all_num_bytes)),
            "max": int(np.max(all_num_bytes)),
        },
        "patches_per_sentence": {
            "mean": round(np.mean(all_num_patches), 2),
            "median": round(float(np.median(all_num_patches)), 2),
            "std": round(np.std(all_num_patches), 2),
            "min": int(np.min(all_num_patches)),
            "max": int(np.max(all_num_patches)),
        },
        "patch_length_bytes": {
            "mean": round(float(np.mean(pl)), 2),
            "median": round(float(np.median(pl)), 2),
            "std": round(float(np.std(pl)), 2),
            "min": int(np.min(pl)),
            "max": int(np.max(pl)),
        },
        "entropy_per_byte": {
            "mean": round(float(np.mean(ent)), 4),
            "median": round(float(np.median(ent)), 4),
            "std": round(float(np.std(ent)), 4),
            "min": round(float(np.min(ent)), 4),
            "max": round(float(np.max(ent)), 4),
        },
        "entropy_at_boundaries": {
            "mean": round(float(np.mean(be)), 4),
            "median": round(float(np.median(be)), 4),
            "std": round(float(np.std(be)), 4),
        },
        "compression_ratio": round(sum(all_num_bytes) / sum(all_num_patches), 2),
    }
    return stats


def save_per_sentence_csv(results, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sentence_index", "num_bytes", "num_patches",
            "avg_patch_length", "mean_entropy", "min_entropy",
            "max_entropy", "text_preview",
        ])
        for r in results:
            ent = r["entropy_per_byte"]
            writer.writerow([
                r["sentence_index"],
                r["num_bytes"],
                r["num_patches"],
                r["avg_patch_length"],
                round(np.mean(ent), 4),
                round(min(ent), 4),
                round(max(ent), 4),
                r["marathi_text"][:80],
            ])


def save_example_patches_csv(results, path, num_examples=10):
    """Save a few example sentences with their patches for a paper table."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sentence_index", "marathi_text", "num_bytes", "num_patches",
            "avg_patch_length", "patch_index", "patch_text",
            "byte_start", "byte_end", "length_bytes",
            "entropy_at_boundary", "mean_entropy_in_patch",
        ])
        for r in results[:num_examples]:
            for p in r["patches"]:
                writer.writerow([
                    r["sentence_index"],
                    r["marathi_text"][:100],
                    r["num_bytes"],
                    r["num_patches"],
                    r["avg_patch_length"],
                    p["patch_index"],
                    p["patch_text"],
                    p["byte_start"],
                    p["byte_end"],
                    p["length_bytes"],
                    p["entropy_at_boundary"],
                    p["mean_entropy_in_patch"],
                ])


# ============================================================
# Figures
# ============================================================

def fig_entropy_distribution(results, path):
    all_ent = []
    for r in results:
        all_ent.extend(r["entropy_per_byte"])
    all_ent = np.array(all_ent)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_ent, bins=100, color="#4C72B0", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(results[0]["threshold_used"], color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {results[0]['threshold_used']:.3f}")
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Per-Byte Entropy (Marathi)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_patch_length_distribution(results, path):
    all_pl = []
    for r in results:
        all_pl.extend(r["patch_lengths"])
    all_pl = np.array(all_pl)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_len = min(int(np.percentile(all_pl, 99)), 100)
    ax.hist(all_pl[all_pl <= max_len], bins=range(1, max_len + 2),
            color="#55A868", edgecolor="white", linewidth=0.3, density=True)
    ax.set_xlabel("Patch Length (bytes)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Patch Lengths (Marathi)")
    ax.set_xlim(0, max_len + 1)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_patches_per_sentence(results, path):
    counts = [r["num_patches"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=80, color="#C44E52", edgecolor="white", linewidth=0.3, density=True)
    ax.axvline(np.mean(counts), color="black", linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(counts):.1f}")
    ax.set_xlabel("Number of Patches per Sentence")
    ax.set_ylabel("Density")
    ax.set_title("Patches per Sentence Distribution (Marathi)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_entropy_vs_patch_length(results, path):
    mean_ents = []
    lengths = []
    for r in results:
        for p in r["patches"]:
            mean_ents.append(p["mean_entropy_in_patch"])
            lengths.append(p["length_bytes"])

    # Subsample if too many points
    me = np.array(mean_ents)
    le = np.array(lengths)
    if len(me) > 50000:
        idx = np.random.default_rng(42).choice(len(me), 50000, replace=False)
        me, le = me[idx], le[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(me, le, alpha=0.08, s=6, color="#8172B2")
    ax.set_xlabel("Mean Entropy in Patch (nats)")
    ax.set_ylabel("Patch Length (bytes)")
    ax.set_title("Mean Patch Entropy vs. Patch Length (Marathi)")
    ax.set_ylim(0, min(np.percentile(lengths, 99.5), 150))
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_example_entropy_profile(results, path, sentence_idx=0):
    """Plot byte-level entropy for one sentence with patch boundaries marked."""
    r = results[sentence_idx]
    ent = np.array(r["entropy_per_byte"])
    boundaries = r["patch_boundaries"]
    threshold = r["threshold_used"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ent, color="#4C72B0", linewidth=0.8, label="Byte entropy")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"Threshold = {threshold:.3f}")

    for b in boundaries:
        ax.axvline(b, color="green", alpha=0.35, linewidth=0.6)

    ax.set_xlabel("Byte Position")
    ax.set_ylabel("Entropy (nats)")
    text_preview = r["marathi_text"][:50].replace("\n", " ")
    ax.set_title(f"Entropy Profile with Patch Boundaries\n\"{text_preview}...\"")
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_avg_patch_length_vs_sentence_length(results, path):
    num_bytes = [r["num_bytes"] for r in results]
    avg_pl = [r["avg_patch_length"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(num_bytes, avg_pl, alpha=0.1, s=6, color="#DD8452")
    ax.set_xlabel("Sentence Length (bytes)")
    ax.set_ylabel("Average Patch Length (bytes)")
    ax.set_title("Avg Patch Length vs. Sentence Length (Marathi)")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_cumulative_entropy(results, path):
    all_ent = []
    for r in results:
        all_ent.extend(r["entropy_per_byte"])
    all_ent = np.sort(all_ent)
    cdf = np.arange(1, len(all_ent) + 1) / len(all_ent)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Subsample for plotting if very large
    step = max(1, len(all_ent) // 10000)
    ax.plot(all_ent[::step], cdf[::step], color="#4C72B0", linewidth=1.2)
    threshold = results[0]["threshold_used"]
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.3f}")
    frac_above = np.mean(np.array(all_ent) > threshold)
    ax.axhline(1 - frac_above, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Cumulative Fraction of Bytes")
    ax.set_title(f"CDF of Per-Byte Entropy ({frac_above*100:.1f}% above threshold)")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper results from BLT patching output")
    parser.add_argument("--input", type=str, default="blt_marathi_patched.jsonl",
                        help="Path to JSONL output from run_blt_patching.py")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--num_example_sentences", type=int, default=10,
                        help="Number of example sentences for the patches table")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results):,} sentences\n")

    # 1. Summary statistics JSON
    print("Computing statistics...")
    stats = compute_statistics(results)
    stats_path = os.path.join(args.results_dir, "summary_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved {stats_path}")

    # Print key stats to console
    print(f"\n  Sentences          : {stats['num_sentences']:,}")
    print(f"  Total bytes        : {stats['total_bytes']:,}")
    print(f"  Total patches      : {stats['total_patches']:,}")
    print(f"  Compression ratio  : {stats['compression_ratio']:.2f}x")
    print(f"  Avg patch length   : {stats['patch_length_bytes']['mean']:.2f} bytes")
    print(f"  Mean byte entropy  : {stats['entropy_per_byte']['mean']:.4f} nats")
    print(f"  Threshold          : {stats['threshold']}")
    print()

    # 2. Per-sentence stats CSV
    csv_path = os.path.join(args.results_dir, "per_sentence_stats.csv")
    save_per_sentence_csv(results, csv_path)
    print(f"  Saved {csv_path}")

    # 3. Example patches table CSV
    ex_path = os.path.join(args.results_dir, "example_patches_table.csv")
    save_example_patches_csv(results, ex_path, num_examples=args.num_example_sentences)
    print(f"  Saved {ex_path}")

    # 4. Figures
    print("\nGenerating figures...")
    fig_entropy_distribution(results, os.path.join(args.results_dir, "fig_entropy_distribution.png"))
    fig_patch_length_distribution(results, os.path.join(args.results_dir, "fig_patch_length_distribution.png"))
    fig_patches_per_sentence(results, os.path.join(args.results_dir, "fig_patches_per_sentence.png"))
    fig_entropy_vs_patch_length(results, os.path.join(args.results_dir, "fig_entropy_vs_patch_length.png"))
    fig_avg_patch_length_vs_sentence_length(results, os.path.join(args.results_dir, "fig_avg_patch_length_vs_sentence_length.png"))
    fig_cumulative_entropy(results, os.path.join(args.results_dir, "fig_cumulative_entropy_cdf.png"))

    # Entropy profile for a few example sentences
    for i in [0, 1, 2]:
        if i < len(results):
            fig_example_entropy_profile(
                results,
                os.path.join(args.results_dir, f"fig_entropy_profile_sent{i}.png"),
                sentence_idx=i,
            )

    print(f"\nAll results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
