"""Run evaluations (noisy corruption tests, multiple seeds) and log results.

Produces CSV results and supports chrF++, BLEU, METEOR, TER via eval_metrics.
"""

import argparse
import csv
import random
import os
from typing import List

from eval_metrics import compute_all


def corrupt_text(s: str, prob: float) -> str:
    # corrupt by replacing a character with random ASCII character with probability prob
    import random

    out = []
    for ch in s:
        if ch.isspace():
            out.append(ch)
            continue
        if random.random() < prob:
            out.append(chr(random.randint(32, 126)))
        else:
            out.append(ch)
    return "".join(out)


def run_eval(
    hyps: List[str],
    refs: List[str],
    out_csv: str,
    seeds: List[int],
    noisy_probs: List[float],
):
    rows = []
    for seed in seeds:
        random.seed(seed)
        for p in noisy_probs:
            hyps_noisy = [corrupt_text(h, p) for h in hyps]
            refs_noisy = refs  # don't corrupt references
            metrics = compute_all(hyps_noisy, refs_noisy)
            row = {"seed": seed, "noise_prob": p}
            row.update(metrics)
            rows.append(row)

    # write CSV
    keys = ["seed", "noise_prob", "BLEU", "chrF++", "TER", "METEOR", "COMET"]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_file", required=True)
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--out_csv", default="results/eval_results.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--noisy_probs", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    args = parser.parse_args()

    with open(args.hyp_file, encoding="utf-8") as f:
        hyps = [l.strip() for l in f if l.strip()]
    with open(args.ref_file, encoding="utf-8") as f:
        refs = [l.strip() for l in f if l.strip()]

    run_eval(hyps, refs, args.out_csv, args.seeds, args.noisy_probs)
