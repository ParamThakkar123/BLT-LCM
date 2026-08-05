"""Run evaluations (noisy corruption tests, multiple seeds) and log results.

Produces CSV results and supports chrF++, BLEU, METEOR, TER via eval_metrics.

Two corruption modes (controlled by ``--corrupt_input``):
  - input corruption (default): noise applied to hypotheses before scoring,
    matching the LCM evaluation convention where model inputs are perturbed.
  - output corruption: noise applied to hypotheses after generation,
    testing output-channel robustness.
"""

import argparse
import csv
import random
import os
from typing import List, Optional

from bhashasetu_utils import add_character_noise
from eval_metrics import compute_all


def run_eval(
    hyps: List[str],
    refs: List[str],
    out_csv: str,
    seeds: List[int],
    noisy_probs: List[float],
    corrupt_input: bool = True,
    comet_model_name: Optional[str] = None,
):
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        for p in noisy_probs:
            if corrupt_input:
                refs_noisy = [
                    add_character_noise(r, p, seed=rng.randint(0, 2**31)) for r in refs
                ]
                hyps_noisy = hyps[:]
            else:
                hyps_noisy = [
                    add_character_noise(h, p, seed=rng.randint(0, 2**31)) for h in hyps
                ]
                refs_noisy = refs[:]
            metrics = compute_all(
                hyps_noisy, refs_noisy, comet_model_name=comet_model_name
            )
            row = {"seed": seed, "noise_prob": p, "corrupt_input": corrupt_input}
            row.update(metrics)
            rows.append(row)

    keys = [
        "seed",
        "noise_prob",
        "corrupt_input",
        "BLEU",
        "chrF++",
        "TER",
        "METEOR",
        "COMET",
    ]
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
    parser.add_argument(
        "--corrupt_input",
        action="store_true",
        default=True,
        help="Corrupt references (inputs) instead of hypotheses (outputs). "
        "Matches LCM evaluation convention.",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default=None,
        help="Optional COMET model name or checkpoint path",
    )
    args = parser.parse_args()

    with open(args.hyp_file, encoding="utf-8") as f:
        hyps = [l.strip() for l in f if l.strip()]
    with open(args.ref_file, encoding="utf-8") as f:
        refs = [l.strip() for l in f if l.strip()]

    run_eval(
        hyps,
        refs,
        args.out_csv,
        args.seeds,
        args.noisy_probs,
        corrupt_input=args.corrupt_input,
        comet_model_name=args.comet_model,
    )
