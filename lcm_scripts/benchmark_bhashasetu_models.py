"""Run BhashaSetu subset/noise benchmarks for all requested baselines.

This orchestrator launches the per-model training scripts for 25%, 50% and 80%
of BhashaSetu and collects their BLEU, chrF++ and TER CSVs for clean, 10% noise
and 20% noise evaluations.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from bhashasetu_utils import DEFAULT_FRACTIONS, DEFAULT_NOISE_LEVELS


SCRIPT_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description="Benchmark BPE Transformer, BPE Llama-8B and SONAR-LCM on BhashaSetu")
    p.add_argument("--models", nargs="+", default=["bpe_transformer", "bpe_llama8b", "sonar_lcm"], choices=["bpe_transformer", "bpe_llama8b", "sonar_lcm"])
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    p.add_argument("--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    p.add_argument("--out_dir", default="runs/bhashasetu_benchmarks")
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--eval_examples", type=int, default=1000)
    p.add_argument("--num_docs", type=int, default=500)
    p.add_argument("--eval_docs", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--src_col", default=None)
    p.add_argument("--tgt_col", default=None)
    p.add_argument("--llama_model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--llama_qlora", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    all_rows: list[dict[str, str]] = []
    for frac in args.fractions:
        frac_tag = str(frac).replace(".", "p")
        common_noise = [str(x) for x in args.noise_levels]
        data_cols = []
        if args.src_col:
            data_cols += ["--src_col", args.src_col]
        if args.tgt_col:
            data_cols += ["--tgt_col", args.tgt_col]
        if args.max_examples:
            data_cols += ["--max_examples", str(args.max_examples)]
        device_cols = ["--device", args.device] if args.device else []

        if "bpe_transformer" in args.models:
            run_dir = out_dir / f"bpe_transformer_{frac_tag}"
            cmd = [py, str(SCRIPT_DIR / "train_bpe_transformer.py"), "--fraction", str(frac), "--epochs", str(args.epochs), "--eval_examples", str(args.eval_examples), "--out_dir", str(run_dir), "--noise_levels", *common_noise, *data_cols, *device_cols]
            run(cmd, args.dry_run)
            all_rows.extend(read_metrics(run_dir / f"metrics_fraction{frac}.csv"))

        if "bpe_llama8b" in args.models:
            run_dir = out_dir / f"bpe_llama8b_{frac_tag}"
            cmd = [py, str(SCRIPT_DIR / "train_bpe_llama8b.py"), "--model_name", args.llama_model_name, "--fraction", str(frac), "--epochs", str(args.epochs), "--eval_examples", str(args.eval_examples), "--out_dir", str(run_dir), "--noise_levels", *common_noise, *data_cols]
            if args.llama_qlora:
                cmd.append("--qlora")
            run(cmd, args.dry_run)
            all_rows.extend(read_metrics(run_dir / f"metrics_fraction{frac}.csv"))

        if "sonar_lcm" in args.models:
            run_dir = out_dir / f"sonar_lcm_{frac_tag}"
            cmd = [py, str(SCRIPT_DIR / "train_lcm_sonar.py"), "--fraction", str(frac), "--epochs", str(args.epochs), "--num_docs", str(args.num_docs), "--eval_docs", str(args.eval_docs), "--out_dir", str(run_dir), "--noise_levels", *common_noise, *device_cols]
            run(cmd, args.dry_run)
            all_rows.extend(read_metrics(run_dir / f"metrics_fraction{frac}.csv"))

    if all_rows:
        summary = out_dir / "summary_metrics.csv"
        fieldnames = ["model", "fraction", "noise", "BLEU", "chrF++", "TER"]
        with open(summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(all_rows)
        print(f"Wrote summary to {summary}")
    elif args.dry_run:
        print("Dry run complete; no metrics were collected.")
    else:
        print("No metrics found. Check per-model logs for failures.")


if __name__ == "__main__":
    main()
