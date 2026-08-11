"""Run full MT metric suite for BLT-LCM checkpoints.

Expected workflow:
1) Generate one hypothesis file per checkpoint (one sentence per line).
2) Run this script to compute BLEU, chrF++, METEOR, COMET, and TER.

By default, checkpoint -> hypothesis mapping is:
  lcm_models/lcm_blt_epoch10.pth -> <hyp_dir>/lcm_blt_epoch10.hyp.txt

Example:
  python lcm_scripts/run_metric_suite.py \
    --checkpoints_dir lcm_models \
    --hyp_dir outputs/checkpoint_hyps \
    --ref_file outputs/ref.txt \
    --out_csv results/blt_lcm_50_metrics.csv \
    --comet_model Unbabel/wmt22-comet-da
"""

import argparse
import csv
import glob
import os
import re
import subprocess
from typing import Dict, List, Optional

from eval_metrics import compute_all
from device_utils import report_device
from checkpoint_utils import StageTracker, add_resume_args, config_fingerprint


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def epoch_key(path: str):
    name = os.path.basename(path)
    m = re.search(r"epoch(\d+)", name)
    if m:
        return (0, int(m.group(1)))
    if "best" in name:
        return (1, 10**9)
    return (2, name)


def checkpoint_to_hyp_path(ckpt_path: str, hyp_dir: str, hyp_suffix: str) -> str:
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(hyp_dir, f"{stem}{hyp_suffix}")


def load_parallel(hyp_path: str, refs: List[str]) -> Optional[List[str]]:
    if not os.path.exists(hyp_path):
        return None
    hyps = read_lines(hyp_path)
    if len(hyps) != len(refs):
        raise ValueError(
            f"Line mismatch for {hyp_path}: {len(hyps)} hyps vs {len(refs)} refs"
        )
    return hyps


def evaluate_checkpoint(
    ckpt_path: str,
    hyp_path: str,
    refs: List[str],
    comet_model: Optional[str],
) -> Dict[str, float]:
    hyps = load_parallel(hyp_path, refs)
    if hyps is None:
        raise FileNotFoundError(f"Missing hypothesis file: {hyp_path}")

    metrics = compute_all(hyps, refs, comet_model_name=comet_model)
    row = {
        "checkpoint": os.path.basename(ckpt_path),
        "hyp_file": os.path.basename(hyp_path),
        "num_sentences": len(hyps),
        "BLEU": metrics.get("BLEU", float("nan")),
        "chrF++": metrics.get("chrF++", float("nan")),
        "METEOR": metrics.get("METEOR", float("nan")),
        "COMET": metrics.get("COMET", float("nan")),
        "TER": metrics.get("TER", float("nan")),
    }
    return row


def maybe_generate_hypothesis(
    ckpt_path: str,
    hyp_path: str,
    generate_cmd: Optional[str],
    force_regen: bool,
) -> None:
    if os.path.exists(hyp_path) and not force_regen:
        return

    if not generate_cmd:
        return

    os.makedirs(os.path.dirname(hyp_path) or ".", exist_ok=True)
    cmd = generate_cmd.format(checkpoint=ckpt_path, hyp_file=hyp_path)
    print(f"Generating hypotheses for {os.path.basename(ckpt_path)}")
    print(f"  cmd: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="lcm_models")
    parser.add_argument("--checkpoint_glob", type=str, default="lcm_blt_*.pth")
    parser.add_argument("--hyp_dir", type=str, required=True)
    parser.add_argument("--hyp_suffix", type=str, default=".hyp.txt")
    parser.add_argument("--ref_file", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="results/blt_lcm_metrics.csv")
    parser.add_argument(
        "--generate_cmd",
        type=str,
        default=None,
        help=(
            "Optional command template to generate hypothesis files per checkpoint. "
            "Use placeholders {checkpoint} and {hyp_file}. "
            "Example: 'python my_infer.py --ckpt {checkpoint} --out {hyp_file}'"
        ),
    )
    parser.add_argument(
        "--force_regen",
        action="store_true",
        help="Regenerate hypothesis files even if they already exist",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default=None,
        help="Optional COMET model name/checkpoint (e.g., Unbabel/wmt22-comet-da)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any checkpoint is missing a matching hypothesis file",
    )
    add_resume_args(parser, training=False)
    args = parser.parse_args()
    # BLEU/chrF++/TER/METEOR are CPU string metrics; COMET runs a model here.
    report_device(label="metrics", warn_cpu=False)

    fingerprint = config_fingerprint(args, extra={"stage": "run_metric_suite"})
    refs = read_lines(args.ref_file)

    pattern = os.path.join(args.checkpoints_dir, args.checkpoint_glob)
    checkpoints = sorted(glob.glob(pattern), key=epoch_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")

    rows = []
    skipped = []
    # COMET alone is a neural forward pass over the whole hypothesis set, per
    # checkpoint. Memoize each checkpoint's row so an interrupted suite resumes
    # at the first checkpoint it had not finished scoring.
    stages = StageTracker(
        os.path.splitext(args.out_csv)[0] + ".state.json",
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    for ckpt in checkpoints:
        hyp_path = checkpoint_to_hyp_path(ckpt, args.hyp_dir, args.hyp_suffix)

        try:
            maybe_generate_hypothesis(
                ckpt,
                hyp_path,
                generate_cmd=args.generate_cmd,
                force_regen=args.force_regen,
            )
        except Exception as e:
            if args.strict:
                raise RuntimeError(
                    f"Failed to generate hypotheses for {ckpt}: {e}"
                ) from e
            skipped.append((ckpt, hyp_path))
            print(f"Generation failed for {os.path.basename(ckpt)}: {e}")
            continue

        if not os.path.exists(hyp_path):
            if args.strict:
                raise FileNotFoundError(
                    f"Missing hypothesis for checkpoint {ckpt}: {hyp_path}"
                )
            skipped.append((ckpt, hyp_path))
            continue

        row = stages.run(
            os.path.basename(ckpt),
            lambda: evaluate_checkpoint(ckpt, hyp_path, refs, args.comet_model),
        )
        rows.append(row)
        print(
            f"{row['checkpoint']}: chrF++={row['chrF++']:.2f} BLEU={row['BLEU']:.2f} "
            f"METEOR={row['METEOR']:.2f} COMET={row['COMET']:.4f} TER={row['TER']:.2f}"
        )

    if not rows:
        raise RuntimeError(
            "No checkpoints evaluated. Ensure hypothesis files exist in hyp_dir "
            "with the expected naming pattern."
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "checkpoint",
        "hyp_file",
        "num_sentences",
        "BLEU",
        "chrF++",
        "METEOR",
        "COMET",
        "TER",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote metric suite results to: {args.out_csv}")
    if skipped:
        print("Skipped checkpoints without matching hypothesis files:")
        for ckpt, hyp_path in skipped:
            print(f"  - {os.path.basename(ckpt)} -> {hyp_path}")


if __name__ == "__main__":
    main()
