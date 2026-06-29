#!/usr/bin/env python3
"""
Extract hypothesis and reference files from an eval_blt CSV for metric_suite.sh.

Usage:
    python scripts/extract_hyps.py 0.25 lcm_blt_25
    python scripts/extract_hyps.py 0.50 lcm_blt_50
    python scripts/extract_hyps.py 0.80 lcm_blt_80

Reads:  results/blt_lcm_<frac_tag>_metrics.csv
Writes: outputs/<run_name>_hyps/lcm_blt_best.hyp.txt
        outputs/ref_<frac_tag>.txt
"""

import csv
import os
import sys

fraction = sys.argv[1] if len(sys.argv) > 1 else "0.25"
run_name = sys.argv[2] if len(sys.argv) > 2 else f"lcm_blt_{fraction.replace('.', '')}"

frac_tag = fraction.replace(".", "")          # "0.25" -> "025"
csv_path = f"results/blt_lcm_{frac_tag}_metrics.csv"
hyp_dir  = f"outputs/{run_name}_hyps"
hyp_path = f"{hyp_dir}/lcm_blt_best.hyp.txt"
ref_path = f"outputs/ref_{frac_tag}.txt"

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

os.makedirs(hyp_dir, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

with open(hyp_path, "w", encoding="utf-8") as f:
    f.write("\n".join(r["hypothesis"] for r in rows))

with open(ref_path, "w", encoding="utf-8") as f:
    f.write("\n".join(r["reference"] for r in rows))

print(f"Written {len(rows)} lines -> {hyp_path}, {ref_path}")
