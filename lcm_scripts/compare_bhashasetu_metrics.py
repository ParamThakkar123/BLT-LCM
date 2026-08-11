"""Build a same-fraction BLT/BPE/SONAR metric comparison table from CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row.setdefault("source_csv", path)
                rows.append(row)
    return rows


def norm_float(value: str) -> str:
    return f"{float(value):.6g}"


def main() -> None:
    p = argparse.ArgumentParser(description="Create three-way same-fraction metric comparison CSV")
    p.add_argument("--inputs", nargs="+", required=True, help="Metric CSVs to combine")
    p.add_argument("--fraction", type=float, required=True, help="Keep only this data fraction")
    p.add_argument("--models", nargs="+", default=["blt_lcm", "bpe_lcm", "sonar_lcm"])
    p.add_argument("--metrics", nargs="+", default=["chrF++", "BLEU"])
    p.add_argument("--out_csv", default="results/three_way_lcm_comparison.csv")
    p.add_argument("--strict", action="store_true", help="Fail unless every noise level has all requested models")
    from device_utils import report_cpu_only

    args = p.parse_args()
    report_cpu_only("CSV aggregation")

    rows = [r for r in read_rows(args.inputs) if abs(float(r.get("fraction", "nan")) - args.fraction) < 1e-9]
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        model = row.get("model", "")
        if model in args.models:
            grouped[norm_float(row.get("noise", "0"))][model] = row

    out_rows = []
    for noise in sorted(grouped, key=float):
        present = grouped[noise]
        missing = [m for m in args.models if m not in present]
        if missing and args.strict:
            raise RuntimeError(f"Missing models at fraction={args.fraction}, noise={noise}: {', '.join(missing)}")
        out = {"fraction": args.fraction, "noise": noise, "models_present": ";".join(sorted(present))}
        for model in args.models:
            row = present.get(model, {})
            for metric in args.metrics:
                out[f"{model}_{metric}"] = row.get(metric, "")
        out_rows.append(out)

    if not out_rows:
        raise RuntimeError("No matching rows found. Check --fraction, --models, and --inputs.")

    fieldnames = list(out_rows[0].keys())
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
