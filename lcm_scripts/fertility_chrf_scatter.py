"""Compute and plot fertility λ vs. per-class Δ chrF++.

This analysis connects the Day 1 fertility audit to the Day 6 MT metric suite.
It aggregates sentences by the morpheme classes observed in
``fertility_by_class_detail.jsonl``, computes chrF++ for BLT-LCM and the BPE-LCM
baseline on each class-specific sentence subset, and plots class fertility
against the BLT-over-BPE chrF++ gain.

Example:
  python lcm_scripts/fertility_chrf_scatter.py \
    --fertility_json results/fertility_by_class.json \
    --fertility_detail_jsonl results/fertility_by_class_detail.jsonl \
    --bpe_hyp_file outputs/bpe_lcm.hyp.txt \
    --blt_hyp_file outputs/blt_lcm.hyp.txt \
    --ref_file outputs/ref.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Set


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from lcm_scripts.device_utils import report_cpu_only
from lcm_scripts.eval_metrics import compute_chrf


DEFAULT_CLASSES = ["noun_root", "verb_inflection", "compound", "postposition"]
DEFAULT_DISPLAY = {
    "noun_root": "Noun roots",
    "verb_inflection": "Verb inflections",
    "compound": "Compound words",
    "postposition": "Postpositions",
    "other": "Other",
}
DEFAULT_COLORS = {
    "noun_root": "#2563eb",
    "verb_inflection": "#16a34a",
    "compound": "#dc2626",
    "postposition": "#9333ea",
    "other": "#6b7280",
}


@dataclass(frozen=True)
class ClassChrfRow:
    """Per-class fertility, chrF++ gain, and bound diagnostics."""

    morpheme_class: str
    display_name: str
    sentence_count: int
    word_count: int
    fertility_lambda: float
    bpe_chrf: float
    blt_chrf: float
    delta_chrf: float
    bpe_error: float
    blt_error: float
    f_lambda: float
    bound_rhs: float
    error_ratio: float
    bound_satisfied: bool

    def to_csv_row(self) -> Dict[str, object]:
        return {
            "class": self.morpheme_class,
            "display_name": self.display_name,
            "sentence_count": self.sentence_count,
            "word_count": self.word_count,
            "fertility_lambda": f"{self.fertility_lambda:.6f}",
            "bpe_chrf": f"{self.bpe_chrf:.6f}",
            "blt_chrf": f"{self.blt_chrf:.6f}",
            "delta_chrf": f"{self.delta_chrf:.6f}",
            "bpe_error": f"{self.bpe_error:.6f}",
            "blt_error": f"{self.blt_error:.6f}",
            "f_lambda": f"{self.f_lambda:.6f}",
            "bound_rhs": f"{self.bound_rhs:.6f}",
            "error_ratio": "nan" if math.isnan(self.error_ratio) else f"{self.error_ratio:.6f}",
            "bound_satisfied": self.bound_satisfied,
        }


def read_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_fertility_summary(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "classes" not in data or not isinstance(data["classes"], dict):
        raise ValueError(f"Fertility summary lacks a 'classes' object: {path}")
    return data["classes"]


def load_class_sentence_ids(path: str, classes: Iterable[str]) -> Dict[str, Set[int]]:
    wanted = set(classes)
    sentence_ids = {cls: set() for cls in wanted}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            cls = row.get("class")
            if cls not in wanted:
                continue
            if "sentence_id" not in row:
                raise ValueError(f"Missing sentence_id at {path}:{line_number}")
            sentence_ids[cls].add(int(row["sentence_id"]))
    return sentence_ids


def validate_parallel_files(bpe_hyps: Sequence[str], blt_hyps: Sequence[str], refs: Sequence[str]) -> None:
    lengths = {"bpe_hyp_file": len(bpe_hyps), "blt_hyp_file": len(blt_hyps), "ref_file": len(refs)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Parallel files must have identical line counts; got {lengths}")


def bounded_subset(items: Sequence[str], ids: Set[int], label: str) -> List[str]:
    if not ids:
        return []
    max_id = max(ids)
    if max_id >= len(items):
        raise ValueError(
            f"Class '{label}' references sentence_id={max_id}, but metric files have only {len(items)} lines"
        )
    return [items[i] for i in sorted(ids)]


def theoretical_factor(fertility_lambda: float, exponent: float) -> float:
    """Return f(λ)=λ^-α for the *empirical diagnostic* E(BLT) ≤ f(λ)·E(BPE).

    This is the heuristic scaling relationship from the Theoretical Analysis
    (a motivation under stated assumptions, not a proven bound); α is fitted /
    swept, not derived. The monotone factor is 1 at λ=1 and decreases as
    fertility rises, matching the expected stronger BLT advantage for
    higher-fertility morpheme classes.
    """

    if fertility_lambda <= 0:
        return float("nan")
    return fertility_lambda ** (-exponent)


def compute_rows(
    fertility_classes: Dict[str, dict],
    class_sentence_ids: Dict[str, Set[int]],
    bpe_hyps: Sequence[str],
    blt_hyps: Sequence[str],
    refs: Sequence[str],
    classes: Sequence[str],
    bound_exponent: float = 1.0,
    metric_fn: Callable[[List[str], List[str]], float] = compute_chrf,
) -> List[ClassChrfRow]:
    rows: List[ClassChrfRow] = []
    for cls in classes:
        class_info = fertility_classes.get(cls)
        if class_info is None:
            raise ValueError(f"Class '{cls}' not found in fertility summary")

        ids = class_sentence_ids.get(cls, set())
        subset_bpe = bounded_subset(bpe_hyps, ids, cls)
        subset_blt = bounded_subset(blt_hyps, ids, cls)
        subset_refs = bounded_subset(refs, ids, cls)
        if not subset_refs:
            continue

        bpe_chrf = float(metric_fn(subset_bpe, subset_refs))
        blt_chrf = float(metric_fn(subset_blt, subset_refs))
        bpe_error = max(0.0, 100.0 - bpe_chrf)
        blt_error = max(0.0, 100.0 - blt_chrf)
        lam = float(class_info.get("fertility_lambda", 0.0))
        f_lam = theoretical_factor(lam, bound_exponent)
        bound_rhs = f_lam * bpe_error
        error_ratio = blt_error / bpe_error if bpe_error > 0 else float("nan")

        rows.append(
            ClassChrfRow(
                morpheme_class=cls,
                display_name=str(class_info.get("display_name", DEFAULT_DISPLAY.get(cls, cls))),
                sentence_count=len(ids),
                word_count=int(class_info.get("num_words", 0)),
                fertility_lambda=lam,
                bpe_chrf=bpe_chrf,
                blt_chrf=blt_chrf,
                delta_chrf=blt_chrf - bpe_chrf,
                bpe_error=bpe_error,
                blt_error=blt_error,
                f_lambda=f_lam,
                bound_rhs=bound_rhs,
                error_ratio=error_ratio,
                bound_satisfied=blt_error <= bound_rhs if not math.isnan(bound_rhs) else False,
            )
        )
    return rows


def write_rows_csv(rows: Sequence[ClassChrfRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].to_csv_row().keys()) if rows else [
        "class",
        "display_name",
        "sentence_count",
        "word_count",
        "fertility_lambda",
        "bpe_chrf",
        "blt_chrf",
        "delta_chrf",
        "bpe_error",
        "blt_error",
        "f_lambda",
        "bound_rhs",
        "error_ratio",
        "bound_satisfied",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson r for paired values, or NaN if undefined."""

    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have identical lengths")
    if len(xs) < 2:
        return float("nan")

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)


def rank_values(values: Sequence[float]) -> List[float]:
    """Return average ranks for values, using 1-indexed ranks and tie averaging."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for original_idx, _ in indexed[i:j]:
            ranks[original_idx] = avg_rank
        i = j
    return ranks


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Spearman ρ for paired values, or NaN if undefined."""

    if len(xs) != len(ys):
        raise ValueError("Correlation inputs must have identical lengths")
    if len(xs) < 2:
        return float("nan")
    return pearson_correlation(rank_values(xs), rank_values(ys))


def summarize_rows(rows: Sequence[ClassChrfRow], bound_exponent: float) -> Dict[str, object]:
    """Build a compact empirical validation summary for the scatter analysis."""

    lambdas = [row.fertility_lambda for row in rows]
    deltas = [row.delta_chrf for row in rows]
    bound_count = sum(1 for row in rows if row.bound_satisfied)
    return {
        "num_classes": len(rows),
        "bound_exponent": bound_exponent,
        "bound_satisfied_count": bound_count,
        "bound_satisfied_fraction": bound_count / len(rows) if rows else float("nan"),
        "pearson_lambda_delta_chrf": pearson_correlation(lambdas, deltas),
        "spearman_lambda_delta_chrf": spearman_correlation(lambdas, deltas),
        "classes": [row.to_csv_row() for row in rows],
    }


def write_summary_json(rows: Sequence[ClassChrfRow], path: str, bound_exponent: float) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summarize_rows(rows, bound_exponent), f, indent=2, ensure_ascii=False)


def plot_rows(rows: Sequence[ClassChrfRow], path: str, bound_exponent: float) -> None:
    if not rows:
        raise ValueError("No class rows available to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for row in rows:
        color = DEFAULT_COLORS.get(row.morpheme_class, "#111827")
        marker = "o" if row.bound_satisfied else "X"
        ax.scatter(
            row.fertility_lambda,
            row.delta_chrf,
            s=max(90.0, min(420.0, math.sqrt(max(row.sentence_count, 1)) * 12.0)),
            color=color,
            edgecolors="white",
            linewidth=1.0,
            marker=marker,
            label=row.display_name,
            alpha=0.88,
        )
        ax.annotate(
            f"{row.display_name}\nΔ={row.delta_chrf:+.2f}",
            (row.fertility_lambda, row.delta_chrf),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0.0, color="#374151", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fertility λ (avg morphemes per word)", fontsize=12)
    ax.set_ylabel("Δ chrF++ (BLT-LCM − BPE-LCM)", fontsize=12)
    ax.set_title("Fertility vs. chrF++ Gain by Morpheme Class", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    summary = summarize_rows(rows, bound_exponent)
    ax.text(
        0.02,
        0.98,
        f"Bound diagnostic: E(BLT) ≤ λ^-{bound_exponent:g} · E(BPE)\n"
        f"Satisfied: {summary['bound_satisfied_count']}/{len(rows)} classes\n"
        f"Pearson r(λ, ΔchrF++): {summary['pearson_lambda_delta_chrf']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fertility λ against BLT-over-BPE chrF++ gains.")
    parser.add_argument("--fertility_json", default="results/fertility_by_class.json")
    parser.add_argument("--fertility_detail_jsonl", default="results/fertility_by_class_detail.jsonl")
    parser.add_argument("--bpe_hyp_file", required=True, help="BPE-LCM/Phase 1 hypotheses, one sentence per line")
    parser.add_argument("--blt_hyp_file", required=True, help="BLT-LCM/Phase 2 hypotheses, one sentence per line")
    parser.add_argument("--ref_file", required=True, help="Reference translations, one sentence per line")
    parser.add_argument("--out_csv", default="results/fertility_chrf_delta_by_class.csv")
    parser.add_argument("--out_plot", default="results/fertility_chrf_delta_scatter.png")
    parser.add_argument("--out_summary", default="results/fertility_chrf_delta_summary.json")
    parser.add_argument("--include_other", action="store_true", help="Include the catch-all 'other' class in outputs")
    parser.add_argument(
        "--bound_exponent",
        type=float,
        default=1.0,
        help="α in f(λ)=λ^-α for the empirical E(BLT) ≤ f(λ)·E(BPE) diagnostic",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_cpu_only("chrF++ scoring and matplotlib rendering")
    classes = DEFAULT_CLASSES + (["other"] if args.include_other else [])

    fertility_classes = load_fertility_summary(args.fertility_json)
    class_sentence_ids = load_class_sentence_ids(args.fertility_detail_jsonl, classes)
    bpe_hyps = read_lines(args.bpe_hyp_file)
    blt_hyps = read_lines(args.blt_hyp_file)
    refs = read_lines(args.ref_file)
    validate_parallel_files(bpe_hyps, blt_hyps, refs)

    rows = compute_rows(
        fertility_classes,
        class_sentence_ids,
        bpe_hyps,
        blt_hyps,
        refs,
        classes,
        bound_exponent=args.bound_exponent,
    )
    write_rows_csv(rows, args.out_csv)
    write_summary_json(rows, args.out_summary, args.bound_exponent)
    plot_rows(rows, args.out_plot, args.bound_exponent)

    print(f"Wrote per-class Δ chrF++ table to: {args.out_csv}")
    print(f"Wrote empirical validation summary to: {args.out_summary}")
    print(f"Wrote fertility/Δ chrF++ scatter plot to: {args.out_plot}")
    for row in rows:
        status = "satisfies" if row.bound_satisfied else "violates"
        print(
            f"{row.display_name}: λ={row.fertility_lambda:.3f}, "
            f"ΔchrF++={row.delta_chrf:+.3f}, {status} bound"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
