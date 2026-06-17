"""
Patch compression ratio analysis for the Tokenization and Statistics angle.

Compares per-sentence BLT patch counts against BPE token counts and breaks the
comparison down by morpheme class using the fertility audit detail JSONL.

Expected inputs:
  - BLT patch JSONL from patching_scratch/run_blt_patching.py, or an existing
    per-sentence scores JSONL from error_analysis/plot_model_comparison.py.
  - fertility_by_class_detail.jsonl from lcm_scripts/fertility_audit.py.
  - Either a BPE tokenizer.json path (preferred for fresh BLT patch JSONL), or
    an existing scores JSONL containing tok_aug/tok_ret fields.

Outputs:
  - patch_bpe_sentence_comparison.csv
  - patch_compression_by_morpheme_class.json
  - patch_compression_by_morpheme_class.png

Example:
  python tokenization_statistics/patch_compression_by_morpheme_class.py \
      --blt-jsonl blt_marathi_patched.jsonl \
      --morpheme-detail results/fertility_by_class_detail.jsonl \
      --bpe-tokenizer ../Tokeniser_Retrained/tokenizer.json

If you already generated error_analysis/all_sentences_scores.jsonl:
  python tokenization_statistics/patch_compression_by_morpheme_class.py \
      --scores-jsonl error_analysis/all_sentences_scores.jsonl \
      --morpheme-detail results/fertility_by_class_detail.jsonl \
      --bpe-column tok_ret
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("patch_compression")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer

CLASS_ORDER = ["noun_root", "verb_inflection", "compound", "postposition", "other"]
CLASS_DISPLAY = {
    "noun_root": "Noun roots",
    "verb_inflection": "Verb inflections",
    "compound": "Compound words",
    "postposition": "Postpositions",
    "other": "Other",
}
CLASS_COLORS = {
    "noun_root": "#2563eb",
    "verb_inflection": "#16a34a",
    "compound": "#dc2626",
    "postposition": "#9333ea",
    "other": "#6b7280",
}


def load_jsonl(path: str) -> list[dict[str, Any]]:
    log.info("Loading JSONL: %s", path)
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    log.info("Loaded %d rows from %s", len(rows), path)
    return rows


def as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required field: {field_name}")
    return int(value)


def resolve_sentence_id(row: dict[str, Any]) -> int:
    for key in ("sentence_index", "sentence_id", "idx"):
        if key in row:
            return as_int(row[key], key)
    raise ValueError(
        "Could not find sentence index field; expected sentence_index, sentence_id, or idx"
    )


def load_blt_records(blt_jsonl: str, tokenizer_path: str) -> dict[int, dict[str, Any]]:
    log.info("Loading BLT records from %s with tokenizer %s", blt_jsonl, tokenizer_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    records: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(blt_jsonl):
        sentence_id = resolve_sentence_id(row)
        text = row.get("marathi_text") or row.get("text") or row.get("sentence")
        if text is None:
            raise ValueError(
                "BLT JSONL rows must contain marathi_text/text/sentence when --bpe-tokenizer is used"
            )
        bpe_tokens = len(tokenizer.encode(text).ids)
        blt_patches = as_int(
            row.get("num_patches") or row.get("patches_blt"), "num_patches"
        )
        num_bytes = as_int(
            row.get("num_bytes") or len(text.encode("utf-8")), "num_bytes"
        )
        records[sentence_id] = {
            "sentence_id": sentence_id,
            "text": text,
            "blt_patches": blt_patches,
            "bpe_tokens": bpe_tokens,
            "num_bytes": num_bytes,
        }
    log.info("Indexed %d BLT records by sentence_id", len(records))
    return records


def load_score_records(scores_jsonl: str, bpe_column: str) -> dict[int, dict[str, Any]]:
    log.info("Loading score records from %s (bpe_column=%s)", scores_jsonl, bpe_column)
    records: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(scores_jsonl):
        sentence_id = resolve_sentence_id(row)
        blt_patches = as_int(
            row.get("patches_blt") or row.get("num_patches"), "patches_blt"
        )
        bpe_tokens = as_int(row.get(bpe_column), bpe_column)
        num_bytes = as_int(row.get("n_bytes") or row.get("num_bytes"), "n_bytes")
        records[sentence_id] = {
            "sentence_id": sentence_id,
            "text": row.get("marathi_text") or row.get("text") or "",
            "blt_patches": blt_patches,
            "bpe_tokens": bpe_tokens,
            "num_bytes": num_bytes,
        }
    log.info("Indexed %d score records by sentence_id", len(records))
    return records


def load_sentence_classes(detail_jsonl: str) -> dict[int, dict[str, int]]:
    log.info("Loading morpheme class detail from %s", detail_jsonl)
    class_counts_by_sentence: dict[int, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in load_jsonl(detail_jsonl):
        sentence_id = resolve_sentence_id(row)
        cls = row.get("class")
        if cls not in CLASS_DISPLAY:
            cls = "other"
        class_counts_by_sentence[sentence_id][cls] += 1
    result = {sid: dict(counts) for sid, counts in class_counts_by_sentence.items()}
    log.info("Loaded class counts for %d sentences", len(result))
    return result


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def build_analysis(
    records: dict[int, dict[str, Any]],
    sentence_classes: dict[int, dict[str, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sentence_rows: list[dict[str, Any]] = []
    class_sentence_ids: dict[str, set[int]] = {cls: set() for cls in CLASS_ORDER}
    class_word_counts: dict[str, int] = {cls: 0 for cls in CLASS_ORDER}

    matched_sentence_ids = sorted(set(records).intersection(sentence_classes))
    log.info(
        "Matched %d sentences between token records and morpheme classes",
        len(matched_sentence_ids),
    )
    if not matched_sentence_ids:
        raise ValueError(
            "No overlapping sentence ids between token/patch records and morpheme detail rows"
        )

    for sentence_id in matched_sentence_ids:
        record = records[sentence_id]
        counts = sentence_classes[sentence_id]
        classes_present = [cls for cls in CLASS_ORDER if counts.get(cls, 0) > 0]
        for cls in classes_present:
            class_sentence_ids[cls].add(sentence_id)
            class_word_counts[cls] += counts[cls]

        bpe_tokens = record["bpe_tokens"]
        blt_patches = record["blt_patches"]
        ratio = blt_patches / bpe_tokens if bpe_tokens else 0.0
        savings_pct = (1.0 - ratio) * 100.0 if bpe_tokens else 0.0
        sentence_rows.append(
            {
                "sentence_id": sentence_id,
                "blt_patches": blt_patches,
                "bpe_tokens": bpe_tokens,
                "patch_to_bpe_ratio": round(ratio, 6),
                "blt_patch_delta": blt_patches - bpe_tokens,
                "blt_patch_savings_pct": round(savings_pct, 4),
                "num_bytes": record["num_bytes"],
                "morpheme_classes": ";".join(classes_present),
                "class_word_counts": json.dumps(
                    counts, ensure_ascii=False, sort_keys=True
                ),
                "text_preview": record.get("text", "")[:100],
            }
        )

    summary: dict[str, Any] = {
        "study": "Patch compression ratio analysis by morpheme class",
        "definition": (
            "Each class bucket contains every sentence with at least one word of that "
            "morpheme class; a sentence can contribute to multiple class buckets."
        ),
        "num_sentences_with_token_counts": len(records),
        "num_sentences_with_morpheme_classes": len(sentence_classes),
        "num_matched_sentences": len(matched_sentence_ids),
        "overall": {},
        "classes": {},
    }

    def values_for(ids: list[int], key: str) -> list[float]:
        return [float(records[sid][key]) for sid in ids]

    def ratios_for(ids: list[int]) -> list[float]:
        vals: list[float] = []
        for sid in ids:
            bpe = float(records[sid]["bpe_tokens"])
            vals.append(float(records[sid]["blt_patches"]) / bpe if bpe else 0.0)
        return vals

    def deltas_for(ids: list[int]) -> list[float]:
        return [
            float(records[sid]["blt_patches"] - records[sid]["bpe_tokens"])
            for sid in ids
        ]

    overall_ids = matched_sentence_ids
    summary["overall"] = {
        "num_sentences": len(overall_ids),
        "avg_blt_patches_per_sentence": summarize(
            values_for(overall_ids, "blt_patches")
        )["mean"],
        "avg_bpe_tokens_per_sentence": summarize(values_for(overall_ids, "bpe_tokens"))[
            "mean"
        ],
        "patch_to_bpe_ratio": summarize(ratios_for(overall_ids)),
        "blt_minus_bpe_tokens": summarize(deltas_for(overall_ids)),
    }

    for cls in CLASS_ORDER:
        ids = sorted(class_sentence_ids[cls])
        class_summary = {
            "display_name": CLASS_DISPLAY[cls],
            "num_sentences": len(ids),
            "num_class_words": class_word_counts[cls],
            "avg_blt_patches_per_sentence": summarize(values_for(ids, "blt_patches"))[
                "mean"
            ],
            "avg_bpe_tokens_per_sentence": summarize(values_for(ids, "bpe_tokens"))[
                "mean"
            ],
            "patch_to_bpe_ratio": summarize(ratios_for(ids)),
            "blt_minus_bpe_tokens": summarize(deltas_for(ids)),
        }
        summary["classes"][cls] = class_summary

    return sentence_rows, summary


def write_sentence_csv(rows: list[dict[str, Any]], path: str) -> None:
    log.info("Writing sentence CSV (%d rows) to %s", len(rows), path)
    fieldnames = [
        "sentence_id",
        "blt_patches",
        "bpe_tokens",
        "patch_to_bpe_ratio",
        "blt_patch_delta",
        "blt_patch_savings_pct",
        "num_bytes",
        "morpheme_classes",
        "class_word_counts",
        "text_preview",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_class_comparison(summary: dict[str, Any], path: str) -> None:
    labels = [CLASS_DISPLAY[cls] for cls in CLASS_ORDER]
    blt_means = [
        summary["classes"][cls]["avg_blt_patches_per_sentence"] for cls in CLASS_ORDER
    ]
    bpe_means = [
        summary["classes"][cls]["avg_bpe_tokens_per_sentence"] for cls in CLASS_ORDER
    ]
    ratios = [
        summary["classes"][cls]["patch_to_bpe_ratio"]["mean"] for cls in CLASS_ORDER
    ]

    x = list(range(len(CLASS_ORDER)))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    ax1.bar(
        [i - width / 2 for i in x],
        blt_means,
        width,
        label="BLT patches / sentence",
        color="#e74c3c",
    )
    ax1.bar(
        [i + width / 2 for i in x],
        bpe_means,
        width,
        label="BPE tokens / sentence",
        color="#3498db",
    )
    ax1.set_ylabel("Average count per sentence")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_title("BLT Patch Count vs. BPE Token Count by Morpheme Class")
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines["top"].set_visible(False)

    for i, (blt, bpe) in enumerate(zip(blt_means, bpe_means)):
        top = max(blt, bpe)
        ax1.text(
            i, top + 0.5, f"{blt:.1f} / {bpe:.1f}", ha="center", va="bottom", fontsize=9
        )

    ax2 = ax1.twinx()
    ax2.plot(x, ratios, color="#111827", marker="o", linewidth=2, label="BLT/BPE ratio")
    ax2.axhline(1.0, color="#111827", linestyle="--", linewidth=1, alpha=0.35)
    ax2.set_ylabel("BLT patches ÷ BPE tokens")
    ax2.spines["top"].set_visible(False)
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--blt-jsonl", help="BLT patch JSONL with num_patches and text"
    )
    input_group.add_argument(
        "--scores-jsonl",
        help="Per-sentence scores JSONL with patches_blt and a BPE column",
    )
    parser.add_argument(
        "--morpheme-detail", required=True, help="fertility_by_class_detail.jsonl path"
    )
    parser.add_argument(
        "--bpe-tokenizer",
        help="tokenizer.json used to compute BPE counts for --blt-jsonl",
    )
    parser.add_argument(
        "--bpe-column",
        default="tok_ret",
        help="BPE token-count column in --scores-jsonl (for example tok_aug or tok_ret)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for CSV, JSON, and PNG outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Starting patch compression analysis, output_dir=%s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.blt_jsonl:
        if not args.bpe_tokenizer:
            raise ValueError("--bpe-tokenizer is required with --blt-jsonl")
        records = load_blt_records(args.blt_jsonl, args.bpe_tokenizer)
        bpe_source = args.bpe_tokenizer
    else:
        records = load_score_records(args.scores_jsonl, args.bpe_column)
        bpe_source = f"{args.scores_jsonl}:{args.bpe_column}"

    sentence_classes = load_sentence_classes(args.morpheme_detail)

    sentence_rows, summary = build_analysis(records, sentence_classes)
    log.info(
        "Analysis built: %d sentence rows, %d classes",
        len(sentence_rows),
        len(summary["classes"]),
    )
    summary["inputs"] = {
        "blt_or_scores_source": args.blt_jsonl or args.scores_jsonl,
        "bpe_source": bpe_source,
        "morpheme_detail": args.morpheme_detail,
    }

    csv_path = os.path.join(args.output_dir, "patch_bpe_sentence_comparison.csv")
    json_path = os.path.join(
        args.output_dir, "patch_compression_by_morpheme_class.json"
    )
    png_path = os.path.join(args.output_dir, "patch_compression_by_morpheme_class.png")

    log.info("Writing CSV output to %s", csv_path)
    write_sentence_csv(sentence_rows, csv_path)
    log.info("Writing JSON summary to %s", json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Plotting class comparison to %s", png_path)
    plot_class_comparison(summary, png_path)

    overall = summary["overall"]
    log.info(
        "Patch compression ratio analysis complete: %d matched sentences, "
        "avg BLT/BPE ratio %.4f",
        summary["num_matched_sentences"],
        overall["patch_to_bpe_ratio"]["mean"],
    )


if __name__ == "__main__":
    main()
