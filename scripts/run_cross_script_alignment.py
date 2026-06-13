"""Run cross-script entropy-boundary alignment on the Hindi sanity corpus.

This is a lightweight companion to ``morpheme_alignment/morpheme_boundary_alignment.py``.
It uses the same byte-transition entropy patching sweep, but keeps the run
self-contained for the checked-in 100-sentence Hindi corpus. Because Indic NLP
morfessor resources are not bundled with this repository, Hindi gold boundaries
combine word boundaries with a small, deterministic suffix heuristic; the output
is therefore a sanity check, not a replacement for a fully annotated Hindi
morpheme benchmark.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable

THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5]
DEFAULT_INPUT = Path("data/cross_script/hindi_sentences_100.json")
DEFAULT_OUTPUT = Path("results/cross_script_hindi_alignment_results.json")
DEFAULT_MARATHI_F1 = 0.6341
HINDI_SUFFIXES = (
    "ों", "ियों", "ाएं", "ाए", "ीय", "ता", "ती", "ते", "ना", "ने", "कर", "पर", "से", "में", "का", "की", "के",
)


def load_sentences(path: Path, limit: int | None = None) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    sentences = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
    return sentences[:limit] if limit else sentences


def build_transition_probs(sentences: Iterable[str]) -> dict[int, dict[int, float]]:
    transitions: dict[int, Counter[int]] = defaultdict(Counter)
    for sentence in sentences:
        seq = sentence.encode("utf-8")
        for i in range(len(seq) - 1):
            transitions[seq[i]][seq[i + 1]] += 1
    probs: dict[int, dict[int, float]] = {}
    for prev, counter in transitions.items():
        total = sum(counter.values())
        probs[prev] = {next_byte: count / total for next_byte, count in counter.items()}
    return probs


def compute_entropy(seq: bytes, probs: dict[int, dict[int, float]], default_entropy: float = 8.0) -> list[float]:
    entropies: list[float] = []
    for i in range(len(seq) - 1):
        prev = seq[i]
        if prev not in probs:
            entropies.append(default_entropy)
            continue
        entropy = 0.0
        for probability in probs[prev].values():
            if probability > 0:
                entropy -= probability * math.log2(probability)
        entropies.append(entropy)
    entropies.append(default_entropy)
    return entropies


def get_patch_boundaries(entropies: list[float], theta: float) -> set[int]:
    return {idx for idx, entropy in enumerate(entropies) if entropy > theta}


def hindi_boundary_offsets(sentence: str) -> set[int]:
    """Return byte offsets for word boundaries plus simple Hindi suffix splits."""
    boundaries: set[int] = set()
    byte_cursor = 0
    for token in sentence.split():
        token = token.strip()
        token_text = token.strip("।,;:!?()[]{}\"'")
        if not token_text:
            byte_cursor += len(token.encode("utf-8")) + 1
            continue

        token_start = sentence.find(token, byte_cursor)
        if token_start < 0:
            token_start = byte_cursor
        token_start_bytes = len(sentence[:token_start].encode("utf-8"))
        token_bytes = token_text.encode("utf-8")

        for suffix in HINDI_SUFFIXES:
            if token_text.endswith(suffix) and len(token_text) > len(suffix) + 1:
                stem = token_text[: -len(suffix)]
                boundaries.add(token_start_bytes + len(stem.encode("utf-8")))
                break

        token_end = token_start_bytes + len(token_bytes)
        if token_end < len(sentence.encode("utf-8")):
            boundaries.add(token_end)
        byte_cursor = token_start + len(token)
    return boundaries


def _count_matched(query_sorted: list[int], ref_sorted: list[int], tolerance: int) -> int:
    matched = 0
    for query in query_sorted:
        lo = bisect.bisect_left(ref_sorted, query - tolerance)
        hi = bisect.bisect_right(ref_sorted, query + tolerance)
        if lo < hi:
            matched += 1
    return matched


def boundary_metrics(predicted: set[int], gold: set[int], tolerance: int) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    pred_sorted = sorted(predicted)
    gold_sorted = sorted(gold)
    precision = _count_matched(pred_sorted, gold_sorted, tolerance) / len(pred_sorted)
    recall = _count_matched(gold_sorted, pred_sorted, tolerance) / len(gold_sorted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run_alignment(sentences: list[str], tolerance: int, marathi_baseline_f1: float) -> dict[str, object]:
    probs = build_transition_probs(sentences)
    aggregates = {tau: {"precision": [], "recall": [], "f1": [], "predicted": [], "gold": []} for tau in THRESHOLDS}
    per_sentence = []

    for sentence_id, sentence in enumerate(sentences):
        seq = sentence.encode("utf-8")
        entropies = compute_entropy(seq, probs)
        gold = hindi_boundary_offsets(sentence)
        row = {"sentence_id": sentence_id, "byte_length": len(seq), "num_gold_boundaries": len(gold)}
        for tau in THRESHOLDS:
            predicted = get_patch_boundaries(entropies, tau)
            precision, recall, f1 = boundary_metrics(predicted, gold, tolerance)
            bucket = aggregates[tau]
            bucket["precision"].append(precision)
            bucket["recall"].append(recall)
            bucket["f1"].append(f1)
            bucket["predicted"].append(len(predicted))
            bucket["gold"].append(len(gold))
            row[f"tau_{tau}_precision"] = round(precision, 4)
            row[f"tau_{tau}_recall"] = round(recall, 4)
            row[f"tau_{tau}_f1"] = round(f1, 4)
        per_sentence.append(row)

    thresholds: dict[str, dict[str, float]] = {}
    best_tau = THRESHOLDS[0]
    best_f1 = -1.0
    for tau in THRESHOLDS:
        f1 = mean(aggregates[tau]["f1"])
        thresholds[str(tau)] = {
            "precision": round(mean(aggregates[tau]["precision"]), 4),
            "recall": round(mean(aggregates[tau]["recall"]), 4),
            "f1": round(f1, 4),
            "avg_predicted_boundaries": round(mean(aggregates[tau]["predicted"]), 2),
            "avg_gold_boundaries": round(mean(aggregates[tau]["gold"]), 2),
        }
        if f1 > best_f1:
            best_tau = tau
            best_f1 = f1

    return {
        "study": "Cross-script Hindi entropy-boundary alignment sanity check",
        "language": "hindi",
        "num_sentences": len(sentences),
        "tolerance_bytes": tolerance,
        "gold_boundary_method": "word boundaries plus deterministic Hindi suffix heuristic",
        "marathi_reference": {
            "source": "fixed_chunk_ablation/fixed_chunk_ablation.py documents tau=1.0 F1=0.6341",
            "best_f1": marathi_baseline_f1,
        },
        "thresholds": thresholds,
        "best_tau": best_tau,
        "best_f1": round(best_f1, 4),
        "delta_vs_marathi_best_f1": round(best_f1 - marathi_baseline_f1, 4),
        "claim_guidance": "Do not call BLT-LCM script-agnostic from this sanity check alone; use it as support pending real Hindi morpheme annotations.",
        "per_sentence": per_sentence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Hindi entropy-boundary F1 against the Marathi reference result.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--tolerance", type=int, default=3)
    parser.add_argument("--marathi-baseline-f1", type=float, default=DEFAULT_MARATHI_F1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sentences = load_sentences(args.input, args.limit)
    results = run_alignment(sentences, tolerance=args.tolerance, marathi_baseline_f1=args.marathi_baseline_f1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Hindi alignment results to {args.output}")
    print(f"Hindi best F1={results['best_f1']:.4f} at tau={results['best_tau']} vs Marathi F1={args.marathi_baseline_f1:.4f}")


if __name__ == "__main__":
    main()
