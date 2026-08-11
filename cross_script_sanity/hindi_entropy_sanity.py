"""Cross-script sanity check for BLT entropy patching on Hindi.

Runs the Marathi-trained byte entropy patcher on 100 Hindi (Devanagari)
sentences and compares entropy patch starts with lightweight Hindi boundary
proxies: word starts plus starts of common postpositions/inflectional suffixes.
The intent is not to replace a gold morphological analyzer; it is a small,
reproducible sanity check that the patch-boundary signal is not Marathi-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "patching_scratch"))
sys.path.append(str(REPO_ROOT / "lcm_scripts"))

from device_utils import report_device  # noqa: E402
from run_blt_patching import (  # noqa: E402
    DEFAULT_THRESHOLD,
    ByteEntropyModel,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
    text_to_byte_tokens,
)

COMMON_HINDI_SUFFIXES = (
    "ों", "ें", "याँ", "ियाँ", "ता", "ती", "ते", "ना", "ने", "कर", "करना",
    "वाला", "वाली", "वाले", "पन", "कारी", "कार", "ई", "आई", "ता है", "ती है",
)
COMMON_HINDI_POSTPOSITIONS = {"का", "की", "के", "को", "से", "में", "पर", "तक", "लिए", "साथ", "बाद", "पहले", "द्वारा"}

SUBJECTS = ["विद्यार्थी", "शिक्षक", "किसान", "डॉक्टर", "इंजीनियर", "लेखक", "कलाकार", "वैज्ञानिक", "पत्रकार", "खिलाड़ी"]
OBJECTS = ["विद्यालय", "अस्पताल", "खेत", "प्रयोगशाला", "पुस्तकालय", "बाजार", "कार्यालय", "गांव", "शहर", "विश्वविद्यालय"]
VERBS = ["जाता है", "काम करता है", "अध्ययन करती है", "रिपोर्ट लिखता है", "प्रशिक्षण लेती है", "सहयोग करता है", "योजना बनाता है", "समस्या समझती है", "निर्णय लेता है", "परिणाम बताती है"]
ADVERBS = ["आज", "कल", "सुबह", "शाम को", "धीरे-धीरे", "अक्सर", "सावधानी से", "मिलकर", "नियमित रूप से", "जल्दी"]


def build_hindi_sentences(n: int = 100) -> list[str]:
    sentences = []
    for i in range(n):
        subj = SUBJECTS[i % len(SUBJECTS)]
        obj = OBJECTS[(i * 3) % len(OBJECTS)]
        verb = VERBS[(i * 7) % len(VERBS)]
        adv = ADVERBS[(i * 5) % len(ADVERBS)]
        post = ["में", "से", "के लिए", "पर", "के साथ"][i % 5]
        sentences.append(f"{adv} {subj} {obj} {post} {verb}।")
    return sentences


def char_to_byte_offsets(text: str) -> list[int]:
    offsets = []
    total = 0
    for ch in text:
        offsets.append(total)
        total += len(ch.encode("utf-8"))
    return offsets


def proxy_boundaries(text: str) -> set[int]:
    """Word-start and suffix-start byte offsets as lightweight Hindi boundary proxy."""
    offsets = char_to_byte_offsets(text)
    boundaries = {0}
    words = text.replace("।", " ।").split()
    search_from = 0
    for word in words:
        char_start = text.find(word, search_from)
        if char_start < 0:
            continue
        boundaries.add(offsets[char_start])
        if word in COMMON_HINDI_POSTPOSITIONS:
            boundaries.add(offsets[char_start])
        for suffix in COMMON_HINDI_SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                suffix_char_start = char_start + len(word) - len(suffix)
                boundaries.add(offsets[suffix_char_start])
        search_from = char_start + len(word)
    return boundaries


def boundary_f1(predicted: set[int], gold: set[int], tolerance: int = 2) -> tuple[float, float, float]:
    matched_gold = set()
    tp = 0
    for pred in predicted:
        candidates = [g for g in gold if g not in matched_gold and abs(pred - g) <= tolerance]
        if candidates:
            best = min(candidates, key=lambda g: abs(pred - g))
            matched_gold.add(best)
            tp += 1
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="patching_scratch/entropy_model_marathi.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num_sentences", type=int, default=100)
    parser.add_argument("--output_dir", default="cross_script_sanity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = report_device(args.device)
    model_path = REPO_ROOT / args.model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, ByteEntropyModel):
        model = checkpoint
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = ByteEntropyModel(**checkpoint.get("config", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise TypeError(f"Expected ByteEntropyModel or state-dict checkpoint, got {type(checkpoint)!r}")
    model.to(device).eval()

    rows = []
    for idx, sentence in enumerate(build_hindi_sentences(args.num_sentences)):
        tokens = text_to_byte_tokens(sentence)
        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        entropies = compute_entropies_for_tokens(token_tensor, model, device=str(device))[0].tolist()
        patch_starts, patch_lengths = entropy_patch_sentence(entropies, args.threshold)
        gold = proxy_boundaries(sentence)
        precision, recall, f1 = boundary_f1(set(patch_starts), gold)
        rows.append({
            "sentence_id": idx,
            "language": "Hindi",
            "script": "Devanagari",
            "sentence": sentence,
            "num_bytes": len(tokens),
            "num_patches": len(patch_starts),
            "num_proxy_boundaries": len(gold),
            "precision_tol2": precision,
            "recall_tol2": recall,
            "f1_tol2": f1,
            "patch_boundaries": patch_starts,
            "patch_lengths": patch_lengths,
            "proxy_boundaries": sorted(gold),
        })

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)
    jsonl_path = output_dir / "hindi_entropy_sanity.jsonl"
    csv_path = output_dir / "hindi_entropy_sanity_summary.csv"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "language": "Hindi",
        "script": "Devanagari",
        "num_sentences": len(rows),
        "threshold": args.threshold,
        "mean_precision_tol2": statistics.mean(r["precision_tol2"] for r in rows),
        "mean_recall_tol2": statistics.mean(r["recall_tol2"] for r in rows),
        "mean_f1_tol2": statistics.mean(r["f1_tol2"] for r in rows),
        "mean_patches_per_sentence": statistics.mean(r["num_patches"] for r in rows),
        "mean_proxy_boundaries_per_sentence": statistics.mean(r["num_proxy_boundaries"] for r in rows),
    }
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {jsonl_path.relative_to(REPO_ROOT)} and {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
