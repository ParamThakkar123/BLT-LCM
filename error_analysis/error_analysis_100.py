"""
Error Analysis: Find 100 sentences where Phase 2 (BLT) scores lower than Phase 1 tokenizers.

Comparison metric: "fertility" — how many tokens/patches a system needs per Devanagari word.
  Phase 1 tokenizers: tokens_per_word (subword fertility)
  Phase 2 BLT: patches_per_word (entropy-based patch fertility)

Lower fertility = better compression = the system handles the sentence well.
A sentence where BLT fertility > Phase1 fertility means Phase 2 regressed.

Categories:
  1. Long compound words
  2. Code-mixed Marathi-English input
  3. Rare Unicode / uncommon Devanagari sequences
  4. Very short sentences (< 5 words)
  5. Domain-specific vocabulary (legal, medical)
"""

import argparse
import json
import os
import re
import sys
import csv
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lcm_scripts"))

import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from device_utils import report_device
from checkpoint_utils import ResumableJsonl, config_fingerprint, load_model_state

from run_blt_patching import (
    text_to_byte_tokens,
    ByteEntropyModel,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
    DEFAULT_THRESHOLD,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
REPO_ROOT = os.path.join(PROJECT_ROOT, "..")

# ── helpers ──────────────────────────────────────────────────

DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
LEGAL_KEYWORDS = [
    "कायदा", "न्यायालय", "न्यायाधीश", "कलम", "विधेयक", "अधिनियम",
    "याचिका", "फिर्याद", "खटला", "अपील", "दंड", "शिक्षा", "जामीन",
    "वकील", "साक्षीदार", "पुरावा", "आरोप", "निकाल", "सुनावणी",
    "घटना", "संविधान", "कोर्ट", "हायकोर्ट", "सुप्रीम", "जिल्हा न्यायालय",
    "फौजदारी", "दिवाणी", "कायदेशीर", "प्रतिवादी", "फिर्यादी",
]
MEDICAL_KEYWORDS = [
    "रुग्ण", "डॉक्टर", "वैद्यकीय", "उपचार", "शस्त्रक्रिया", "रक्त",
    "रोग", "लक्षणे", "औषध", "हृदय", "कर्करोग", "मधुमेह", "रुग्णालय",
    "दवाखाना", "तपासणी", "निदान", "आजार", "संसर्ग", "प्रतिकारशक्ती",
    "लसीकरण", "इंजेक्शन", "ऑपरेशन", "अस्थि", "मज्जातंतू",
    "दंतवैद्य", "नेत्र", "चिकित्सा", "पॅथॉलॉजी",
]
RARE_UNICODE_RE = re.compile(
    r"[\u200B-\u200F\u202A-\u202E\uFEFF\u0900\u0901\u093C\u094D\u0951-\u0954"
    r"\u0955-\u0957\u0962\u0963\u0970\u0971]"
)
COMPOUND_DEVANAGARI = re.compile(r"[\u0900-\u097F]{15,}")


def split_marathi_words(text):
    return [w for w in re.split(r"\s+", text.strip()) if w]


def count_devanagari_words(words):
    return sum(1 for w in words if DEVANAGARI_RANGE.search(w))


def has_english_mixing(text):
    latin_matches = LATIN_WORD.findall(text)
    has_dev = bool(DEVANAGARI_RANGE.search(text))
    return has_dev and len(latin_matches) >= 1


def has_long_compounds(text):
    return bool(COMPOUND_DEVANAGARI.search(text))


def has_rare_unicode(text):
    count = len(RARE_UNICODE_RE.findall(text))
    return count >= 2


def is_short_sentence(words):
    return len(words) < 5


def is_domain_specific(text, words):
    lower_text = text.lower()
    legal_count = sum(1 for kw in LEGAL_KEYWORDS if kw in text)
    medical_count = sum(1 for kw in MEDICAL_KEYWORDS if kw in text)
    return legal_count >= 2 or medical_count >= 2, legal_count, medical_count


def categorize_sentence(text):
    words = split_marathi_words(text)
    categories = []

    if has_long_compounds(text):
        categories.append("Long compound words")
    if has_english_mixing(text):
        categories.append("Code-mixed Marathi-English")
    if has_rare_unicode(text):
        categories.append("Rare Unicode / uncommon Devanagari")
    if is_short_sentence(words):
        categories.append("Very short sentence (< 5 words)")
    is_domain, legal_c, med_c = is_domain_specific(text, words)
    if is_domain:
        if legal_c >= med_c:
            categories.append("Domain-specific: legal")
        else:
            categories.append("Domain-specific: medical")

    if not categories:
        categories.append("Other")

    return categories


# ── fertility computation ────────────────────────────────────

def compute_tokenizer_fertility(tokenizer, text):
    words = split_marathi_words(text)
    if not words:
        return float("inf"), 0, 0
    tokens = tokenizer.encode(text)
    n_tokens = len(tokens.ids)
    n_words = len(words)
    return n_tokens / n_words, n_tokens, n_words


def compute_blt_fertility(text, entropy_model, device, threshold=DEFAULT_THRESHOLD):
    words = split_marathi_words(text)
    if not words:
        return float("inf"), 0, 0
    byte_tokens = text_to_byte_tokens(text)
    if len(byte_tokens) == 0:
        return float("inf"), 0, 0
    tokens_tensor = torch.tensor([byte_tokens], dtype=torch.long).to(device)
    entropies = compute_entropies_for_tokens(tokens_tensor, entropy_model, device=device)
    entropies_list = entropies[0].tolist()
    boundaries, patch_lengths = entropy_patch_sentence(entropies_list, threshold)
    n_patches = len(boundaries)
    n_words = len(words)
    return n_patches / n_words, n_patches, n_words


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        default="auto",
        metavar="auto|never",
        help="'auto' (default) continues from a partial per-sentence scores "
        "file whose configuration matches; 'never' discards it.",
    )
    parser.add_argument(
        "--progress_jsonl",
        default=os.path.join(SCRIPT_DIR, "all_sentences_scores.jsonl"),
        help="Per-sentence fertility rows, streamed as they are computed.",
    )
    args = parser.parse_args()

    device = str(report_device())

    # Load Phase 1 tokenizers
    print("Loading Phase 1 tokenizers...")
    aug_path = os.path.join(REPO_ROOT, "Tokeniser_Augmented", "tokenizer.json")
    ret_path = os.path.join(REPO_ROOT, "Tokeniser_Retrained", "tokenizer.json")
    tok_aug = Tokenizer.from_file(aug_path)
    tok_ret = Tokenizer.from_file(ret_path)
    print(f"  Augmented vocab: {tok_aug.get_vocab_size()}")
    print(f"  Retrained vocab: {tok_ret.get_vocab_size()}")

    # Load Phase 2 BLT entropy model
    print("Loading Phase 2 BLT entropy model...")
    ckpt_path = os.path.join(PROJECT_ROOT, "patching_scratch", "entropy_model_marathi.pt")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    entropy_model = ByteEntropyModel(
        vocab_size=cfg.get("vocab_size", 260),
        dim=cfg.get("dim", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        max_seqlen=cfg.get("max_seqlen", 512),
        ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", 1.3),
    ).to(device)
    entropy_model.load_state_dict(load_model_state(checkpoint))
    entropy_model.eval()
    print("  Entropy model loaded.")

    # Load sentences
    sentences_path = os.path.join(PROJECT_ROOT, "marathi_sentences.json")
    print(f"Loading sentences from {sentences_path}...")
    with open(sentences_path, "r", encoding="utf-8") as f:
        all_sentences = json.load(f)
    print(f"  Loaded {len(all_sentences)} sentences.")

    # Use all sentences to find enough regressions
    sentences = all_sentences
    print(f"  Analyzing {len(sentences)} sentences...")

    # Compute per-sentence scores. The BLT pass is a neural forward per
    # sentence over the whole corpus, so rows stream to disk as they are
    # produced and an interrupted run resumes at the first one it had not
    # written.
    fingerprint = config_fingerprint(
        {
            "corpus": sentences_path,
            "num_sentences": len(sentences),
            "threshold": DEFAULT_THRESHOLD,
            "tok_aug": aug_path,
            "tok_ret": ret_path,
        }
    )
    writer = ResumableJsonl(
        args.progress_jsonl,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        key="idx",
    )
    if writer.done:
        print(f"  Resuming: {len(writer.done)}/{len(sentences)} already scored")

    for idx, text in enumerate(tqdm(sentences, desc="Computing fertility")):
        if writer.is_done(idx):
            continue
        text = text.strip()
        if not text:
            continue

        # Phase 1: best of augmented & retrained
        fert_aug, _, _ = compute_tokenizer_fertility(tok_aug, text)
        fert_ret, _, _ = compute_tokenizer_fertility(tok_ret, text)
        phase1_best = min(fert_aug, fert_ret)
        phase1_source = "augmented" if fert_aug <= fert_ret else "retrained"

        # Phase 2: BLT patch fertility
        fert_blt, n_patches, n_words = compute_blt_fertility(
            text, entropy_model, device
        )

        # Regression = BLT fertility is higher (worse) than Phase 1 best
        delta = fert_blt - phase1_best

        writer.append({
            "idx": idx,
            "text": text,
            "n_words": n_words,
            "fert_aug": round(fert_aug, 4),
            "fert_ret": round(fert_ret, 4),
            "phase1_best": round(phase1_best, 4),
            "phase1_source": phase1_source,
            "fert_blt": round(fert_blt, 4),
            "n_patches": n_patches,
            "delta": round(delta, 4),
        })
    writer.close()
    results = writer.all_records()

    # Sort by delta descending — biggest regressions first
    results.sort(key=lambda r: r["delta"], reverse=True)

    # Take top 100 where Phase 2 is worse (delta > 0), deduplicated by text
    seen_texts = set()
    failures = []
    for r in results:
        if r["delta"] <= 0:
            continue
        text_key = r["text"].strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        failures.append(r)
        if len(failures) >= 100:
            break

    if len(failures) < 100:
        print(f"\nWARNING: Only found {len(failures)} regression sentences (delta > 0).")
        print("Including all of them.")

    print(f"\nFound {len(failures)} failure sentences for analysis.")

    # Categorize each failure
    category_counts = {}
    for f in failures:
        cats = categorize_sentence(f["text"])
        f["categories"] = cats
        for c in cats:
            category_counts[c] = category_counts.get(c, 0) + 1

    # ── Write outputs ────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # 1. Detailed JSONL
    jsonl_path = os.path.join(out_dir, "failure_sentences_100.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(failures):
            record = {
                "rank": i + 1,
                "original_idx": r["idx"],
                "text": r["text"],
                "n_words": r["n_words"],
                "fertility_augmented": r["fert_aug"],
                "fertility_retrained": r["fert_ret"],
                "phase1_best_fertility": r["phase1_best"],
                "phase1_best_source": r["phase1_source"],
                "blt_fertility": r["fert_blt"],
                "n_patches": r["n_patches"],
                "delta_regression": r["delta"],
                "categories": r["categories"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {jsonl_path}")

    # 2. CSV summary
    csv_path = os.path.join(out_dir, "failure_sentences_100.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "OrigIdx", "Text_Preview", "N_Words",
            "Fert_Augmented", "Fert_Retrained", "Phase1_Best",
            "BLT_Fertility", "N_Patches", "Delta",
            "Categories",
        ])
        for i, r in enumerate(failures):
            writer.writerow([
                i + 1,
                r["idx"],
                r["text"][:120],
                r["n_words"],
                r["fert_aug"],
                r["fert_ret"],
                r["phase1_best"],
                r["fert_blt"],
                r["n_patches"],
                r["delta"],
                "; ".join(r["categories"]),
            ])
    print(f"Wrote {csv_path}")

    # 3. Category summary
    summary_path = os.path.join(out_dir, "category_summary.json")
    summary = {
        "total_sentences_analyzed": len(sentences),
        "total_regressions_found": len([r for r in results if r["delta"] > 0]),
        "top_100_analyzed": len(failures),
        "category_counts": dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        "avg_delta_top100": round(
            sum(f["delta"] for f in failures) / max(len(failures), 1), 4
        ),
        "max_delta": failures[0]["delta"] if failures else 0,
        "min_delta_in_top100": failures[-1]["delta"] if failures else 0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Wrote {summary_path}")

    # 4. Human-readable report
    report_path = os.path.join(out_dir, "error_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Error Analysis: Phase 2 (BLT) vs Phase 1 Tokenizers\n\n")
        f.write(f"**Sentences analyzed:** {len(sentences)}\n")
        f.write(f"**Total regressions (BLT fertility > Phase1 best):** "
                f"{summary['total_regressions_found']}\n")
        f.write(f"**Top 100 worst regressions analyzed below.**\n\n")

        f.write("## Category Distribution\n\n")
        f.write("| Category | Count | % of 100 |\n")
        f.write("|----------|------:|--------:|\n")
        for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {cat} | {cnt} | {cnt}% |\n")

        f.write(f"\n## Summary Statistics\n\n")
        f.write(f"- Average fertility delta (top 100): **{summary['avg_delta_top100']}**\n")
        f.write(f"- Worst regression delta: **{summary['max_delta']}**\n")
        f.write(f"- 100th regression delta: **{summary['min_delta_in_top100']}**\n\n")

        f.write("## Top 100 Failure Sentences\n\n")
        for i, r in enumerate(failures):
            f.write(f"### {i+1}. (delta={r['delta']:.4f})\n")
            f.write(f"**Categories:** {', '.join(r['categories'])}\n\n")
            preview = r["text"][:200]
            f.write(f"> {preview}{'...' if len(r['text']) > 200 else ''}\n\n")
            f.write(f"| Metric | Value |\n|--------|------:|\n")
            f.write(f"| Words | {r['n_words']} |\n")
            f.write(f"| Phase1 Augmented fertility | {r['fert_aug']} |\n")
            f.write(f"| Phase1 Retrained fertility | {r['fert_ret']} |\n")
            f.write(f"| Phase1 Best | {r['phase1_best']} ({r['phase1_source']}) |\n")
            f.write(f"| BLT fertility | {r['fert_blt']} |\n")
            f.write(f"| BLT patches | {r['n_patches']} |\n")
            f.write(f"| Delta (regression) | {r['delta']} |\n\n")

    print(f"Wrote {report_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Sentences analyzed: {len(sentences)}")
    print(f"Total regressions:  {summary['total_regressions_found']}")
    print(f"Top 100 avg delta:  {summary['avg_delta_top100']}")
    print(f"\nCategory breakdown:")
    for cat, cnt in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:45s} {cnt:3d}")
    print()


if __name__ == "__main__":
    main()
