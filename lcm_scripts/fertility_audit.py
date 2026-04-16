"""
Fertility audit by morpheme class for Marathi.

Computes fertility (λ) — average morphemes per word — broken down by:
  1. Noun roots        (UPOS: NOUN, PROPN)
  2. Verb inflections  (UPOS: VERB, AUX)
  3. Compound words    (multi-morpheme nouns ≥ 2 segments)
  4. Postpositions     (UPOS: ADP)

Uses:
  - Stanza for POS tagging (UPOS)
  - Indic NLP UnsupervisedMorphAnalyzer for morpheme segmentation

Outputs:
  - results/fertility_by_class.json          : per-class λ + global λ
  - results/fertility_by_class_detail.jsonl   : per-word detail
  - Console table of fertility (λ) per morpheme class
"""

import sys
import io
import json
import os
import re
from collections import defaultdict

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── Indic NLP setup ──────────────────────────────────────────────────────────

from indicnlp import common
common.set_resources_path("D:/phase2/indic_nlp_resources")
from indicnlp import loader
loader.load()
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── Stanza setup ─────────────────────────────────────────────────────────────

import stanza

# Download model if not already present (no-op if cached)
stanza.download("mr", verbose=False)
nlp = stanza.Pipeline("mr", processors="tokenize,pos,lemma", verbose=False)

# ── Morpheme class definitions ───────────────────────────────────────────────

# UPOS tags → morpheme class mapping
NOUN_TAGS = {"NOUN", "PROPN"}
VERB_TAGS = {"VERB", "AUX"}
POSTP_TAGS = {"ADP"}

# Devanagari virama (halant) — used to detect conjunct clusters in compounds
VIRAMA = "\u094D"


def classify_word(upos, text, morphemes):
    """
    Assign a word to one of the four morpheme classes.

    Priority:
      1. Compound words — multi-morpheme nouns (≥2 morphemes) or words with
         conjunct clusters (virama) that are long (≥6 chars)
      2. Noun roots — single-morpheme nouns / proper nouns
      3. Verb inflections — verbs and auxiliaries
      4. Postpositions — adpositions
      5. Other — everything else (DET, CCONJ, PRON, ADV, ADJ, etc.)
    """
    n_morphemes = len(morphemes)

    # Compound word detection: multi-morpheme noun, or long word with conjuncts
    if upos in NOUN_TAGS and n_morphemes >= 2:
        return "compound"
    if n_morphemes >= 2 and VIRAMA in text and len(text) >= 6:
        return "compound"

    if upos in NOUN_TAGS:
        return "noun_root"
    if upos in VERB_TAGS:
        return "verb_inflection"
    if upos in POSTP_TAGS:
        return "postposition"

    return "other"


CLASS_LABELS = ["noun_root", "verb_inflection", "compound", "postposition", "other"]
CLASS_DISPLAY = {
    "noun_root": "Noun roots",
    "verb_inflection": "Verb inflections",
    "compound": "Compound words",
    "postposition": "Postpositions",
    "other": "Other (ADJ, ADV, DET, PRON, ...)",
}

# ── Load corpus ──────────────────────────────────────────────────────────────

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "marathi_sentences.json")
CORPUS_PATH = os.path.normpath(CORPUS_PATH)

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    all_sentences = json.load(f)

# Use a sample for efficiency — Stanza POS tagging is slower than byte-level ops.
# 5000 sentences gives statistically robust estimates; increase if needed.
SAMPLE_SIZE = min(5000, len(all_sentences))
sentences = all_sentences[:SAMPLE_SIZE]
print(f"Fertility audit: processing {SAMPLE_SIZE} / {len(all_sentences)} sentences")

# ── Process sentences ────────────────────────────────────────────────────────

# Accumulators per class
class_morpheme_counts = defaultdict(list)   # class -> list of morpheme counts per word
class_word_examples = defaultdict(list)     # class -> sample words (capped)

detail_rows = []   # per-word JSONL records
total_words = 0
total_morphemes = 0

BATCH_SIZE = 50   # Stanza batch size for efficiency

for batch_start in range(0, len(sentences), BATCH_SIZE):
    batch = sentences[batch_start : batch_start + BATCH_SIZE]

    # Stanza processes a list of sentences efficiently
    docs = [nlp(sent) for sent in batch]

    for doc_idx, doc in enumerate(docs):
        sent_idx = batch_start + doc_idx
        for sent in doc.sentences:
            for word in sent.words:
                text = word.text
                upos = word.upos if word.upos else "X"

                # Skip punctuation and symbols
                if upos in ("PUNCT", "SYM", "X") or not re.search(r"[\u0900-\u097F]", text):
                    continue

                # Morpheme segmentation via Indic NLP
                try:
                    morphemes = morph_analyzer.morph_analyze(text)
                except Exception:
                    morphemes = [text]

                n_morphemes = len(morphemes)
                cls = classify_word(upos, text, morphemes)

                class_morpheme_counts[cls].append(n_morphemes)
                total_words += 1
                total_morphemes += n_morphemes

                # Keep a few examples per class
                if len(class_word_examples[cls]) < 20:
                    class_word_examples[cls].append({
                        "word": text,
                        "upos": upos,
                        "morphemes": morphemes,
                        "n_morphemes": n_morphemes,
                    })

                detail_rows.append({
                    "sentence_id": sent_idx,
                    "word": text,
                    "upos": upos,
                    "morphemes": morphemes,
                    "n_morphemes": n_morphemes,
                    "class": cls,
                })

    # Progress
    processed = min(batch_start + BATCH_SIZE, len(sentences))
    if processed % 500 == 0 or processed == len(sentences):
        print(f"  Processed {processed}/{len(sentences)} sentences "
              f"({total_words} words so far)")

# ── Compute fertility per class ──────────────────────────────────────────────

results = {
    "study": "Fertility audit by morpheme class",
    "corpus": "ParamTh/BhashaSetu (Marathi)",
    "num_sentences": len(sentences),
    "total_words": total_words,
    "total_morphemes": total_morphemes,
    "global_fertility": round(total_morphemes / max(1, total_words), 4),
    "classes": {},
}

for cls in CLASS_LABELS:
    counts = class_morpheme_counts.get(cls, [])
    n = len(counts)
    if n == 0:
        avg = 0.0
        std = 0.0
        min_m = 0
        max_m = 0
    else:
        avg = sum(counts) / n
        variance = sum((c - avg) ** 2 for c in counts) / n
        std = variance ** 0.5
        min_m = min(counts)
        max_m = max(counts)

    results["classes"][cls] = {
        "display_name": CLASS_DISPLAY[cls],
        "num_words": n,
        "total_morphemes": sum(counts),
        "fertility_lambda": round(avg, 4),
        "std": round(std, 4),
        "min_morphemes": min_m,
        "max_morphemes": max_m,
        "examples": class_word_examples.get(cls, [])[:5],
    }

# ── Save outputs ─────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

out_json = os.path.join(RESULTS_DIR, "fertility_by_class.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

out_jsonl = os.path.join(RESULTS_DIR, "fertility_by_class_detail.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for row in detail_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ── Print results table ──────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("FERTILITY (λ) BY MORPHEME CLASS — Marathi (BhashaSetu)")
print(f"{len(sentences):,} sentences | {total_words:,} words | "
      f"global λ = {results['global_fertility']:.4f}")
print("=" * 78)
print(f"{'Class':<28} | {'Words':>8} | {'λ (avg)':>8} | {'σ':>6} | "
      f"{'Min':>4} | {'Max':>4}")
print("-" * 78)

for cls in CLASS_LABELS:
    d = results["classes"][cls]
    print(f"{d['display_name']:<28} | {d['num_words']:>8,} | "
          f"{d['fertility_lambda']:>8.4f} | {d['std']:>6.4f} | "
          f"{d['min_morphemes']:>4} | {d['max_morphemes']:>4}")

print("-" * 78)
print(f"{'GLOBAL':<28} | {total_words:>8,} | "
      f"{results['global_fertility']:>8.4f} |        |      |")
print("=" * 78)

# ── Print sample words per class ─────────────────────────────────────────────

print("\nSample words per class:")
for cls in CLASS_LABELS:
    examples = results["classes"][cls].get("examples", [])
    if examples:
        print(f"\n  {CLASS_DISPLAY[cls]}:")
        for ex in examples:
            morph_str = " + ".join(ex["morphemes"])
            print(f"    {ex['word']:<20} ({ex['upos']:<5}) → {morph_str}  "
                  f"[λ={ex['n_morphemes']}]")

print(f"\nResults saved: {out_json}")
print(f"Detail saved:  {out_jsonl}")
