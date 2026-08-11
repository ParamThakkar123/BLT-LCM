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

import argparse
import sys
import io
import json
import os
import re
import logging
import time
from collections import defaultdict

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcm_scripts.checkpoint_utils import ResumableJsonl, config_fingerprint
from lcm_scripts.device_utils import report_device

# ── Logging setup ─────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "fertility_audit.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("fertility_audit")
logger.info("=" * 60)
logger.info("Fertility audit script started")
logger.info("=" * 60)

# ── CLI ──────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--resume",
    default=os.environ.get("FERTILITY_RESUME", "auto"),
    metavar="auto|never",
    help="'auto' (default) continues from a partial per-word detail JSONL whose "
    "configuration matches; 'never' discards it and starts over.",
)
ARGS = _parser.parse_args()

# ── Indic NLP setup ──────────────────────────────────────────────────────────

from lcm_scripts.indic_resources import configure_indic_resources
from indicnlp import loader

logger.info("Configuring Indic NLP resources…")
configure_indic_resources()
loader.load()
from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

morph_analyzer = UnsupervisedMorphAnalyzer("mr")

# ── Stanza setup ─────────────────────────────────────────────────────────────

import stanza

# Stanza's POS tagger is the one neural component here and picks up a GPU on its
# own; the Indic NLP morphology below is pure python.
report_device(logger=logger, label="Stanza POS", warn_cpu=False)

logger.info("Downloading Stanza model for Marathi (if needed)…")
stanza.download("mr", verbose=False)
logger.info("Loading Stanza pipeline (tokenize, pos, lemma)…")
nlp = stanza.Pipeline("mr", processors="tokenize,pos,lemma", verbose=False)
logger.info("Stanza pipeline ready")

# ── Morpheme class definitions ───────────────────────────────────────────────

# UPOS tags → morpheme class mapping
NOUN_TAGS = {"NOUN", "PROPN"}
VERB_TAGS = {"VERB", "AUX"}
POSTP_TAGS = {"ADP"}

# Devanagari virama (halant) — used to detect conjunct clusters in compounds
VIRAMA = "\u094d"


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
# How many Marathi sentences to materialize if the corpus file is missing.
CORPUS_GEN_SIZE = int(os.environ.get("FERTILITY_CORPUS_SIZE", "20000"))


def _generate_corpus_from_bhashasetu(path: str, num_sentences: int) -> list:
    """Build ``marathi_sentences.json`` from BhashaSetu so the audit is
    reproducible from the repository alone (the corpus file is not tracked).

    Streams the Marathi column deterministically and caches the result to
    ``path`` for subsequent runs.
    """
    from datasets import load_dataset

    logger.info(
        "Corpus file not found; generating %d Marathi sentences from "
        "ParamTh/BhashaSetu (one-time, cached to %s)…",
        num_sentences,
        path,
    )
    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    collected = []
    for row in ds:
        text = (row.get("marathi") or "").strip()
        if len(text) > 5:
            collected.append(text)
        if len(collected) >= num_sentences:
            break
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(collected, fh, ensure_ascii=False)
    logger.info("Generated and cached %d sentences to %s", len(collected), path)
    return collected


logger.info("Loading corpus from %s…", CORPUS_PATH)
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        all_sentences = json.load(f)
else:
    all_sentences = _generate_corpus_from_bhashasetu(CORPUS_PATH, CORPUS_GEN_SIZE)
logger.info("Corpus loaded: %d sentences total", len(all_sentences))

# Use a sample for efficiency — Stanza POS tagging is slower than byte-level ops.
# 5000 sentences gives statistically robust estimates; increase if needed.
SAMPLE_SIZE = min(5000, len(all_sentences))
sentences = all_sentences[:SAMPLE_SIZE]
logger.info("Processing sample: %s / %s sentences", SAMPLE_SIZE, len(all_sentences))

# ── Process sentences ────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FINGERPRINT = config_fingerprint(
    {
        "corpus": CORPUS_PATH,
        "sample_size": SAMPLE_SIZE,
        "class_labels": CLASS_LABELS,
    }
)

BATCH_SIZE = 50  # Stanza batch size for efficiency
logger.info("Beginning sentence processing (batch size=%d)…", BATCH_SIZE)
batch_t0 = time.time()

# Stanza tagging plus morphological analysis runs at a few sentences per second,
# so a 5000-sentence audit is a long job. Results are grouped one row per
# sentence and streamed to a progress file, so an interrupted audit resumes at
# the first sentence it had not written. The flat per-word JSONL that downstream
# scripts consume is rebuilt from those rows below.
progress = ResumableJsonl(
    os.path.join(RESULTS_DIR, "fertility_audit_progress.jsonl"),
    fingerprint=FINGERPRINT,
    resume=ARGS.resume != "never",
    key="sentence_id",
)
if progress.done:
    logger.info(
        "Resuming audit: %d/%d sentences already processed",
        len(progress.done),
        len(sentences),
    )

for batch_start in range(0, len(sentences), BATCH_SIZE):
    batch_idx = [
        i
        for i in range(batch_start, min(batch_start + BATCH_SIZE, len(sentences)))
        if not progress.is_done(i)
    ]
    if not batch_idx:
        continue
    batch = [sentences[i] for i in batch_idx]

    # Stanza accepts a list of strings in one call, returning list[Document]
    docs = nlp(batch)

    for doc_idx, doc in enumerate(docs):
        sent_idx = batch_idx[doc_idx]
        words = []
        for sent in doc.sentences:
            for word in sent.words:
                text = word.text
                upos = word.upos if word.upos else "X"

                # Skip punctuation and symbols
                if upos in ("PUNCT", "SYM", "X") or not re.search(
                    r"[\u0900-\u097F]", text
                ):
                    continue

                # Morpheme segmentation via Indic NLP
                try:
                    morphemes = morph_analyzer.morph_analyze(text)
                except Exception:
                    morphemes = [text]

                words.append(
                    {
                        "word": text,
                        "upos": upos,
                        "morphemes": morphemes,
                        "n_morphemes": len(morphemes),
                        "class": classify_word(upos, text, morphemes),
                    }
                )
        # One row per sentence, emitted even when no word qualifies, so those
        # sentences are not re-tagged on every resume.
        progress.append({"sentence_id": sent_idx, "words": words})

    # Progress
    processed = min(batch_start + BATCH_SIZE, len(sentences))
    if processed % 500 == 0 or processed == len(sentences):
        elapsed = time.time() - batch_t0
        rate = processed / elapsed if elapsed > 0 else 0
        logger.info(
            "Processed %s/%s sentences (%.1f sent/s)",
            processed,
            len(sentences),
            rate,
        )

progress.close()

# ── Flatten into the per-word records the rest of the pipeline expects ───────

# Accumulators per class
class_morpheme_counts = defaultdict(list)  # class -> list of morpheme counts per word
class_word_examples = defaultdict(list)  # class -> sample words (capped)

detail_rows = []  # per-word JSONL records
total_words = 0
total_morphemes = 0

for row in progress.all_records():
    sent_idx = row["sentence_id"]
    for w in row["words"]:
        cls = w["class"]
        class_morpheme_counts[cls].append(w["n_morphemes"])
        total_words += 1
        total_morphemes += w["n_morphemes"]

        # Keep a few examples per class
        if len(class_word_examples[cls]) < 20:
            class_word_examples[cls].append(
                {k: w[k] for k in ("word", "upos", "morphemes", "n_morphemes")}
            )

        detail_rows.append({"sentence_id": sent_idx, **w})

logger.info(
    "Sentence processing complete (%d words in %.1fs)",
    total_words,
    time.time() - batch_t0,
)

# ── Compute fertility per class ──────────────────────────────────────────────

logger.info("Computing fertility statistics per class…")
compute_t0 = time.time()
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
        std = variance**0.5
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

logger.info("Statistics computed in %.3fs", time.time() - compute_t0)

# ── Save outputs ─────────────────────────────────────────────────────────────

out_json = os.path.join(RESULTS_DIR, "fertility_by_class.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
logger.info("Saved JSON results to %s", out_json)

out_jsonl = os.path.join(RESULTS_DIR, "fertility_by_class_detail.jsonl")
with open(out_jsonl, "w", encoding="utf-8") as f:
    for row in detail_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
logger.info("Saved JSONL detail (%d rows) to %s", len(detail_rows), out_jsonl)

# ── Print results table ──────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("FERTILITY (λ) BY MORPHEME CLASS — Marathi (BhashaSetu)")
print(
    f"{len(sentences):,} sentences | {total_words:,} words | "
    f"global λ = {results['global_fertility']:.4f}"
)
print("=" * 78)
print(
    f"{'Class':<28} | {'Words':>8} | {'λ (avg)':>8} | {'σ':>6} | "
    f"{'Min':>4} | {'Max':>4}"
)
print("-" * 78)

for cls in CLASS_LABELS:
    d = results["classes"][cls]
    print(
        f"{d['display_name']:<28} | {d['num_words']:>8,} | "
        f"{d['fertility_lambda']:>8.4f} | {d['std']:>6.4f} | "
        f"{d['min_morphemes']:>4} | {d['max_morphemes']:>4}"
    )

print("-" * 78)
print(
    f"{'GLOBAL':<28} | {total_words:>8,} | "
    f"{results['global_fertility']:>8.4f} |        |      |"
)
print("=" * 78)

# ── Print sample words per class ─────────────────────────────────────────────

print("\nSample words per class:")
for cls in CLASS_LABELS:
    examples = results["classes"][cls].get("examples", [])
    if examples:
        print(f"\n  {CLASS_DISPLAY[cls]}:")
        for ex in examples:
            morph_str = " + ".join(ex["morphemes"])
            print(
                f"    {ex['word']:<20} ({ex['upos']:<5}) → {morph_str}  "
                f"[λ={ex['n_morphemes']}]"
            )

logger.info("Results saved: %s", out_json)
logger.info("Detail saved:  %s", out_jsonl)
logger.info("Log file:      %s", log_path)
logger.info("=" * 60)
logger.info("Fertility audit script finished")
logger.info("=" * 60)
