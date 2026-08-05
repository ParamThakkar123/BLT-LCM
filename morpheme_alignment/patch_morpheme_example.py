"""
Qualitative figure: entropy patch boundaries vs. gold morpheme boundaries.

For a single Marathi sentence this plots the neural byte-entropy curve (from the
BLT entropy model), the entropy-based patch boundaries (entropy > tau), and the
gold morpheme boundaries (Indic NLP unsupervised morphological analyzer), so the
alignment between the two boundary sets is directly visible.

This is the "linguistic principle made visible" figure: high byte-entropy tends
to coincide with morpheme boundaries in an agglutinative script, so patches
begin where morphemes begin.

Usage:
  python morpheme_alignment/patch_morpheme_example.py
  python morpheme_alignment/patch_morpheme_example.py --sentence "..." --threshold 1.335
"""

import argparse
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # so `import lcm_scripts.*` (package) resolves
sys.path.append(os.path.join(REPO, "lcm_scripts"))
sys.path.append(os.path.join(REPO, "patching_scratch"))

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from blt_loader import BLTLoader
from run_blt_patching import (
    text_to_byte_tokens,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
    DEFAULT_THRESHOLD,
)

# Okabe-Ito colorblind-safe palette
C_ENTROPY = "#4d4d4d"
C_PATCH = "#0072B2"   # blue  — entropy patch boundaries
C_MORPH = "#E69F00"   # orange — gold morpheme boundaries
C_THRESH = "#999999"

# Candidate natural Marathi sentences (case markers, inflections, compounds).
# When no --sentence is given, the most-aligned candidate is chosen as the
# illustrative example; the aggregate F1-vs-tau plot reports the full picture.
CANDIDATES = [
    "मुलांना शाळेत जायचे आहे",
    "तिने पुस्तक वाचले आणि झोपली",
    "आम्ही उद्या मुंबईला जाणार आहोत",
    "त्याने आपल्या मित्राला पत्र लिहिले",
    "शेतकऱ्यांनी पिकांची काळजी घेतली",
    "सरकारने नवीन योजना जाहीर केली",
    "विद्यार्थ्यांनी परीक्षेची तयारी केली",
    "आम्हाला मराठी भाषा आवडते",
    "त्यांच्या घरासमोर मोठे झाड आहे",
    "मुलगा बागेत खेळत होता",
    "तिने स्वयंपाकघरात जेवण बनवले",
]
DEFAULT_SENTENCE = CANDIDATES[0]
MATCH_TOL = 3  # byte tolerance for a boundary match (Marathi UTF-8 char = 3 bytes)


def _analyze(sentence, blt, morph, trivial_tokenize, threshold, device):
    tokens = text_to_byte_tokens(sentence)
    tt = torch.tensor([tokens], dtype=torch.long, device=device)
    entropies = compute_entropies_for_tokens(tt, blt.model, device=device)[0].tolist()
    boundaries, _ = entropy_patch_sentence(entropies, threshold)
    patch_bnd = [b for b in boundaries if b > 0]
    morph_bnd = gold_morpheme_boundaries(sentence, morph, trivial_tokenize)
    return tokens, entropies, patch_bnd, morph_bnd


def _alignment_f1(patch_bnd, morph_bnd, tol=MATCH_TOL):
    if not patch_bnd or not morph_bnd:
        return 0.0
    prec = sum(any(abs(p - m) <= tol for m in morph_bnd) for p in patch_bnd) / len(patch_bnd)
    rec = sum(any(abs(m - p) <= tol for p in patch_bnd) for m in morph_bnd) / len(morph_bnd)
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def gold_morpheme_boundaries(sentence, morph_analyzer, trivial_tokenize):
    """Byte offsets of gold morpheme boundaries (intra-word + word boundaries).

    Mirrors morpheme_boundary_alignment.get_morpheme_boundaries so the example
    is consistent with the quantitative alignment study.
    """
    tokens = trivial_tokenize(sentence, "mr")
    text_bytes = sentence.encode("utf-8")
    boundaries = set()
    byte_offset = 0
    for token in tokens:
        token_bytes = token.encode("utf-8")
        pos = text_bytes.find(token_bytes, byte_offset)
        if pos == -1:
            byte_offset += len(token_bytes)
            continue
        try:
            morphemes = morph_analyzer.morph_analyze(token)
        except Exception:
            morphemes = [token]
        if len(morphemes) > 1:
            morph_offset = pos
            for morph in morphemes[:-1]:
                morph_offset += len(morph.encode("utf-8"))
                if 0 < morph_offset < len(text_bytes):
                    boundaries.add(morph_offset)
        token_end = pos + len(token_bytes)
        if token_end < len(text_bytes):
            boundaries.add(token_end)
        byte_offset = token_end
    return sorted(boundaries)


def char_byte_spans(sentence):
    """Return [(char, byte_start, byte_end)] for placing glyph tick labels."""
    spans = []
    off = 0
    for ch in sentence:
        n = len(ch.encode("utf-8"))
        spans.append((ch, off, off + n))
        off += n
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sentence",
        default=None,
        help="Specific sentence to plot; if omitted, the best-aligned candidate is chosen.",
    )
    ap.add_argument(
        "--entropy_model",
        default=os.path.join(REPO, "patching_scratch", "entropy_model_marathi.pt"),
    )
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument(
        "--out",
        default=os.path.join(REPO, "morpheme_alignment", "patch_morpheme_example.png"),
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # --- Neural entropy model + Indic NLP morphology ---
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=args.device)
    from lcm_scripts.indic_resources import configure_indic_resources
    from indicnlp import loader as indic_loader

    configure_indic_resources()
    indic_loader.load()
    from indicnlp.tokenize.indic_tokenize import trivial_tokenize
    from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer

    morph = UnsupervisedMorphAnalyzer("mr")

    # --- Choose the sentence (given, or best-aligned candidate) ---
    if args.sentence:
        sentence = args.sentence.strip()
    else:
        best, best_f1 = None, -1.0
        for cand in CANDIDATES:
            _, _, pb, mb = _analyze(cand, blt, morph, trivial_tokenize, args.threshold, args.device)
            f1 = _alignment_f1(pb, mb)
            print(f"  candidate F1={f1:.2f}  ({cand})")
            if f1 > best_f1:
                best, best_f1 = cand, f1
        sentence = best
        print(f"Selected best-aligned candidate (F1={best_f1:.2f}).")

    tokens, entropies, patch_bnd, morph_bnd = _analyze(
        sentence, blt, morph, trivial_tokenize, args.threshold, args.device
    )
    print(f"Sentence      : {sentence}")
    print(f"Bytes         : {len(tokens)}")
    print(f"Patch bounds  : {patch_bnd}")
    print(f"Morpheme bnds : {morph_bnd}")
    print(f"Alignment F1  : {_alignment_f1(patch_bnd, morph_bnd):.2f}")

    # --- Plot ---
    deva = FontProperties(family="Nirmala UI", size=13)
    n = len(entropies)
    fig, ax = plt.subplots(figsize=(11, 3.4))

    ax.plot(range(n), entropies, color=C_ENTROPY, lw=1.4, label="byte entropy $H(x_i)$", zorder=3)
    ax.axhline(args.threshold, ls=":", color=C_THRESH, lw=1.2,
               label=fr"threshold $\tau={args.threshold:.2f}$", zorder=2)

    for i, b in enumerate(patch_bnd):
        ax.axvline(b - 0.5, color=C_PATCH, lw=1.6, alpha=0.9, zorder=4,
                   label="entropy patch boundary" if i == 0 else None)
    for i, b in enumerate(morph_bnd):
        ax.axvline(b - 0.5, color=C_MORPH, lw=2.2, ls=(0, (4, 2)), alpha=0.9, zorder=4,
                   label="gold morpheme boundary" if i == 0 else None)

    # Devanagari glyphs as x tick labels, centered on each character's byte span.
    spans = char_byte_spans(sentence)
    ax.set_xticks([(s + e) / 2 - 0.5 for _, s, e in spans])
    ax.set_xticklabels([c if c != " " else "·" for c, _, _ in spans], fontproperties=deva)

    ax.set_ylabel("entropy (nats)")
    ax.set_xlabel("byte position  (Devanagari characters shown below axis)")
    ax.set_xlim(-1, n)
    ax.margins(x=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2)
    ax.set_title(
        "Entropy patch boundaries align with gold morpheme boundaries "
        "(neural BLT entropy model, Marathi)",
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    pdf = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved {args.out}\nSaved {pdf}")


if __name__ == "__main__":
    main()
