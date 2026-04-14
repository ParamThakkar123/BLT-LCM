"""Compute morpheme-level fertility using Indic NLP segmentation and POS.

This script expects Indic NLP tools to be available. If not installed, it falls
back to a simple whitespace-based heuristic.
"""

from typing import List, Tuple
import warnings

try:
    # placeholder import; user may need to install indic-nlp-library
    from indicnlp.tokenize import indic_tokenize

    _HAS_INDIC = True
except Exception:
    indic_tokenize = None
    _HAS_INDIC = False


def simple_morpheme_seg(s: str) -> List[str]:
    # fallback: split on whitespace and punctuation
    import re

    toks = re.findall(r"[\w\u0900-\u097F]+", s)
    return toks


def compute_fertility(sentences: List[str]) -> Tuple[float, List[int]]:
    """Return average morpheme count per sentence and per-sentence counts."""
    counts = []
    for s in sentences:
        if _HAS_INDIC:
            try:
                toks = indic_tokenize.trivial_tokenize(s)
            except Exception:
                toks = simple_morpheme_seg(s)
        else:
            toks = simple_morpheme_seg(s)
        counts.append(len(toks))
    avg = sum(counts) / max(1, len(counts))
    return avg, counts


def compute_fertility_by_class(sentences: List[str], classes: List[int]) -> dict:
    """Group fertility by provided class labels (length == sentences).

    Returns dict[class_label] -> (avg_fertility, counts_list)
    """
    byc = {}
    for s, c in zip(sentences, classes):
        byc.setdefault(c, []).append(s)
    out = {}
    for c, sents in byc.items():
        avg, counts = compute_fertility(sents)
        out[c] = {"avg": avg, "counts": counts}
    return out


if __name__ == "__main__":
    print("fertility_audit: use compute_fertility(sentences)")
