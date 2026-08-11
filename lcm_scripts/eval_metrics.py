# Lightweight evaluation metric wrappers (BLEU, chrF, METEOR, TER, COMET optional)
import warnings
from typing import List, Tuple, Dict, Optional, Union, Sequence

try:
    import sacrebleu

    _HAS_SACREBLEU = True
except Exception:
    sacrebleu = None
    _HAS_SACREBLEU = False

try:
    from nltk.translate import meteor_score

    _HAS_NLTK = True
except Exception:
    meteor_score = None
    _HAS_NLTK = False

try:
    # comet (unbabel-comet) is optional; if present we attempt to load utilities
    import comet

    try:
        from comet import download_model
    except Exception:
        download_model = None
    try:
        from comet.models import load_from_checkpoint
    except Exception:
        load_from_checkpoint = None
    _HAS_COMET = True
except Exception:
    comet = None
    download_model = None
    load_from_checkpoint = None
    _HAS_COMET = False


def _ensure_refs(refs: Union[List[str], List[List[str]]]) -> List[List[str]]:
    if not refs:
        return []
    if isinstance(refs[0], str):
        return [refs]
    return refs


def compute_bleu(hyps: List[str], refs: Union[List[str], List[List[str]]]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        bleu = sacrebleu.corpus_bleu(hyps, refs2)
        return float(bleu.score)
    else:
        warnings.warn("sacrebleu not installed; BLEU unavailable")
        return float("nan")


def compute_chrf(hyps: List[str], refs: Union[List[str], List[List[str]]]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        # chrF++ corresponds to enabling word n-grams (word_order=2).
        chrf = sacrebleu.corpus_chrf(hyps, refs2, word_order=2)
        return float(chrf.score)
    else:
        warnings.warn("sacrebleu not installed; chrF unavailable")
        return float("nan")


def compute_ter(hyps: List[str], refs: Union[List[str], List[List[str]]]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        ter = sacrebleu.corpus_ter(hyps, refs2)
        return float(ter.score)
    else:
        warnings.warn("sacrebleu not installed; TER unavailable")
        return float("nan")


def _ensure_wordnet() -> bool:
    """Make sure NLTK's WordNet corpus is present (METEOR needs it)."""
    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet.zip")
            return True
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.data.find("corpora/wordnet.zip")
            return True
    except Exception as e:
        warnings.warn(f"Could not provision NLTK WordNet for METEOR: {e}")
        return False


def compute_meteor(hyps: List[str], refs: List[str]) -> float:
    """METEOR via NLTK.

    NOTE: NLTK's METEOR uses the *English* WordNet for its synonymy-matching
    stage. On Devanagari output the synonym/stem modules contribute nothing, so
    this degrades to a bare unigram alignment score and is not comparable to
    METEOR as reported for English MT. Prefer chrF++ / COMET for Marathi; treat
    this number as a weak diagnostic only.
    """
    if not _HAS_NLTK:
        warnings.warn("nltk not installed; METEOR unavailable")
        return float("nan")
    if not _ensure_wordnet():
        warnings.warn("NLTK WordNet unavailable; METEOR reported as NaN")
        return float("nan")

    scores = []
    n_failed = 0
    first_error: Optional[BaseException] = None
    for h, r in zip(hyps, refs):
        try:
            scores.append(meteor_score.single_meteor_score(r.split(), h.split()))
        except Exception as e:  # do NOT silently fold failures into 0.0
            n_failed += 1
            if first_error is None:
                first_error = e

    if n_failed:
        warnings.warn(
            f"METEOR failed on {n_failed}/{len(hyps)} segments "
            f"(first error: {first_error!r})"
        )
    if not scores:
        # Every segment failed. Returning 0.0 here would be indistinguishable
        # from a genuinely terrible system, so report NaN instead.
        return float("nan")
    return float(100.0 * sum(scores) / len(scores))


def compute_comet(
    hyps: List[str],
    refs: List[str],
    srcs: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    model_obj: Optional[object] = None,
) -> float:
    """COMET system score.

    ``srcs`` is REQUIRED. The standard checkpoints (e.g. ``Unbabel/wmt22-comet-da``)
    are source-aware: they encode (src, mt, ref) jointly. Passing empty strings as
    sources does not "skip" the source -- it feeds the model an out-of-distribution
    input and yields numbers that look plausible but are not COMET scores. We
    therefore refuse to score rather than emit an invalid value.
    """
    if not _HAS_COMET:
        warnings.warn("comet not installed; COMET unavailable")
        return float("nan")

    if srcs is None:
        warnings.warn(
            "COMET requires source sentences (srcs=...). Scoring with empty "
            "sources produces invalid scores, so COMET is reported as NaN. "
            "Pass the source side of the eval set to compute_all(...)."
        )
        return float("nan")
    if len(srcs) != len(hyps):
        warnings.warn(
            f"COMET: got {len(srcs)} sources for {len(hyps)} hypotheses; "
            "refusing to score misaligned data"
        )
        return float("nan")

    model = None
    if model_obj is not None:
        model = model_obj
    else:
        if model_name is None:
            warnings.warn("No COMET model name/path provided; skipping COMET")
            return float("nan")
        # download_model returns a *path*, then load_from_checkpoint loads it
        try:
            if download_model is not None and load_from_checkpoint is not None:
                model_path = download_model(model_name)
                model = load_from_checkpoint(model_path)
        except Exception:
            model = None
        if model is None and load_from_checkpoint is not None:
            try:
                model = load_from_checkpoint(model_name)
            except Exception:
                model = None

    if model is None:
        warnings.warn("Failed to load COMET model; skipping COMET")
        return float("nan")

    try:
        samples = [
            {"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)
        ]
        res = model.predict(samples, batch_size=32, gpus=0)

        if isinstance(res, tuple):
            # comet returns (scores_list, system_score)
            return float(res[1]) if len(res) > 1 else float("nan")
        if isinstance(res, dict) and "system_score" in res:
            return float(res["system_score"])
        if isinstance(res, dict) and "scores" in res:
            scores = res["scores"]
            return float(sum(scores) / len(scores))
    except Exception as e:
        warnings.warn(f"COMET prediction failed: {e}")
        return float("nan")

    warnings.warn("COMET returned unexpected result format")
    return float("nan")


def compute_all(
    hyps: List[str],
    refs: Union[List[str], List[List[str]]],
    srcs: Optional[List[str]] = None,
    comet_model_name: Optional[str] = None,
    comet_model_obj: Optional[object] = None,
    include_meteor: bool = True,
) -> Dict[str, float]:
    """Compute the full metric suite.

    ``srcs`` should be the *clean* source sentences of the eval set. It is only
    used by COMET, which is source-aware; without it COMET is reported as NaN
    rather than silently scored against empty sources.
    """
    # METEOR/COMET require List[str] refs; extract first reference if multi-ref
    if refs and isinstance(refs[0], list):
        refs_flat: List[str] = [r[0] for r in refs]
    else:
        refs_flat = refs  # type: ignore[assignment]

    out = {}
    out["BLEU"] = compute_bleu(hyps, refs)
    out["chrF++"] = compute_chrf(hyps, refs)
    out["TER"] = compute_ter(hyps, refs)
    if include_meteor:
        out["METEOR"] = compute_meteor(hyps, refs_flat)
    out["COMET"] = compute_comet(
        hyps,
        refs_flat,
        srcs=srcs,
        model_name=comet_model_name,
        model_obj=comet_model_obj,
    )
    return out


if __name__ == "__main__":
    from device_utils import report_device

    print("eval_metrics module. Use compute_all(hyps, refs)")
    # BLEU/chrF++/TER/METEOR are CPU string metrics; COMET runs a model here.
    report_device(label="COMET", warn_cpu=False)
