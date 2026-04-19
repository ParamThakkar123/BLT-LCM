# Lightweight evaluation metric wrappers (BLEU, chrF, METEOR, TER, COMET optional)
import warnings
from typing import List, Tuple, Dict, Optional

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


def _ensure_refs(refs: List[List[str]]):
    # sacrebleu expects list of reference lists (one list per reference file)
    # We accept refs as List[str] or List[List[str]]
    if not refs:
        return []
    if isinstance(refs[0], str):
        return [refs]
    return refs


def compute_bleu(hyps: List[str], refs: List[str]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        bleu = sacrebleu.corpus_bleu(hyps, refs2)
        return float(bleu.score)
    else:
        warnings.warn("sacrebleu not installed; BLEU unavailable")
        return float("nan")


def compute_chrf(hyps: List[str], refs: List[str]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        # chrF++ corresponds to enabling word n-grams (word_order=2).
        chrf = sacrebleu.corpus_chrf(hyps, refs2, word_order=2)
        return float(chrf.score)
    else:
        warnings.warn("sacrebleu not installed; chrF unavailable")
        return float("nan")


def compute_ter(hyps: List[str], refs: List[str]) -> float:
    refs2 = _ensure_refs(refs)
    if _HAS_SACREBLEU:
        ter = sacrebleu.corpus_ter(hyps, refs2)
        return float(ter.score)
    else:
        warnings.warn("sacrebleu not installed; TER unavailable")
        return float("nan")


def compute_meteor(hyps: List[str], refs: List[str]) -> float:
    if not _HAS_NLTK:
        warnings.warn("nltk not installed; METEOR unavailable")
        return float("nan")
    scores = []
    for h, r in zip(hyps, refs):
        try:
            scores.append(meteor_score.single_meteor_score(r.split(), h.split()))
        except Exception:
            scores.append(0.0)
    return float(100.0 * sum(scores) / max(1, len(scores)))


def compute_comet(
    hyps: List[str],
    refs: List[str],
    model_name: Optional[str] = None,
    model_obj: Optional[object] = None,
) -> float:
    if not _HAS_COMET:
        warnings.warn("comet not installed; COMET unavailable")
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
            {"src": "", "mt": h, "ref": r} for h, r in zip(hyps, refs)
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
    refs: List[str],
    comet_model_name: Optional[str] = None,
    comet_model_obj: Optional[object] = None,
) -> Dict[str, float]:
    # refs can be List[str] (single ref) or List[List[str]]
    # expand refs to single list when computing metrics that expect parallel lists
    if isinstance(refs[0], list):
        # choose first reference for metrics that do not support multiple refs
        refs_single = [r[0] for r in zip(*refs)] if refs else []
    else:
        refs_single = refs

    out = {}
    out["BLEU"] = compute_bleu(hyps, refs)
    out["chrF++"] = compute_chrf(hyps, refs)
    out["TER"] = compute_ter(hyps, refs)
    out["METEOR"] = compute_meteor(hyps, refs_single)
    # COMET skipped by default
    out["COMET"] = compute_comet(
        hyps, refs, model_name=comet_model_name, model_obj=comet_model_obj
    )
    return out


if __name__ == "__main__":
    print("eval_metrics module. Use compute_all(hyps, refs)")
