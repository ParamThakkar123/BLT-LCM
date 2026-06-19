"""Utilities for BhashaSetu training/evaluation splits and noisy benchmarks.

The helpers centralize dataset loading for the BPE Transformer, BPE Llama and
SONAR-LCM baselines. They intentionally avoid hard-coding one column schema:
BhashaSetu-like exports used in this repository may expose parallel columns as
``source``/``target``, language names (for example ``hindi``/``marathi``), or a
nested ``translation`` dictionary.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional, Sequence


DEFAULT_DATASET = "ParamTh/BhashaSetu"
DEFAULT_FRACTIONS = (0.25, 0.50, 0.80)
DEFAULT_NOISE_LEVELS = (0.0, 0.10, 0.20)

_SRC_CANDIDATES = ("source", "src", "input", "english", "en", "hindi", "hi")
_TGT_CANDIDATES = ("target", "tgt", "output", "marathi", "mr", "translation")


@dataclass(frozen=True)
class ParallelExample:
    """One source/reference pair used by training and benchmarking scripts."""

    source: str
    target: str


def split_sentences(text: str) -> list[str]:
    """Split Indic/plain text into non-empty sentence-like chunks."""

    return [s.strip() for s in re.split(r"[.!?।]+", text.replace("\n", " ")) if s.strip()]


def add_character_noise(text: str, noise_prob: float, seed: Optional[int] = None) -> str:
    """Corrupt a percentage of non-space characters for robustness benchmarks.

    The corruption uses missing-matra deletions and Devanagari substitutions
    for Indic characters, plus ASCII substitutions otherwise. References remain
    clean; only source inputs or hypotheses should be noised by callers.
    """

    if noise_prob <= 0:
        return text
    rng = random.Random(seed)
    # Independent noise operations used by the robustness benchmark:
    # remove dependent vowel signs (missing matras) and substitute remaining
    # characters with script-appropriate alternatives.
    matras = "ािीुूृॄॅॆेैॉॊोौ्ॎॏंँः"
    devanagari = "अआइईउऊएऐओऔकखगघचछजझटठडढतथदधनपफबभमयरलवशषसह" + matras
    ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out: list[str] = []
    for ch in text:
        if ch.isspace() or rng.random() >= noise_prob:
            out.append(ch)
        elif "\u0900" <= ch <= "\u097f":
            if ch in matras and rng.random() < 0.5:
                # Missing-matra corruption: delete the vowel/diacritic mark.
                continue
            out.append(rng.choice(devanagari))
        else:
            out.append(rng.choice(ascii_chars))
    return "".join(out)


def _value(row: dict, key: str) -> str:
    val = row.get(key, "")
    if isinstance(val, dict):
        # Prefer Marathi-like target values from nested translation records.
        for nested_key in ("mr", "marathi", "target", "tgt"):
            if nested_key in val and str(val[nested_key]).strip():
                return str(val[nested_key]).strip()
        for nested_key in ("en", "english", "hi", "hindi", "source", "src"):
            if nested_key in val and str(val[nested_key]).strip():
                return str(val[nested_key]).strip()
        return ""
    return str(val).strip() if val is not None else ""


def infer_parallel_columns(
    row: dict,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Infer source/target columns from a BhashaSetu row.

    Explicit columns always win. If only a single Marathi/monolingual text column
    is available, callers can use pseudo next-sentence prediction instead of MT.
    """

    keys = set(row.keys())
    src = src_col if src_col in keys else None
    tgt = tgt_col if tgt_col in keys else None
    if src is None:
        src = next((k for k in _SRC_CANDIDATES if k in keys and _value(row, k)), None)
    if tgt is None:
        tgt = next((k for k in _TGT_CANDIDATES if k in keys and _value(row, k)), None)
    if src == tgt:
        src = next((k for k in keys if k != tgt and _value(row, k)), None)
    return src, tgt


def row_to_example(
    row: dict,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
) -> Optional[ParallelExample]:
    """Convert a dataset row to a parallel example when possible."""

    # Nested HF translation feature.
    trans = row.get("translation")
    if isinstance(trans, dict):
        src_key = src_col or next((k for k in ("en", "english", "hi", "hindi") if k in trans), None)
        tgt_key = tgt_col or next((k for k in ("mr", "marathi") if k in trans), None)
        if src_key and tgt_key:
            src = str(trans.get(src_key, "")).strip()
            tgt = str(trans.get(tgt_key, "")).strip()
            if src and tgt:
                return ParallelExample(src, tgt)

    src_key, tgt_key = infer_parallel_columns(row, src_col, tgt_col)
    if not src_key or not tgt_key:
        return None
    src = _value(row, src_key)
    tgt = _value(row, tgt_key)
    if src and tgt and src != tgt:
        return ParallelExample(src, tgt)
    return None


def load_bhashasetu_pairs(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    fraction: float = 1.0,
    max_examples: Optional[int] = None,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
    seed: int = 42,
    streaming: bool = False,
) -> list[ParallelExample]:
    """Load a deterministic fraction of BhashaSetu parallel examples."""

    from datasets import load_dataset

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    ds = load_dataset(dataset_name, split=split, streaming=streaming)
    if streaming:
        # Reservoir-like deterministic cap for streaming datasets.
        limit = max_examples or 100_000
        rows = []
        for row in ds.shuffle(seed=seed, buffer_size=10_000):
            ex = row_to_example(row, src_col, tgt_col)
            if ex is not None:
                rows.append(ex)
            if len(rows) >= int(limit * fraction):
                break
        return rows

    total = len(ds)
    take = min(total, int(total * fraction))
    if max_examples is not None:
        take = min(take, max_examples)
    ds = ds.shuffle(seed=seed).select(range(take))
    pairs = [ex for row in ds if (ex := row_to_example(row, src_col, tgt_col)) is not None]
    return pairs


def load_bhashasetu_documents(
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    fraction: float = 1.0,
    max_sent_per_doc: int = 20,
    text_col: str = "marathi",
    seed: int = 42,
) -> list[list[str]]:
    """Load a deterministic fraction of Marathi sentence documents for LCM training."""

    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    total = len(ds)
    take = min(total, int(total * fraction))
    ds = ds.shuffle(seed=seed).select(range(take))
    docs: list[list[str]] = []
    buf: list[str] = []
    for row in ds:
        text = _value(row, text_col) or _value(row, "target") or _value(row, "translation")
        if not text:
            continue
        buf.extend(split_sentences(text))
        while len(buf) >= max_sent_per_doc:
            docs.append(buf[:max_sent_per_doc])
            buf = buf[max_sent_per_doc:]
    if len(buf) >= 2:
        docs.append(buf[:max_sent_per_doc])
    return docs


def write_parallel_text(pairs: Sequence[ParallelExample], src_path: str, tgt_path: str) -> None:
    """Write source and target text files, one example per line."""

    with open(src_path, "w", encoding="utf-8") as src_f, open(tgt_path, "w", encoding="utf-8") as tgt_f:
        for ex in pairs:
            src_f.write(ex.source.replace("\n", " ") + "\n")
            tgt_f.write(ex.target.replace("\n", " ") + "\n")


def split_train_eval_documents(
    docs: list[list[str]], eval_docs: int
) -> tuple[list[list[str]], list[list[str]]]:
    """Split fraction-selected documents into training and evaluation sets."""

    if eval_docs <= 0 or len(docs) <= eval_docs:
        return docs, docs
    train_docs = docs[:-eval_docs]
    eval_split = docs[-eval_docs:]
    return train_docs, eval_split
