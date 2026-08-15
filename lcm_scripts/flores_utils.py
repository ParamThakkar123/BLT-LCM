"""FLORES-200 loading, so results are comparable to published systems.

Every number in this repository was previously computed on an ad-hoc BhashaSetu
split, which no other paper reports on -- a reader cannot tell whether a chrF++
of 42 is good. FLORES-200 is the standard evaluation set for the 200 languages
NLLB covers, including 24 Indic languages, and it is what IndicTrans2, NLLB and
every recent Indic MT paper report on.

The two splits are ``dev`` (997 sentences, for tuning) and ``devtest`` (1012,
for reporting). FLORES has no train split by design: it is n-way parallel
evaluation data only, so there is nothing here to train on and no risk of the
test set leaking into training.

Language codes are FLORES's ``<iso639-3>_<script>`` form (``mar_Deva``,
``eng_Latn``). Friendly aliases are accepted everywhere: ``mr``, ``marathi``
and ``mar_Deva`` all resolve to the same column.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence

from bhashasetu_utils import ParallelExample

# Hub ids that carry FLORES-200, newest first. `mteb/flores` stores every
# language as a column of one table, which is what makes n-way pivoting cheap.
FLORES_DATASET_CANDIDATES = (
    "mteb/flores",
    "openlanguagedata/flores_plus",
    "facebook/flores",
)

DEFAULT_SPLIT = "devtest"

# The Indic languages FLORES-200 covers, with the script each is written in.
# Used by the multilingual generalization experiment: a byte-level model should
# transfer across these far better than a per-language BPE vocabulary does.
INDIC_LANGUAGES: dict[str, str] = {
    "asm_Beng": "Assamese",
    "awa_Deva": "Awadhi",
    "ben_Beng": "Bengali",
    "bho_Deva": "Bhojpuri",
    "guj_Gujr": "Gujarati",
    "hin_Deva": "Hindi",
    "hne_Deva": "Chhattisgarhi",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri (Arabic)",
    "kas_Deva": "Kashmiri (Devanagari)",
    "mag_Deva": "Magahi",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mni_Beng": "Manipuri",
    "npi_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "snd_Arab": "Sindhi",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
}

# A default multilingual panel: same script as Marathi (Devanagari) plus three
# other scripts, so a byte-level claim is tested across writing systems rather
# than only within one.
DEFAULT_INDIC_PANEL = (
    "mar_Deva",  # the paper's primary language
    "hin_Deva",  # same script, high resource
    "npi_Deva",  # same script, lower resource
    "ben_Beng",  # different script, related family
    "tam_Taml",  # different script, different family (Dravidian)
    "guj_Gujr",  # different script, neighbouring language
)

_ALIASES: dict[str, str] = {
    "en": "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
    "mr": "mar_Deva", "mar": "mar_Deva", "marathi": "mar_Deva",
    "hi": "hin_Deva", "hin": "hin_Deva", "hindi": "hin_Deva",
    "bn": "ben_Beng", "ben": "ben_Beng", "bengali": "ben_Beng",
    "gu": "guj_Gujr", "guj": "guj_Gujr", "gujarati": "guj_Gujr",
    "ta": "tam_Taml", "tam": "tam_Taml", "tamil": "tam_Taml",
    "te": "tel_Telu", "tel": "tel_Telu", "telugu": "tel_Telu",
    "kn": "kan_Knda", "kan": "kan_Knda", "kannada": "kan_Knda",
    "ml": "mal_Mlym", "mal": "mal_Mlym", "malayalam": "mal_Mlym",
    "pa": "pan_Guru", "pan": "pan_Guru", "punjabi": "pan_Guru",
    "or": "ory_Orya", "ory": "ory_Orya", "odia": "ory_Orya",
    "as": "asm_Beng", "asm": "asm_Beng", "assamese": "asm_Beng",
    "ne": "npi_Deva", "npi": "npi_Deva", "nepali": "npi_Deva",
    "ur": "urd_Arab", "urd": "urd_Arab", "urdu": "urd_Arab",
    "sa": "san_Deva", "san": "san_Deva", "sanskrit": "san_Deva",
    "sd": "snd_Arab", "snd": "snd_Arab", "sindhi": "snd_Arab",
}


class FloresUnavailable(RuntimeError):
    """FLORES could not be loaded from any known hub id or the local cache."""


def resolve_lang(name: str) -> str:
    """Map ``mr`` / ``marathi`` / ``mar_Deva`` to the FLORES column name."""
    if not name:
        raise ValueError("empty language name")
    key = name.strip()
    if "_" in key:  # already a FLORES code
        return key
    return _ALIASES.get(key.lower(), key)


def language_name(code: str) -> str:
    """Human-readable name for a FLORES code, for figure titles."""
    code = resolve_lang(code)
    if code == "eng_Latn":
        return "English"
    return INDIC_LANGUAGES.get(code, code)


@functools.lru_cache(maxsize=4)
def _load_split(split: str, dataset: Optional[str] = None):
    """Load one FLORES split, trying each known hub id in turn.

    Cached: the multilingual sweep asks for the same split once per language
    pair, and re-reading a 200-column table each time is pure waste.
    """
    from datasets import load_dataset

    candidates = (dataset,) if dataset else FLORES_DATASET_CANDIDATES
    errors = []
    for name in candidates:
        for config in (None, "default", "all"):
            try:
                ds = (
                    load_dataset(name, config, split=split)
                    if config
                    else load_dataset(name, split=split)
                )
                return ds
            except Exception as e:  # try the next id/config
                errors.append(f"{name}[{config}]: {type(e).__name__}: {e}")
    raise FloresUnavailable(
        "Could not load FLORES-200 from any of "
        f"{candidates}. Tried:\n  " + "\n  ".join(errors[:6])
    )


def available_languages(
    split: str = DEFAULT_SPLIT, dataset: Optional[str] = None
) -> list[str]:
    """Every language column present in the loaded FLORES split."""
    return sorted(_load_split(split, dataset).column_names)


def load_flores_pairs(
    src_lang: str = "eng_Latn",
    tgt_lang: str = "mar_Deva",
    split: str = DEFAULT_SPLIT,
    max_examples: Optional[int] = None,
    dataset: Optional[str] = None,
) -> list[ParallelExample]:
    """Load one FLORES language pair as ``ParallelExample`` records.

    FLORES is n-way parallel: the same 1012 sentences exist in all 200
    languages, so any (src, tgt) pair is a genuine parallel corpus and every
    language pair is scored on identical content. That is exactly what makes
    the cross-language comparison in the multilingual experiment meaningful --
    a difference between languages is a difference in the model, not in the
    difficulty of the sentences.
    """
    src = resolve_lang(src_lang)
    tgt = resolve_lang(tgt_lang)
    ds = _load_split(split, dataset)

    missing = [c for c in (src, tgt) if c not in ds.column_names]
    if missing:
        raise ValueError(
            f"FLORES split '{split}' has no column(s) {missing}. "
            f"{len(ds.column_names)} languages available; "
            f"e.g. {sorted(ds.column_names)[:5]}"
        )

    pairs: list[ParallelExample] = []
    for row in ds:
        s = (row.get(src) or "").strip()
        t = (row.get(tgt) or "").strip()
        if s and t:
            pairs.append(ParallelExample(s, t))
        if max_examples is not None and len(pairs) >= max_examples:
            break
    if not pairs:
        raise ValueError(f"No usable {src}->{tgt} pairs in FLORES '{split}'")
    return pairs


def load_flores_panel(
    languages: Sequence[str] = DEFAULT_INDIC_PANEL,
    src_lang: str = "eng_Latn",
    split: str = DEFAULT_SPLIT,
    max_examples: Optional[int] = None,
    dataset: Optional[str] = None,
) -> dict[str, list[ParallelExample]]:
    """Load several target languages at once, keyed by FLORES code."""
    out: dict[str, list[ParallelExample]] = {}
    for lang in languages:
        code = resolve_lang(lang)
        try:
            out[code] = load_flores_pairs(
                src_lang, code, split, max_examples, dataset
            )
        except Exception as e:
            print(f"[flores] skipping {code}: {e}")
    return out


def add_flores_args(parser, *, default_tgt: str = "mar_Deva") -> None:
    """Add the standard FLORES selection flags to a script's parser."""
    parser.add_argument(
        "--flores_split",
        default=DEFAULT_SPLIT,
        choices=["dev", "devtest"],
        help="FLORES-200 split. Report on 'devtest'; tune on 'dev'.",
    )
    parser.add_argument(
        "--flores_src",
        default="eng_Latn",
        help="Source language (FLORES code, or an alias such as 'en').",
    )
    parser.add_argument(
        "--flores_tgt",
        default=default_tgt,
        help="Target language (FLORES code, or an alias such as 'mr').",
    )
    parser.add_argument(
        "--flores_dataset",
        default=None,
        help="Override the FLORES hub id (default: try mteb/flores, then "
        "openlanguagedata/flores_plus, then facebook/flores).",
    )
    parser.add_argument(
        "--flores_max_examples",
        type=int,
        default=None,
        help="Cap the number of FLORES sentences (smoke tests only -- a "
        "partial devtest is not comparable to published numbers).",
    )
