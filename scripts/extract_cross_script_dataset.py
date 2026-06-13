"""Extract a small second-script sanity-check corpus for BLT entropy patching.

Default usage creates a 100-sentence Hindi (Devanagari, non-Latin) JSON file:

    python scripts/extract_cross_script_dataset.py

The output is intentionally small so the entropy-boundary alignment pipeline can
be rerun as a cross-script smoke test without committing a large dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

DEFAULT_DATASET = "cfilt/iitb-english-hindi"
DEFAULT_SPLIT = "test"
DEFAULT_LANGUAGE = "hindi"
DEFAULT_OUTPUT = Path("data/cross_script/hindi_sentences_100.json")
LANGUAGE_KEYS = {
    "hindi": ("hi", "hindi", "Hindi"),
    "tamil": ("ta", "tamil", "Tamil"),
}


def extract_sentence(row: dict[str, Any], language: str, text_column: str | None = None) -> str | None:
    """Return one sentence from a dataset row for the requested language."""
    if text_column:
        value = row.get(text_column)
        return _clean_text(value)

    keys = LANGUAGE_KEYS.get(language.lower(), (language.lower(),))
    for key in keys:
        value = row.get(key)
        text = _clean_text(value)
        if text:
            return text

    translation = row.get("translation")
    if isinstance(translation, dict):
        for key in keys:
            text = _clean_text(translation.get(key))
            if text:
                return text

    for value in row.values():
        text = _clean_text(value)
        if text and _looks_non_latin(text):
            return text

    return None


def collect_sentences(rows: Iterable[dict[str, Any]], language: str, limit: int, text_column: str | None = None) -> list[str]:
    """Collect up to ``limit`` unique non-empty sentences from an iterable of rows."""
    sentences: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sentence = extract_sentence(row, language=language, text_column=text_column)
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
        if len(sentences) >= limit:
            break
    return sentences


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _looks_non_latin(text: str) -> bool:
    return any(ord(char) > 0x024F and char.isalpha() for char in text)


def _read_local_rows(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return

    payload = json.loads(text)
    if isinstance(payload, list):
        for item in payload:
            yield {"text": item} if isinstance(item, str) else item
        return
    raise ValueError(f"Unsupported local source format in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a 100-sentence Hindi/Tamil sanity-check corpus.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id to stream from.")
    parser.add_argument("--config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to read.")
    parser.add_argument("--language", choices=sorted(LANGUAGE_KEYS), default=DEFAULT_LANGUAGE)
    parser.add_argument("--text-column", default=None, help="Optional explicit sentence column; otherwise common language/translation columns are inferred.")
    parser.add_argument("--limit", type=int, default=100, help="Number of sentences to write.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--source-file", type=Path, default=None, help="Optional local JSON/JSONL source file for offline extraction.")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True, help="Stream the dataset instead of downloading a full local copy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_file:
        ds = _read_local_rows(args.source_file)
    else:
        load_kwargs: dict[str, Any] = {"split": args.split, "streaming": args.streaming}
        ds = load_dataset(args.dataset, args.config, **load_kwargs) if args.config else load_dataset(args.dataset, **load_kwargs)
    sentences = collect_sentences(ds, language=args.language, limit=args.limit, text_column=args.text_column)
    if len(sentences) < args.limit:
        raise RuntimeError(f"Only found {len(sentences)} {args.language} sentences; expected {args.limit}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(sentences)} {args.language} sentences to {args.output}")


if __name__ == "__main__":
    main()
