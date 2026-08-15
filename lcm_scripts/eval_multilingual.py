"""Cross-language transfer of the byte entropy model and its patching.

Two reviewer questions, one experiment, because they share a measurement.

**Does the patcher transfer across languages and scripts?** The entropy model
was trained on Marathi. Every other language it is asked to patch -- including
the English *source* side of the translation task -- is out of distribution.
The code says so in a comment; this script measures it. For each language it
reports bits-per-byte under the Marathi entropy model and the patch statistics
that follow (patches per sentence, bytes per patch, compression ratio). A model
that transfers shows similar bits-per-byte across Devanagari languages and
degrades gracefully on other scripts; one that does not shows the patcher
falling back to near-byte-level segmentation, which is exactly the failure the
compression ratio makes visible.

**Does the approach generalize beyond Marathi?** FLORES-200 is n-way parallel,
so all languages here are scored on *the same 1012 sentences*. A difference
between languages is therefore a property of the model, not of the test set --
which is what makes a cross-language claim meaningful at all.

The English row matters most: it is the source side the MT model actually
encodes, so its bits-per-byte is a direct measurement of the mismatch the
paper needs to own.

Usage:
  python lcm_scripts/eval_multilingual.py \
      --entropy_model patching_scratch/entropy_model_marathi.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm

from device_utils import report_device
from flores_utils import (
    DEFAULT_INDIC_PANEL,
    add_flores_args,
    language_name,
    load_flores_pairs,
    resolve_lang,
)
from plot_utils import (
    add_plot_args,
    plot_formats,
    plot_grouped_bars,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from checkpoint_utils import StageTracker, add_resume_args, config_fingerprint

from run_blt_patching import (
    DEFAULT_THRESHOLD,
    VOCAB_SIZE,
    ByteEntropyModel,
    compute_entropies_batched,
    entropy_patch_sentence,
    text_to_byte_tokens,
)

# The language the entropy model was trained on. Every other row is measured
# relative to this one.
TRAINING_LANGUAGE = "mar_Deva"


def load_entropy_model(path: str, device: str) -> ByteEntropyModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = ByteEntropyModel(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=cfg.get("dim", 256),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 4),
        max_seqlen=cfg.get("max_seqlen", 512),
        ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", 1.3),
        attn_window=cfg.get("attn_window", None),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def measure_language(
    model, texts: list[str], threshold: float, device: str, batch_size: int, desc: str
) -> dict:
    """Bits-per-byte and patch statistics for one language."""
    total_bytes = 0
    total_patches = 0
    total_entropy = 0.0
    patch_lengths: list[int] = []
    per_sentence_patches: list[int] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        chunk = texts[start : start + batch_size]
        token_lists = [text_to_byte_tokens(t) for t in chunk]
        token_lists = [t for t in token_lists if t]
        if not token_lists:
            continue
        entropies = compute_entropies_batched(token_lists, model, device=device)
        for tokens, ent in zip(token_lists, entropies):
            if not ent:
                continue
            # The entropy model's own output IS the quantity of interest: mean
            # nats/byte, converted to bits/byte, is how surprised a Marathi
            # model is by this language.
            total_entropy += sum(ent)
            total_bytes += len(tokens)
            boundaries, lengths = entropy_patch_sentence(ent, threshold)
            total_patches += len(lengths)
            patch_lengths.extend(lengths)
            per_sentence_patches.append(len(lengths))

    if not total_bytes:
        return {}
    nats_per_byte = total_entropy / total_bytes
    return {
        "sentences": len(per_sentence_patches),
        "bytes": total_bytes,
        "patches": total_patches,
        "nats_per_byte": nats_per_byte,
        "bits_per_byte": nats_per_byte / math.log(2),
        "bytes_per_patch": total_bytes / max(total_patches, 1),
        "patches_per_sentence": total_patches / max(len(per_sentence_patches), 1),
        # Compression ratio is the headline: 1.0 means the patcher gave up and
        # is emitting one patch per byte.
        "compression_ratio": total_bytes / max(total_patches, 1),
        "mean_patch_length": (
            sum(patch_lengths) / len(patch_lengths) if patch_lengths else 0.0
        ),
    }


def main():
    p = argparse.ArgumentParser(
        description="Measure entropy-model transfer across languages and scripts"
    )
    p.add_argument("--entropy_model", default="patching_scratch/entropy_model_marathi.pt")
    p.add_argument(
        "--languages", nargs="+", default=list(DEFAULT_INDIC_PANEL),
        help="FLORES codes or aliases. English is always added, because it is "
             "the source side the MT model actually encodes.",
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_sentences", type=int, default=None,
                   help="Cap sentences per language (default: the full split).")
    p.add_argument("--out_csv", default="results/multilingual_entropy.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_flores_args(p)
    add_resume_args(p, training=False)
    add_plot_args(p)
    add_results_args(p)
    args = p.parse_args()

    device = report_device(args.device)
    fingerprint = config_fingerprint(args, extra={"stage": "eval_multilingual"})

    languages = [resolve_lang(x) for x in args.languages]
    # English is the source side of the translation task; leaving it out would
    # omit the one mismatch that affects every reported MT number.
    if "eng_Latn" not in languages:
        languages.append("eng_Latn")
    if TRAINING_LANGUAGE not in languages:
        languages.insert(0, TRAINING_LANGUAGE)

    print(f"Entropy model: {args.entropy_model} (trained on Marathi)")
    print(f"Languages: {', '.join(languages)}")
    model = load_entropy_model(args.entropy_model, str(device))

    stages = StageTracker(
        os.path.splitext(args.out_csv)[0] + ".state.json",
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    rows: list[dict] = []
    for code in languages:
        try:
            # FLORES is n-way parallel, so pulling each language as the TARGET
            # of an en->X pair gives the same sentences in every language.
            if code == "eng_Latn":
                pairs = load_flores_pairs(
                    "eng_Latn", TRAINING_LANGUAGE, args.flores_split,
                    args.max_sentences, args.flores_dataset,
                )
                texts = [ex.source for ex in pairs]
            else:
                pairs = load_flores_pairs(
                    "eng_Latn", code, args.flores_split, args.max_sentences,
                    args.flores_dataset,
                )
                texts = [ex.target for ex in pairs]
        except Exception as e:
            print(f"[multilingual] skipping {code}: {e}")
            continue

        stats = stages.run(
            code,
            lambda code=code, texts=texts: measure_language(
                model, texts, args.threshold, str(device), args.batch_size, code
            ),
        )
        if not stats:
            continue
        row = {
            "language": code,
            "name": language_name(code),
            "script": code.split("_")[1] if "_" in code else "",
            "is_training_language": code == TRAINING_LANGUAGE,
            **stats,
        }
        rows.append(row)
        print(
            f"  {language_name(code):<12} bits/byte={row['bits_per_byte']:.3f}  "
            f"bytes/patch={row['bytes_per_patch']:.2f}  "
            f"patches/sent={row['patches_per_sentence']:.1f}"
        )

    if not rows:
        print("No languages measured.")
        return

    # Everything relative to the training language, which is the comparison
    # that actually answers "does it transfer".
    base = next((r for r in rows if r["language"] == TRAINING_LANGUAGE), None)
    if base:
        for r in rows:
            r["bits_per_byte_ratio"] = r["bits_per_byte"] / base["bits_per_byte"]
            r["compression_ratio_rel"] = (
                r["compression_ratio"] / base["compression_ratio"]
            )
        print(f"\n{'=' * 66}")
        print(f"  Relative to {language_name(TRAINING_LANGUAGE)} (the training language)")
        print(f"{'=' * 66}")
        for r in sorted(rows, key=lambda x: x["bits_per_byte_ratio"]):
            flag = "  <- training language" if r["is_training_language"] else ""
            print(
                f"  {r['name']:<14} ({r['script']})  "
                f"bits/byte {r['bits_per_byte_ratio']:5.2f}x  "
                f"compression {r['compression_ratio_rel']:5.2f}x{flag}"
            )
        eng = next((r for r in rows if r["language"] == "eng_Latn"), None)
        if eng:
            print(
                f"\n  English source side: {eng['bits_per_byte_ratio']:.2f}x the "
                f"bits/byte of Marathi. This is the mismatch every reported "
                f"En->Mr number is computed through."
            )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")

    plot_dir = resolve_plot_dir(args, os.path.dirname(args.out_csv) or "results")
    figures: list[str] = []
    if plot_dir:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, "multilingual_entropy")
        ordered = sorted(rows, key=lambda r: r["bits_per_byte"])
        names = [f"{r['name']}\n({r['script']})" for r in ordered]
        figures += plot_grouped_bars(
            names,
            {"bits / byte": [r["bits_per_byte"] for r in ordered]},
            f"{prefix}_bits_per_byte",
            title=(
                "Marathi byte entropy model applied to other languages\n"
                f"(FLORES-200 {args.flores_split}, identical sentences)"
            ),
            x_label="Language (script)",
            y_label="Bits per byte",
            formats=formats,
        )
        figures += plot_grouped_bars(
            names,
            {"bytes / patch": [r["bytes_per_patch"] for r in ordered]},
            f"{prefix}_compression",
            title=(
                f"Patch granularity across languages (τ={args.threshold:g})\n"
                "1.0 byte/patch means the patcher has degenerated to byte level"
            ),
            x_label="Language (script)",
            y_label="Bytes per patch",
            formats=formats,
        )
        figures += plot_table(
            [
                [
                    r["name"], r["script"],
                    f"{r['bits_per_byte']:.3f}",
                    f"{r.get('bits_per_byte_ratio', float('nan')):.2f}x",
                    f"{r['bytes_per_patch']:.2f}",
                    f"{r['patches_per_sentence']:.1f}",
                    "yes" if r["is_training_language"] else "",
                ]
                for r in ordered
            ],
            ["Language", "Script", "Bits/byte", "vs Marathi", "Bytes/patch",
             "Patches/sent", "Trained on"],
            f"{prefix}_table",
            title=f"Entropy-model transfer, FLORES-200 {args.flores_split}",
            highlight_row=next(
                (i for i, r in enumerate(ordered) if r["is_training_language"]), None
            ),
            formats=formats,
        )

    recorder = ResultsRecorder(
        args, run_name="multilingual_entropy", script="eval_multilingual.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, args.out_csv)
    for r in rows:
        recorder.add_metrics(**{f"{r['language']}_bits_per_byte": r["bits_per_byte"]})
        recorder.add_metrics(
            **{f"{r['language']}_bytes_per_patch": r["bytes_per_patch"]}
        )
    recorder.add_info(
        entropy_model=args.entropy_model,
        training_language=TRAINING_LANGUAGE,
        threshold=args.threshold,
        benchmark=f"flores200-{args.flores_split}",
        languages=[r["language"] for r in rows],
        scripts=sorted({r["script"] for r in rows}),
    )
    recorder.publish()

    return rows


if __name__ == "__main__":
    main()
