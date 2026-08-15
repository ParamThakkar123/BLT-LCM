"""Score published MT systems on the same FLORES-200 segments.

The first question a reviewer asks is "how does this compare to IndicTrans2 /
NLLB?". Without an answer, an in-house-only comparison reads as avoiding it.
This script runs the public systems on exactly the segments
``eval_flores.py`` uses and writes their hypotheses to disk, so the two are
scored on identical data and can be compared with a paired test.

Supported out of the box:

  * ``nllb-600m`` / ``nllb-1.3b`` / ``nllb-3.3b`` -- facebook/nllb-200-*
  * ``indictrans2`` -- ai4bharat/indictrans2-en-indic-1B (and the indic-en
    direction), which is the state of the art for these language pairs
  * ``--model`` with any seq2seq checkpoint, for anything else

These are large downloads and need network access the first time. Nothing else
in the repository depends on this script, so a machine without access simply
does not run it.

Usage:
  python lcm_scripts/eval_public_baselines.py --systems nllb-600m indictrans2 \
      --flores_tgt mar_Deva --out_dir outputs/flores_baselines
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import torch
from tqdm import tqdm

from bhashasetu_utils import add_character_noise
from device_utils import report_device
from eval_metrics import compute_all
from flores_utils import add_flores_args, language_name, load_flores_pairs, resolve_lang
from plot_utils import (
    add_plot_args,
    plot_formats,
    plot_grouped_bars,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from checkpoint_utils import StageTracker, add_resume_args, config_fingerprint


# name -> (hub id, kind). "kind" decides how the language codes are passed.
KNOWN_SYSTEMS: dict[str, tuple[str, str]] = {
    "nllb-600m": ("facebook/nllb-200-distilled-600M", "nllb"),
    "nllb-1.3b": ("facebook/nllb-200-distilled-1.3B", "nllb"),
    "nllb-3.3b": ("facebook/nllb-200-3.3B", "nllb"),
    "indictrans2": ("ai4bharat/indictrans2-en-indic-1B", "indictrans2"),
    "indictrans2-indic-en": ("ai4bharat/indictrans2-indic-en-1B", "indictrans2"),
}

# IndicTrans2 uses its own language tags rather than FLORES codes.
INDICTRANS2_CODES = {
    "eng_Latn": "eng_Latn", "mar_Deva": "mar_Deva", "hin_Deva": "hin_Deva",
    "ben_Beng": "ben_Beng", "guj_Gujr": "guj_Gujr", "tam_Taml": "tam_Taml",
    "tel_Telu": "tel_Telu", "kan_Knda": "kan_Knda", "mal_Mlym": "mal_Mlym",
    "pan_Guru": "pan_Guru", "ory_Orya": "ory_Orya", "asm_Beng": "asm_Beng",
    "npi_Deva": "npi_Deva", "urd_Arab": "urd_Arab", "san_Deva": "san_Deva",
}


def load_system(hub_id: str, device: str, dtype=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(hub_id, token=token, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        hub_id,
        token=token,
        trust_remote_code=True,
        torch_dtype=dtype or (torch.float16 if device.startswith("cuda") else torch.float32),
    ).to(device)
    model.eval()
    return tok, model


def translate_nllb(tok, model, srcs, src_code, tgt_code, args, device):
    """NLLB takes the source language as the tokenizer's src_lang and the
    target as a forced BOS token."""
    tok.src_lang = src_code
    try:
        forced = tok.convert_tokens_to_ids(tgt_code)
    except Exception:
        forced = None
    if forced is None or forced == tok.unk_token_id:
        # Older/newer tokenizer versions expose this differently; both forms
        # are tried rather than silently generating in the wrong language.
        forced = getattr(tok, "lang_code_to_id", {}).get(tgt_code)
    if forced is None:
        raise ValueError(
            f"NLLB tokenizer does not know target language '{tgt_code}'"
        )

    hyps = []
    for i in tqdm(range(0, len(srcs), args.batch_size), desc="nllb"):
        batch = srcs[i : i + args.batch_size]
        enc = tok(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_len,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                forced_bos_token_id=forced,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        hyps.extend(tok.batch_decode(out, skip_special_tokens=True))
    return hyps


def translate_indictrans2(tok, model, srcs, src_code, tgt_code, args, device):
    """IndicTrans2 expects its own preprocessing (IndicProcessor)."""
    try:
        from IndicTransToolkit.processor import IndicProcessor
    except Exception:
        try:
            from IndicTransToolkit import IndicProcessor  # older layout
        except Exception as e:
            raise RuntimeError(
                "IndicTrans2 needs the IndicTransToolkit preprocessor:\n"
                "  pip install IndicTransToolkit\n"
                "Without it the model is fed unnormalized text and its scores "
                f"are not the published ones. ({e})"
            ) from e

    ip = IndicProcessor(inference=True)
    src = INDICTRANS2_CODES.get(src_code, src_code)
    tgt = INDICTRANS2_CODES.get(tgt_code, tgt_code)

    hyps = []
    for i in tqdm(range(0, len(srcs), args.batch_size), desc="indictrans2"):
        batch = srcs[i : i + args.batch_size]
        prepped = ip.preprocess_batch(batch, src_lang=src, tgt_lang=tgt)
        enc = tok(
            prepped, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_len,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams
            )
        decoded = tok.batch_decode(out, skip_special_tokens=True)
        hyps.extend(ip.postprocess_batch(decoded, lang=tgt))
    return hyps


def main():
    p = argparse.ArgumentParser(
        description="Score published MT systems on FLORES-200"
    )
    p.add_argument(
        "--systems",
        nargs="+",
        default=["nllb-600m"],
        help=f"Named systems ({', '.join(KNOWN_SYSTEMS)}) and/or 'name=hub_id'.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=5,
                   help="Published FLORES numbers use beam search; beam 5 is "
                        "the NLLB/IndicTrans2 convention.")
    p.add_argument(
        "--noise_levels", type=float, nargs="+", default=[0.0],
        help="Source-noise levels. Pass 0.0 0.1 0.2 to reproduce the "
             "robustness comparison against BLT-LCM.",
    )
    p.add_argument("--out_dir", default="outputs/flores_baselines")
    p.add_argument("--out_csv", default="results/flores_public_baselines.csv")
    p.add_argument("--comet_model", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_flores_args(p)
    add_resume_args(p, training=False)
    add_plot_args(p)
    add_results_args(p)
    args = p.parse_args()

    device = report_device(args.device)
    fingerprint = config_fingerprint(args, extra={"stage": "eval_public_baselines"})

    src_code = resolve_lang(args.flores_src)
    tgt_code = resolve_lang(args.flores_tgt)
    pair = f"{src_code}->{tgt_code}"
    # ">" is illegal in a Windows filename; paths use the hyphenated form.
    pair_tag = f"{src_code}-{tgt_code}"

    pairs = load_flores_pairs(
        src_code, tgt_code, args.flores_split, args.flores_max_examples,
        args.flores_dataset,
    )
    srcs = [ex.source for ex in pairs]
    refs = [ex.target for ex in pairs]
    print(f"FLORES-200 {args.flores_split} {pair}: {len(pairs)} sentences")

    os.makedirs(args.out_dir, exist_ok=True)
    stages = StageTracker(
        os.path.splitext(args.out_csv)[0] + ".state.json",
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    rows: list[dict] = []
    for spec in args.systems:
        if "=" in spec:
            name, hub_id = spec.split("=", 1)
            kind = "indictrans2" if "indictrans" in hub_id.lower() else "nllb"
        elif spec in KNOWN_SYSTEMS:
            name = spec
            hub_id, kind = KNOWN_SYSTEMS[spec]
        else:
            print(f"[baselines] unknown system '{spec}'; skipping")
            continue

        print(f"\n=== {name} ({hub_id}) ===")
        try:
            tok, model = load_system(hub_id, str(device))
        except Exception as e:
            print(
                f"[baselines] could not load {hub_id}: {e}\n"
                "  These are large downloads and need network access the "
                "first time. Skipping this system."
            )
            continue

        for noise in args.noise_levels:
            def _run(noise=noise, name=name, tok=tok, model=model, kind=kind):
                inputs = (
                    [add_character_noise(s, noise, seed=i) for i, s in enumerate(srcs)]
                    if noise
                    else srcs
                )
                fn = translate_indictrans2 if kind == "indictrans2" else translate_nllb
                hyps = fn(tok, model, inputs, src_code, tgt_code, args, str(device))
                metrics = compute_all(
                    hyps, refs, srcs=srcs,
                    comet_model_name=args.comet_model, device=str(device),
                )
                return {"hyps": hyps, "metrics": metrics}

            res = stages.run(f"{name}|noise={noise}", _run)
            # Written out so eval_flores.py can paired-test against them:
            #   --compare nllb-600m=outputs/flores_baselines/nllb-600m_...hyp.txt
            hp = os.path.join(
                args.out_dir, f"{name}_{pair_tag}_noise{noise}.hyp.txt"
            )
            with open(hp, "w", encoding="utf-8") as f:
                f.write("\n".join(res["hyps"]) + "\n")
            rows.append({
                "model": name,
                "hub_id": hub_id,
                "benchmark": f"flores200-{args.flores_split}",
                "pair": pair,
                "noise": noise,
                "n": len(refs),
                "hyp_file": hp,
                **res["metrics"],
            })
            print(
                f"  noise={noise:.0%}: "
                + "  ".join(f"{k}={v:.2f}" for k, v in res["metrics"].items() if v == v)
                + f"  -> {hp}"
            )

        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not rows:
        print("\nNo systems were scored. Nothing written.")
        return

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = ["model", "hub_id", "benchmark", "pair", "noise", "n", "hyp_file",
              "BLEU", "chrF++", "TER", "METEOR", "COMET"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")

    plot_dir = resolve_plot_dir(args, os.path.dirname(args.out_csv) or "results")
    figures: list[str] = []
    if plot_dir:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, f"flores_public_{tgt_code}")
        models = sorted({r["model"] for r in rows})
        noises = sorted({r["noise"] for r in rows})

        def cell(m, n, metric):
            for r in rows:
                if r["model"] == m and r["noise"] == n:
                    v = r.get(metric)
                    return float(v) if isinstance(v, (int, float)) else float("nan")
            return float("nan")

        for metric in ("BLEU", "chrF++"):
            figures += plot_grouped_bars(
                [f"{n:.0%}" for n in noises],
                {m: [cell(m, n, metric) for n in noises] for m in models},
                f"{prefix}_{metric.replace('+', 'p')}",
                title=(
                    f"Published systems on FLORES-200 {args.flores_split} "
                    f"({language_name(src_code)}→{language_name(tgt_code)})"
                ),
                x_label="Source noise",
                y_label=metric,
                formats=formats,
            )
        figures += plot_table(
            [
                [r["model"], f"{r['noise']:.0%}", f"{r['BLEU']:.2f}",
                 f"{r['chrF++']:.2f}", f"{r['TER']:.2f}",
                 f"{r['COMET']:.4f}" if r["COMET"] == r["COMET"] else "—"]
                for r in rows
            ],
            ["System", "Noise", "BLEU ↑", "chrF++ ↑", "TER ↓", "COMET ↑"],
            f"{prefix}_table",
            title=f"Published baselines, FLORES-200 {args.flores_split} {pair}",
            formats=formats,
        )

    recorder = ResultsRecorder(
        args,
        run_name=f"flores_public_baselines_{tgt_code}",
        script="eval_public_baselines.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, args.out_csv)
    for r in rows:
        if r["noise"] == 0.0:
            for metric in ("BLEU", "chrF++", "TER"):
                recorder.add_metrics(**{f"{r['model']}_{metric}": r.get(metric)})
    recorder.add_info(
        benchmark=f"flores200-{args.flores_split}",
        pair=pair,
        sentences=len(refs),
        systems=[r["model"] for r in rows],
        num_beams=args.num_beams,
    )
    recorder.publish()


if __name__ == "__main__":
    main()
