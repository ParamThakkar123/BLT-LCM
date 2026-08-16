"""Evaluate BLT-LCM on FLORES-200, the standard benchmark, with error bars.

Why this exists: every other evaluation in this repository runs on an ad-hoc
BhashaSetu split that no other paper reports on, so a chrF++ of 42 cannot be
situated against anything. FLORES-200 devtest is what IndicTrans2, NLLB and the
Indic MT literature report on, and it is n-way parallel, so the same 1012
sentences are used for every language pair.

What it produces:

  * BLEU / chrF++ / TER / COMET on FLORES devtest, at each source-noise level
  * a paired bootstrap test against any number of comparison systems
    (``--compare name=path/to/hyps.txt``), so a gap comes with a p-value
  * mean ± std across ``--seeds``, when several trained checkpoints are given
  * the standard figures, and a published results record

Usage:
  python lcm_scripts/eval_flores.py \
    --lcm_checkpoint lcm_models/lcm_blt_mt_fraction0.25_s42_best.pth \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --pooler lcm_models/blt_pooler.pth --decoder lcm_models/blt_decoder.pth \
    --flores_tgt mar_Deva --comet_model Unbabel/wmt22-comet-da \
    --compare nllb=outputs/nllb_flores_mr.txt
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

from base_lcm import BaseLCM
from blt_loader import BLTLoader
from blt_decoder import load_decoder
from bhashasetu_utils import add_character_noise
from eval_metrics import compute_all
from device_utils import report_device
from flores_utils import (
    add_flores_args,
    language_name,
    load_flores_pairs,
    resolve_lang,
)
from plot_utils import (
    add_plot_args,
    plot_formats,
    plot_metric_bars,
    plot_noise_curves,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from significance import (
    add_significance_args,
    paired_test,
    seed_summary,
)
from checkpoint_utils import (
    StageTracker,
    add_resume_args,
    config_fingerprint,
    load_model_state,
    seed_everything,
)


def encode_concepts(blt, texts, batch_size=128, desc="encoding"):
    """Frozen concept encoding; returns a [N, dim] CPU tensor."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        embs.extend(
            [e.detach().cpu() for e in blt.encode_sentences_batch(texts[i : i + batch_size])]
        )
    return torch.stack(embs)


def translate(lcm, decoder, blt, srcs, args, device, desc="translate"):
    """Source text -> concept -> predicted concept -> target text."""
    src_emb = encode_concepts(blt, srcs, args.encode_batch_size, f"encode ({desc})").to(
        device
    )
    hyps: list[str] = []
    for i in tqdm(range(0, len(src_emb), args.batch_size), desc=desc):
        batch = src_emb[i : i + args.batch_size].unsqueeze(1)  # [b, 1, dim]
        with torch.no_grad():
            pred = lcm(batch)
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            hyps.extend(decoder.decode(pred, max_len=args.max_decode_len))
    return hyps


def read_hypotheses(path: str, expected: int) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) != expected:
        raise ValueError(
            f"{path}: {len(lines)} lines but FLORES has {expected} sentences. "
            "A comparison system must be scored on exactly the same segments, "
            "or the paired test is meaningless."
        )
    return lines


def main():
    p = argparse.ArgumentParser(
        description="Evaluate BLT-LCM on FLORES-200 with significance testing"
    )
    p.add_argument(
        "--lcm_checkpoint",
        nargs="+",
        required=True,
        help="One checkpoint, or several (one per training seed) to get error bars.",
    )
    p.add_argument("--entropy_model", required=True)
    p.add_argument("--pooler", default="lcm_models/blt_pooler.pth")
    p.add_argument("--decoder", default="lcm_models/blt_decoder.pth")
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--encode_batch_size", type=int, default=128)
    p.add_argument("--max_decode_len", type=int, default=256)
    p.add_argument(
        "--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.2],
        help="Character noise applied to the SOURCE side at eval time.",
    )
    p.add_argument(
        "--compare",
        nargs="*",
        default=[],
        metavar="NAME=PATH",
        help="Comparison systems as name=hypothesis_file, one sentence per "
        "line, in FLORES devtest order. Each is paired-bootstrap tested "
        "against this model on the clean condition.",
    )
    p.add_argument("--comet_model", default=None)
    p.add_argument("--out_csv", default="results/flores_blt_lcm.csv")
    p.add_argument("--out_hyp_dir", default=None,
                   help="Write this model's hypotheses here, so they can be "
                        "reused as a --compare system elsewhere.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_flores_args(p)
    add_resume_args(p, training=False)
    add_significance_args(p)
    add_plot_args(p)
    add_results_args(p)
    args = p.parse_args()

    device = report_device(args.device)
    seed_everything(args.seed)
    fingerprint = config_fingerprint(args, extra={"stage": "eval_flores"})

    src_code = resolve_lang(args.flores_src)
    tgt_code = resolve_lang(args.flores_tgt)
    pair = f"{src_code}->{tgt_code}"
    # ">" is not a legal filename character on Windows, so anything that
    # reaches a path uses the hyphenated form; the arrow stays in the CSV and
    # in printed output where it reads better.
    pair_tag = f"{src_code}-{tgt_code}"
    print(f"FLORES-200 {args.flores_split}: {pair}")

    pairs = load_flores_pairs(
        src_code, tgt_code, args.flores_split, args.flores_max_examples,
        args.flores_dataset,
    )
    srcs = [ex.source for ex in pairs]
    refs = [ex.target for ex in pairs]
    print(f"  {len(pairs)} sentences")
    if args.flores_max_examples:
        print(
            "  WARNING: --flores_max_examples truncates the benchmark; these "
            "numbers are NOT comparable to published FLORES results."
        )

    # --- encoder / decoder (shared across checkpoints) ---
    blt = BLTLoader(
        entropy_model_path=args.entropy_model,
        device=str(device),
        pooler_path=args.pooler,
    )
    if not os.path.exists(args.decoder):
        raise FileNotFoundError(
            f"FLORES evaluation needs the trained BLTDecoder at '{args.decoder}'."
        )
    decoder = load_decoder(args.decoder, device=str(device))
    if decoder.embed_dim != blt.dim:
        raise ValueError(
            f"Decoder embed_dim ({decoder.embed_dim}) != concept dim ({blt.dim})."
        )

    stages = StageTracker(
        os.path.splitext(args.out_csv)[0] + ".state.json",
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    rows: list[dict] = []
    clean_hyps_by_ckpt: dict[str, list[str]] = {}

    for ckpt_path in args.lcm_checkpoint:
        tag = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\n=== {tag} ===")
        lcm = BaseLCM(
            embed_dim=blt.dim, model_dim=args.model_dim,
            n_layers=args.n_layers, n_heads=args.n_heads,
        ).to(device)
        lcm.load_state_dict(load_model_state(ckpt_path, map_location=device))
        lcm.eval()

        for noise in args.noise_levels:
            def _run(noise=noise, lcm=lcm, tag=tag):
                noisy = (
                    [add_character_noise(s, noise, seed=i) for i, s in enumerate(srcs)]
                    if noise
                    else srcs
                )
                hyps = translate(
                    lcm, decoder, blt, noisy, args, device, f"{tag} noise={noise}"
                )
                # COMET is scored against the CLEAN source: the question is
                # whether the output conveys the true source meaning, and the
                # true meaning is the uncorrupted sentence.
                metrics = compute_all(
                    hyps, refs, srcs=srcs,
                    comet_model_name=args.comet_model, device=str(device),
                )
                return {"hyps": hyps, "metrics": metrics}

            result = stages.run(f"{tag}|noise={noise}", _run)
            metrics = result["metrics"]
            if noise == 0.0:
                clean_hyps_by_ckpt[tag] = result["hyps"]
            row = {
                "model": "blt_lcm",
                "checkpoint": tag,
                "benchmark": f"flores200-{args.flores_split}",
                "pair": pair,
                "noise": noise,
                "n": len(refs),
                **metrics,
            }
            rows.append(row)
            print(
                f"  noise={noise:.0%}: "
                + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items() if v == v)
            )

            if args.out_hyp_dir:
                os.makedirs(args.out_hyp_dir, exist_ok=True)
                hp = os.path.join(
                    args.out_hyp_dir, f"{tag}_{pair_tag}_noise{noise}.hyp.txt"
                )
                with open(hp, "w", encoding="utf-8") as f:
                    f.write("\n".join(result["hyps"]) + "\n")

    # --- seed aggregation: mean ± std over checkpoints ---
    clean_rows = [r for r in rows if r["noise"] == 0.0]
    summaries = seed_summary(
        clean_rows,
        metrics=("BLEU", "chrF++", "TER", "COMET"),
        group_keys=("model", "pair"),
    )
    print(f"\n{'=' * 60}\n  FLORES-200 {args.flores_split} {pair}, clean input\n{'=' * 60}")
    for key, per_metric in summaries.items():
        print(f"  {key}")
        for s in per_metric.values():
            print(f"    {s.format()}")

    # --- paired significance tests against the comparison systems ---
    comparisons = []
    if args.compare and args.significance_test != "none" and clean_hyps_by_ckpt:
        ours = clean_hyps_by_ckpt[list(clean_hyps_by_ckpt)[0]]
        systems = {}
        for spec in args.compare:
            if "=" not in spec:
                print(f"[compare] ignoring '{spec}' (expected name=path)")
                continue
            name, path = spec.split("=", 1)
            try:
                systems[name] = read_hypotheses(path, len(refs))
            except Exception as e:
                print(f"[compare] {name}: {e}")
        if systems:
            print(f"\nPaired {args.significance_test} test vs BLT-LCM "
                  f"({args.significance_samples} resamples):")
            # Our system is the baseline, so a positive delta means the
            # comparison system beat us -- stated plainly either way.
            comparisons = paired_test(
                ours, systems, refs,
                test_type=args.significance_test,
                n_samples=args.significance_samples,
            )
            for c in comparisons:
                print(f"  {c.format()}")

    # --- CSV ---
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = ["model", "checkpoint", "benchmark", "pair", "noise", "n",
              "BLEU", "chrF++", "TER", "METEOR", "COMET"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")

    sig_csv = None
    if comparisons:
        sig_csv = os.path.splitext(args.out_csv)[0] + "_significance.csv"
        with open(sig_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comparisons[0].as_dict()))
            w.writeheader()
            w.writerows(c.as_dict() for c in comparisons)
        print(f"Wrote {sig_csv}")

    # --- figures ---
    plot_dir = resolve_plot_dir(args, os.path.dirname(args.out_csv) or "results")
    figures: list[str] = []
    if plot_dir and rows:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, f"flores_{tgt_code}")
        title_pair = f"{language_name(src_code)}→{language_name(tgt_code)}"
        figures += plot_noise_curves(
            rows,
            f"{prefix}_noise_robustness",
            title=f"BLT-LCM on FLORES-200 {args.flores_split} ({title_pair})",
            metrics=("BLEU", "chrF++", "TER", "COMET"),
            series_key="checkpoint" if len(args.lcm_checkpoint) > 1 else None,
            formats=formats,
        )
        clean = clean_rows[0] if clean_rows else {}
        figures += plot_metric_bars(
            {k: clean.get(k) for k in ("BLEU", "chrF++", "TER", "METEOR", "COMET")},
            f"{prefix}_clean_metrics",
            title=f"FLORES-200 {args.flores_split} {title_pair}, clean input",
            subtitle=f"{len(refs)} sentences",
            formats=formats,
        )
        # Mean ± std across seeds is what belongs in the paper's table.
        table_rows = []
        for key, per_metric in summaries.items():
            for metric, s in per_metric.items():
                lo, hi = s.confidence_interval()
                table_rows.append([
                    metric, f"{s.mean:.2f}",
                    f"{s.std:.2f}" if s.n > 1 else "—",
                    str(s.n),
                    f"[{lo:.2f}, {hi:.2f}]" if s.n > 1 else "—",
                ])
        if table_rows:
            figures += plot_table(
                table_rows,
                ["Metric", "Mean", "Std", "Seeds", "95% CI"],
                f"{prefix}_seed_summary",
                title=f"FLORES-200 {args.flores_split} {title_pair} across seeds",
                formats=formats,
            )
        if comparisons:
            figures += plot_table(
                [
                    [c.system, c.metric, f"{c.baseline_score:.2f}",
                     f"{c.system_score:.2f}", f"{c.delta:+.2f}",
                     "n/a" if c.p_value is None else f"{c.p_value:.4f}",
                     "yes" if c.significant else "no"]
                    for c in comparisons
                ],
                ["System", "Metric", "BLT-LCM", "System", "Δ", "p", "p<0.05"],
                f"{prefix}_significance",
                title=(
                    f"Paired {args.significance_test} test vs BLT-LCM "
                    f"({args.significance_samples} resamples)"
                ),
                formats=formats,
            )

    recorder = ResultsRecorder(
        args,
        run_name=f"flores_{tgt_code}_{args.flores_split}",
        script="eval_flores.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, args.out_csv, sig_csv or "")
    for key, per_metric in summaries.items():
        for metric, s in per_metric.items():
            recorder.add_metrics(**{f"clean_{metric}_mean": s.mean})
            if s.n > 1:
                recorder.add_metrics(**{f"clean_{metric}_std": s.std})
    recorder.add_info(
        benchmark=f"flores200-{args.flores_split}",
        pair=pair,
        sentences=len(refs),
        checkpoints=list(args.lcm_checkpoint),
        noise_levels=args.noise_levels,
        significance_test=args.significance_test,
        significant_vs=[c.system for c in comparisons if c.significant],
    )
    recorder.publish()

    return rows


if __name__ == "__main__":
    main()
