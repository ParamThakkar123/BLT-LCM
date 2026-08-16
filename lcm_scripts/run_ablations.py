"""Run the ablations a reviewer will ask for, and tabulate them together.

Three separate asks, one driver, because they share a harness:

**1. LCM variant** (``--ablation variant``). ``train_lcm_blt.py`` implements all
four variants from the LCM paper -- Base (MSE), One-Tower diffusion, Two-Tower
diffusion, and Quant-LCM -- and the paper reports that diffusion beats the MSE
baseline. Reporting only ``base`` invites the question "did you try the variant
your own citation says is better?".

**2. Decoding method** (``--ablation decode``). ``eval_lcm_blt.py`` supports
generative decoding through the trained BLTDecoder and a nearest-neighbour
retrieval baseline. Retrieval can only ever emit a sentence that already exists
in the training corpus, so it flatters corpus-level metrics without the model
generating anything. Reporting both separates "the concept space is good" from
"the decoder is good".

**3. Compute-matched comparison** (``--ablation compute``). BLT-LCM against the
BPE Transformer is only a fair comparison if the two see the same parameter
budget and the same number of training tokens. This mode sizes the baseline to
match and records both budgets in the results, so the table can state the match
rather than leaving a reviewer to assume the win came from the mismatch.

Each configuration is a full sub-run with its own figures and published record;
this script adds the cross-configuration comparison on top.

Usage:
  python lcm_scripts/run_ablations.py --ablation variant \
      --entropy_model patching_scratch/entropy_model_marathi.pt --fraction 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from device_utils import report_device
from plot_utils import (
    add_plot_args,
    plot_formats,
    plot_grouped_bars,
    plot_lines,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args

SCRIPT_DIR = Path(__file__).resolve().parent

LCM_VARIANTS = ("base", "one_tower", "two_tower", "quant")
DECODE_METHODS = ("generative", "retrieval")


def run(cmd: list[str], dry_run: bool = False) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def lcm_parameter_count(embed_dim, model_dim, n_layers, n_heads) -> int:
    """Parameters of a BaseLCM at this size, without training anything."""
    from base_lcm import BaseLCM

    return count_parameters(
        BaseLCM(
            embed_dim=embed_dim, model_dim=model_dim,
            n_layers=n_layers, n_heads=n_heads,
        )
    )


def transformer_parameter_count(vocab, d_model, nhead, layers, dim_ff) -> int:
    """Parameter count of ``BPETransformer`` at this size, computed analytically.

    Instantiating each candidate would be simpler, but the search grid reaches
    past a billion parameters and allocating those just to call ``numel()``
    exhausts memory. The formula below mirrors ``nn.Transformer``'s layout and
    is checked against a real model by
    ``tests/test_run_ablations.py::test_analytic_count_matches_a_real_model``.
    """
    d, ff, L = d_model, dim_ff, layers
    # Embeddings: separate source and target tables, plus the output projection.
    embeddings = 2 * vocab * d + (d * vocab + vocab)
    # Encoder layer: self-attention (in_proj 3d x d + bias, out_proj d x d +
    # bias), two feed-forward linears, two LayerNorms.
    enc_layer = (3 * d * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d
    # Decoder layer adds cross-attention and a third LayerNorm.
    dec_layer = enc_layer + (3 * d * d + 3 * d) + (d * d + d) + 2 * d
    # nn.Transformer wraps each stack in a final LayerNorm.
    return embeddings + L * enc_layer + L * dec_layer + 2 * (2 * d)


def match_transformer_to_lcm(target_params: int, vocab: int, nhead: int = 8) -> dict:
    """Pick a BPE-Transformer size whose parameter count matches the LCM's.

    Searched rather than solved: the embedding tables dominate and depend on
    the vocabulary, so a closed form would be wrong as soon as --vocab_size
    changed. The search is over a small grid and costs a few model
    constructions, no training.
    """
    best = None
    # The grid has to reach past the LCM's size in both directions, or the
    # "match" is just the largest configuration on offer and the comparison is
    # still unmatched -- at embed 1024 / model 2048 / 12 layers the LCM is
    # ~610M parameters, well above a 6-layer 512-wide baseline.
    for layers in (2, 4, 6, 8, 12, 16, 20, 24):
        for d_model in (256, 384, 512, 640, 768, 1024, 1280, 1536, 2048):
            if d_model % nhead:
                continue
            for ff_mult in (2, 4, 8):
                dim_ff = d_model * ff_mult
                try:
                    n = transformer_parameter_count(
                        vocab, d_model, nhead, layers, dim_ff
                    )
                except Exception:
                    continue
                err = abs(n - target_params) / max(target_params, 1)
                cand = {
                    "d_model": d_model, "nhead": nhead, "num_layers": layers,
                    "dim_ff": dim_ff, "parameters": n, "relative_error": err,
                }
                if best is None or err < best["relative_error"]:
                    best = cand
    if best and best["relative_error"] > 0.05:
        # Said out loud rather than quietly reported as "matched": a 30% gap
        # is exactly the confound this mode exists to remove.
        print(
            f"[compute-match] WARNING: closest configuration is "
            f"{best['relative_error'] * 100:.1f}% from the target parameter "
            "count. Widen the search grid, or state the residual gap in the "
            "paper rather than calling this compute-matched."
        )
    return best or {}


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #


def ablation_variant(args, out_dir: Path) -> list[dict]:
    """Train and evaluate each LCM variant at the same size and data."""
    rows = []
    for variant in args.variants:
        run_dir = out_dir / f"variant_{variant}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(SCRIPT_DIR / "train_lcm_blt.py"),
            "--entropy_model", args.entropy_model,
            "--lcm_variant", variant,
            "--fraction", str(args.fraction),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--model_dim", str(args.model_dim),
            "--n_layers", str(args.n_layers),
            "--n_heads", str(args.n_heads),
            "--model_dir", str(run_dir),
            "--embed_cache", str(args.embed_cache or run_dir / "emb.pth"),
            "--plot_format", args.plot_format,
        ] + (["--device", args.device] if args.device else [])
        if run(cmd, args.dry_run) != 0:
            print(f"[ablation] variant '{variant}' failed; continuing")
            continue
        hist = read_json(run_dir / f"lcm_blt_{variant}_history.json")
        epochs = hist.get("epochs", [])
        rows.append({
            "ablation": "variant",
            "setting": variant,
            "final_loss": epochs[-1]["train_loss"] if epochs else float("nan"),
            "best_loss": min((e["train_loss"] for e in epochs), default=float("nan")),
            "epochs": len(epochs),
            "model_dir": str(run_dir),
        })
    return rows


def ablation_decode(args, out_dir: Path) -> list[dict]:
    """Evaluate one trained LCM under generative and retrieval decoding."""
    if not args.lcm_checkpoint:
        raise SystemExit(
            "--ablation decode needs --lcm_checkpoint (the trained LCM to evaluate)"
        )
    rows = []
    for method in args.decode_methods:
        run_dir = out_dir / f"decode_{method}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out_csv = run_dir / f"eval_{method}.csv"
        cmd = [
            sys.executable, str(SCRIPT_DIR / "eval_lcm_blt.py"),
            "--lcm_checkpoint", args.lcm_checkpoint,
            "--entropy_model", args.entropy_model,
            "--pooler", args.pooler,
            "--decoder", args.decoder,
            "--decode_method", method,
            "--fraction", str(args.fraction),
            "--out_csv", str(out_csv),
            "--plot_dir", str(run_dir),
            "--plot_format", args.plot_format,
        ] + (["--device", args.device] if args.device else [])
        if args.comet_model:
            cmd += ["--comet_model", args.comet_model]
        if run(cmd, args.dry_run) != 0:
            print(f"[ablation] decode '{method}' failed; continuing")
            continue
        # eval_lcm_blt writes its metrics into the published record.
        rec = read_json(
            Path(args.results_dir) / f"eval_lcm_blt_{method}" / "run.json"
        )
        metrics = rec.get("metrics", {})
        rows.append({
            "ablation": "decode",
            "setting": method,
            **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        })
    return rows


def ablation_compute(args, out_dir: Path) -> list[dict]:
    """BLT-LCM vs a parameter-matched BPE Transformer on the same data."""
    embed_dim = args.embed_dim
    lcm_params = lcm_parameter_count(
        embed_dim, args.model_dim, args.n_layers, args.n_heads
    )
    matched = match_transformer_to_lcm(lcm_params, args.vocab_size, args.n_heads_baseline)
    print(
        f"\n[compute-match] BLT-LCM: {lcm_params:,} parameters\n"
        f"[compute-match] matched BPE Transformer: {matched.get('parameters', 0):,} "
        f"({matched.get('relative_error', 1) * 100:.1f}% off) "
        f"d_model={matched.get('d_model')} layers={matched.get('num_layers')} "
        f"ff={matched.get('dim_ff')}"
    )
    if not matched:
        raise SystemExit("could not find a matching transformer configuration")

    rows = [{
        "ablation": "compute",
        "setting": "blt_lcm",
        "parameters": lcm_params,
        "model_dim": args.model_dim,
        "n_layers": args.n_layers,
    }]

    lcm_dir = out_dir / "compute_blt_lcm"
    lcm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "train_lcm_blt.py"),
        "--entropy_model", args.entropy_model,
        "--fraction", str(args.fraction),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--model_dim", str(args.model_dim),
        "--n_layers", str(args.n_layers),
        "--n_heads", str(args.n_heads),
        "--model_dir", str(lcm_dir),
        "--embed_cache", str(args.embed_cache or lcm_dir / "emb.pth"),
        "--plot_format", args.plot_format,
    ] + (["--device", args.device] if args.device else [])
    if run(cmd, args.dry_run) == 0:
        hist = read_json(lcm_dir / "lcm_blt_base_history.json")
        eps = hist.get("epochs", [])
        rows[0]["final_loss"] = eps[-1]["train_loss"] if eps else float("nan")

    tf_dir = out_dir / "compute_bpe_transformer"
    tf_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "train_bpe_transformer.py"),
        "--fraction", str(args.fraction),
        "--epochs", str(args.epochs),
        "--vocab_size", str(args.vocab_size),
        "--d_model", str(matched["d_model"]),
        "--nhead", str(matched["nhead"]),
        "--num_layers", str(matched["num_layers"]),
        "--dim_ff", str(matched["dim_ff"]),
        "--out_dir", str(tf_dir),
        "--plot_format", args.plot_format,
    ] + (["--device", args.device] if args.device else [])
    tf_row = {
        "ablation": "compute",
        "setting": "bpe_transformer_matched",
        "parameters": matched["parameters"],
        "model_dim": matched["d_model"],
        "n_layers": matched["num_layers"],
    }
    if run(cmd, args.dry_run) == 0:
        for r in read_csv(tf_dir / f"metrics_fraction{args.fraction}.csv"):
            if float(r.get("noise", 1)) == 0.0:
                tf_row.update({
                    k: float(r[k]) for k in ("BLEU", "chrF++", "TER") if r.get(k)
                })
    rows.append(tf_row)
    return rows


# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Run BLT-LCM ablations")
    p.add_argument(
        "--ablation", nargs="+",
        choices=["variant", "decode", "compute"], default=["variant"],
    )
    p.add_argument("--entropy_model", default="patching_scratch/entropy_model_marathi.pt")
    p.add_argument("--pooler", default="lcm_models/blt_pooler.pth")
    p.add_argument("--decoder", default="lcm_models/blt_decoder.pth")
    p.add_argument("--lcm_checkpoint", default=None,
                   help="Trained LCM, required by --ablation decode.")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--embed_cache", default=None)
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=1024,
                   help="Concept dimension, for the parameter-count match.")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--n_heads_baseline", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=list(LCM_VARIANTS),
                   choices=list(LCM_VARIANTS))
    p.add_argument("--decode_methods", nargs="+", default=list(DECODE_METHODS),
                   choices=list(DECODE_METHODS))
    p.add_argument("--comet_model", default=None)
    p.add_argument("--out_dir", default="runs/ablations")
    p.add_argument("--out_csv", default="results/ablations.csv")
    p.add_argument("--device", default=None)
    p.add_argument("--dry_run", action="store_true")
    add_plot_args(p)
    add_results_args(p)
    args = p.parse_args()

    report_device(args.device, label="ablation orchestrator")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in args.ablation:
        print(f"\n{'=' * 60}\n  ABLATION: {name}\n{'=' * 60}")
        if name == "variant":
            rows += ablation_variant(args, out_dir)
        elif name == "decode":
            rows += ablation_decode(args, out_dir)
        elif name == "compute":
            rows += ablation_compute(args, out_dir)

    if not rows:
        print("\nNo ablation rows produced.")
        return

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    fields = ["ablation", "setting"] + [f for f in fields if f not in ("ablation", "setting")]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv}")

    plot_dir = resolve_plot_dir(args, os.path.dirname(args.out_csv) or "results")
    figures: list[str] = []
    if plot_dir:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, "ablation")
        for name in args.ablation:
            sub = [r for r in rows if r["ablation"] == name]
            if not sub:
                continue
            settings = [r["setting"] for r in sub]
            # Whichever numeric columns this ablation actually produced.
            metrics = [
                m for m in ("final_loss", "best_loss", "BLEU", "chrF++", "TER",
                            "parameters", "samples")
                if any(isinstance(r.get(m), (int, float)) for r in sub)
            ]
            for metric in metrics:
                vals = [
                    float(r[metric]) if isinstance(r.get(metric), (int, float))
                    else float("nan")
                    for r in sub
                ]
                figures += plot_grouped_bars(
                    settings, {metric: vals},
                    f"{prefix}_{name}_{metric.replace('+', 'p')}",
                    title=f"{name} ablation: {metric}",
                    x_label=name, y_label=metric, formats=formats,
                )
            figures += plot_table(
                [[r.get(c, "—") if r.get(c) is not None else "—" for c in fields]
                 for r in sub],
                fields,
                f"{prefix}_{name}_table",
                title=f"{name} ablation",
                formats=formats,
            )

    recorder = ResultsRecorder(
        args, run_name="ablations", script="run_ablations.py"
    )
    recorder.add_source(*figures, args.out_csv)
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)):
                recorder.add_metrics(**{f"{r['ablation']}_{r['setting']}_{k}": v})
    recorder.add_info(
        ablations=args.ablation,
        fraction=args.fraction,
        epochs=args.epochs,
        variants=args.variants,
        decode_methods=args.decode_methods,
    )
    recorder.publish()


if __name__ == "__main__":
    main()
