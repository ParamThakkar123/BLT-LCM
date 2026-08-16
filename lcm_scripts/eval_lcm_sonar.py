"""Evaluate an existing SONAR-embedding + BaseLCM checkpoint on BhashaSetu.

This script fills the gap where a SONAR-LCM training run produced checkpoints or
TensorBoard logs but no metrics CSV. It rebuilds the deterministic BhashaSetu
document split used by ``train_lcm_sonar.py``, loads one checkpoint, decodes next
sentence predictions by nearest-neighbor retrieval, and writes BLEU, chrF++ and
TER for the requested clean/noisy settings.

Example:
  uv run lcm_scripts/eval_lcm_sonar.py \
    --checkpoint runs/lcm_sonar/lcm_sonar_fraction0.25_epoch2.pth \
    --fraction 0.25 \
    --eval_docs 100 \
    --noise_levels 0.0 0.10 0.20 \
    --out_csv runs/lcm_sonar/metrics_fraction0.25.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

import torch
from tqdm import tqdm

from base_lcm import BaseLCM
from bhashasetu_utils import (
    DEFAULT_DATASET,
    DEFAULT_NOISE_LEVELS,
    add_character_noise,
    load_bhashasetu_documents,
    split_train_eval_documents,
)
from embedding_retriever import EmbeddingRetriever
from eval_metrics import compute_bleu, compute_chrf, compute_ter
from sonar_loader import SonarLoader
from train_lcm_sonar import encode_docs
from device_utils import report_device
from plot_utils import (
    add_plot_args,
    plot_formats,
    plot_noise_curves,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from checkpoint_utils import (
    StageTracker,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    load_model_state,
)


def evaluate(model, docs, encoder, retriever, args, noise: float, device) -> dict[str, float | int]:
    hyps, refs = [], []
    model.eval()
    for doc_idx, doc in enumerate(tqdm(docs, desc=f"eval noise={noise:.2f}")):
        if len(doc) < args.min_prefix + 1:
            continue
        noisy_doc = (
            [add_character_noise(s, noise, seed=doc_idx * 1000 + i) for i, s in enumerate(doc)]
            if noise
            else doc
        )
        embs = encoder.encode_sentences(noisy_doc).detach().cpu()
        for i in range(args.min_prefix, len(doc)):
            with torch.no_grad():
                pred = model(embs[:i].unsqueeze(0).to(device))
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            hyps.append(retriever.retrieve(pred)[0])
            refs.append(doc[i])
    return {
        "num_predictions": len(hyps),
        "BLEU": compute_bleu(hyps, refs),
        "chrF++": compute_chrf(hyps, refs),
        "TER": compute_ter(hyps, refs),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an existing SONAR-LCM checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to lcm_sonar_*.pth checkpoint")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--eval_docs", type=int, default=100)
    p.add_argument("--max_sent_per_doc", type=int, default=20)
    p.add_argument("--text_col", default="marathi")
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--min_prefix", type=int, default=2)
    p.add_argument("--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    p.add_argument("--out_csv", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--embed_cache",
        default=None,
        help="Optional cache for the SONAR encodings of the retrieval corpus, "
        "so a rerun skips re-encoding it.",
    )
    add_resume_args(p, training=False)
    add_plot_args(p)
    add_results_args(p)
    args = p.parse_args()

    device = report_device(args.device)
    fingerprint = config_fingerprint(args, extra={"stage": "eval_lcm_sonar"})
    docs = load_bhashasetu_documents(
        args.dataset,
        args.split,
        args.fraction,
        args.max_sent_per_doc,
        args.text_col,
    )
    train_docs, eval_docs = split_train_eval_documents(docs, args.eval_docs)
    if not train_docs or not eval_docs:
        raise RuntimeError("No BhashaSetu documents available for SONAR-LCM evaluation")

    encoder = SonarLoader(device=str(device))
    train_seqs = cached_torch(
        args.embed_cache,
        lambda: encode_docs(train_docs, encoder, args.encode_batch_size),
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=lambda s: bool(s),
        label="SONAR retrieval-corpus encodings",
    )
    flat_sents = [s for doc in train_docs for s in doc]
    flat_embs = torch.cat([s for s in train_seqs if s.shape[0] > 0], dim=0)
    retriever = EmbeddingRetriever(flat_sents, flat_embs)

    embed_dim = train_seqs[0].shape[1]
    model = BaseLCM(embed_dim=embed_dim, model_dim=args.model_dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    # Accepts both the resumable checkpoint payload and older bare state dicts.
    model.load_state_dict(load_model_state(args.checkpoint, map_location=device))

    # Each noise level is a full retrieval decode of the eval set; memoize the
    # ones that finish so an interrupted run does not redo them.
    stages = StageTracker(
        str(Path(args.checkpoint).with_name(f"eval_state_fraction{args.fraction}.json")),
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )
    rows = []
    for noise in args.noise_levels:
        metrics = stages.run(
            f"noise={noise}",
            lambda noise=noise: evaluate(
                model, eval_docs, encoder, retriever, args, noise, device
            ),
        )
        row = {
            "model": "sonar_lcm",
            "fraction": args.fraction,
            "noise": noise,
            "checkpoint": os.path.basename(args.checkpoint),
            **metrics,
        }
        rows.append(row)
        print(row)

    out_csv = args.out_csv or str(Path(args.checkpoint).with_name(f"metrics_fraction{args.fraction}.csv"))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "fraction", "noise", "checkpoint", "num_predictions", "BLEU", "chrF++", "TER"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")

    plot_dir = resolve_plot_dir(args, os.path.dirname(out_csv) or ".")
    figures: list[str] = []
    if plot_dir and rows:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, f"eval_lcm_sonar_fraction{args.fraction}")
        figures += plot_noise_curves(
            rows,
            f"{prefix}_noise_robustness",
            title=(
                f"SONAR-LCM robustness to input noise "
                f"({os.path.basename(args.checkpoint)}, fraction {args.fraction})"
            ),
            formats=formats,
        )
        figures += plot_table(
            [
                [
                    f"{r['noise']:.0%}",
                    f"{r['num_predictions']:,}",
                    f"{r['BLEU']:.2f}",
                    f"{r['chrF++']:.2f}",
                    f"{r['TER']:.2f}",
                ]
                for r in rows
            ],
            ["Input noise", "Predictions", "BLEU ↑", "chrF++ ↑", "TER ↓"],
            f"{prefix}_metrics_table",
            title=f"SONAR-LCM metrics (fraction {args.fraction})",
            formats=formats,
        )

    recorder = ResultsRecorder(
        args,
        run_name=f"eval_lcm_sonar_fraction{args.fraction}",
        script="eval_lcm_sonar.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, out_csv)
    clean = next((r for r in rows if r["noise"] == 0.0), rows[0] if rows else {})
    recorder.add_metrics(
        **{f"clean_{k}": v for k, v in clean.items() if isinstance(v, (int, float))}
    )
    recorder.add_info(
        checkpoint=args.checkpoint,
        fraction=args.fraction,
        eval_docs=len(eval_docs),
        noise_levels=args.noise_levels,
    )
    recorder.publish()


if __name__ == "__main__":
    main()
