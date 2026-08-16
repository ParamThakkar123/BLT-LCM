"""
BLT-LCM Machine Translation: English -> Marathi at the concept level.

This is the ACTUAL translation task the paper claims. Earlier the BLT/SONAR-LCM
path only used the Marathi column and predicted the *next Marathi sentence*
(monolingual language modeling), which does not match a "Machine Translation"
paper. BhashaSetu is a parallel corpus with ``english`` and ``marathi`` columns,
so we use both:

  1. Encode the English source sentence to a concept vector (BLT encoder;
     byte-level, so Latin script is handled directly).
  2. Encode the Marathi target sentence to a concept vector (the training target).
  3. Train BaseLCM to map source concept -> target concept (MSE regression). The
     source concept is the cross-attention memory; BaseLCM already has this
     encoder-decoder structure, so a single source concept predicts a single
     target concept.
  4. At evaluation, decode the predicted target concept back to Marathi text with
     the trained generative BLTDecoder (the LCM->SONAR-decoder analog) and score
     BLEU / chrF++ / TER (+ optional METEOR / COMET) against the reference.

Prerequisites (share ONE concept space across all three):
  * a learned pooler + BLTDecoder from ``blt_decoder.py`` (trained on Marathi),
    passed via ``--pooler`` / ``--decoder``.

Note: the entropy model here was trained on Marathi, so English patch boundaries
are approximate. Byte-level encoding still works; a multilingual byte encoder is
the natural quality improvement (left as future work).

Usage:
  python lcm_scripts/train_lcm_blt_mt.py \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --pooler lcm_models/blt_pooler.pth \
    --decoder lcm_models/blt_decoder.pth \
    --fraction 0.25 --epochs 3 \
    --noise_levels 0.0 0.1 0.2 \
    --out_csv results/blt_lcm_mt_25.csv
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import argparse
import csv
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from blt_loader import BLTLoader
from blt_decoder import load_decoder
from base_lcm import BaseLCM
from eval_metrics import compute_all
from bhashasetu_utils import load_bhashasetu_pairs, add_character_noise
from device_utils import report_device
from plot_utils import (
    TrainingHistory,
    add_plot_args,
    plot_formats,
    plot_noise_curves,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from train_control import EpochBudget, add_epoch_control_args
from checkpoint_utils import (
    DEFAULT_FINGERPRINT_IGNORE,
    ResumableLoader,
    StageTracker,
    TrainingCheckpointer,
    add_resume_args,
    cached_torch,
    config_fingerprint,
)


def set_seed(seed: int) -> None:
    """Seed every RNG that affects training, so --seed gives real run-to-run variance."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_concepts(blt, texts, batch_size=128, desc="encoding"):
    """Frozen concept encoding; returns a list of [dim] CPU tensors."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        embs.extend([e.detach().cpu() for e in blt.encode_sentences_batch(batch)])
    return embs


def main():
    parser = argparse.ArgumentParser(
        description="Train + evaluate BLT-LCM English->Marathi machine translation"
    )
    parser.add_argument("--entropy_model", required=True)
    parser.add_argument("--pooler", default="lcm_models/blt_pooler.pth")
    parser.add_argument("--decoder", default="lcm_models/blt_decoder.pth")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--eval_examples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--src_col", default="english", help="Source (input) column")
    parser.add_argument("--tgt_col", default="marathi", help="Target (reference) column")
    parser.add_argument(
        "--noise_levels", type=float, nargs="+", default=[0.0, 0.1, 0.2],
        help="Character-noise levels applied to the SOURCE input at eval time",
    )
    parser.add_argument("--model_dir", default="lcm_models")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Training seed: model init, batch shuffling and eval-noise draws. "
             "Vary this (e.g. 42/43/44) to get genuine error bars.",
    )
    parser.add_argument(
        "--data_seed", type=int, default=42,
        help="Seed for the corpus fraction/split. Keep FIXED across --seed runs "
             "so every seed sees the same train/eval split.",
    )
    parser.add_argument("--max_decode_len", type=int, default=256)
    parser.add_argument("--comet_model", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--embed_cache",
        default=None,
        help="Optional path for the cached frozen train-concept encodings. Set "
        "this so a resumed run reuses them instead of re-encoding the corpus.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run the forward/backward in bfloat16 autocast on CUDA. Roughly "
        "halves step time on Ampere and newer at a small numerical cost; the "
        "loss itself is still reduced in fp32.",
    )
    add_resume_args(parser, default_interval_steps=200)
    add_plot_args(parser)
    add_epoch_control_args(parser)
    add_results_args(parser)
    args = parser.parse_args()

    device = report_device(args.device)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    if args.amp and amp_device != "cuda":
        print("  [amp] no CUDA device; running in fp32")
        args.amp = False
    # --batch_size is a VRAM knob, not part of what the run computes: the
    # auto-setup driver halves it and retries after a CUDA OOM, and that retry
    # has to resume the interrupted run (and reuse the cached encodings) rather
    # than abort on a fingerprint mismatch. Every other flag still invalidates.
    fingerprint = config_fingerprint(
        args, ignore=DEFAULT_FINGERPRINT_IGNORE | {"batch_size"}
    )
    # One identity for every file this run writes -- checkpoints, eval state,
    # training curve, figures and the published results directory. It carries
    # BOTH the fraction and the seed because the grid runs 3 fractions x 3 seeds
    # into a single --model_dir: with the seed alone, the 0.50 run resolved to
    # the 0.25 run's checkpoint and died on the fingerprint check, while the
    # curves and eval state of the earlier fraction were silently overwritten.
    run_name = f"lcm_blt_mt_fraction{args.fraction:g}_s{args.seed}"
    if not args.comet_model:
        print(
            "WARNING: --comet_model not set, so the COMET column will be NaN. "
            "Pass --comet_model Unbabel/wmt22-comet-da to report COMET."
        )
    set_seed(args.seed)
    print(f"seed={args.seed} (training)  data_seed={args.data_seed} (corpus split)")

    # --- Parallel data (English -> Marathi) ---
    # data_seed is deliberately independent of --seed: the train/eval split must
    # stay identical across seeds, otherwise seed-to-seed spread would conflate
    # training variance with a changing evaluation set.
    print("Loading BhashaSetu parallel pairs (English -> Marathi)...")
    pairs = load_bhashasetu_pairs(
        fraction=args.fraction,
        max_examples=args.max_examples,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
        seed=args.data_seed,
    )
    if len(pairs) < args.eval_examples + 10:
        raise ValueError(
            f"Only {len(pairs)} parallel pairs found; need more than "
            f"eval_examples ({args.eval_examples}). Increase --fraction, or check "
            f"--src_col/--tgt_col match the dataset schema."
        )
    eval_pairs = pairs[-args.eval_examples :]
    train_pairs = pairs[: -args.eval_examples]
    print(f"Train pairs: {len(train_pairs)}  |  Eval pairs: {len(eval_pairs)}")

    # --- Encoder (+ learned pooler) and generative decoder ---
    blt = BLTLoader(
        entropy_model_path=args.entropy_model,
        device=str(device),
        pooler_path=args.pooler,
    )
    if not os.path.exists(args.decoder):
        raise FileNotFoundError(
            f"Machine-translation eval needs a trained BLTDecoder at "
            f"'{args.decoder}'. Train one first (jointly with the pooler):\n"
            f"  python lcm_scripts/blt_decoder.py --entropy_model {args.entropy_model} "
            f"--pooler_save_path {args.pooler} --save_path {args.decoder}"
        )
    decoder = load_decoder(args.decoder, device=str(device))
    if decoder.embed_dim != blt.dim:
        raise ValueError(
            f"Decoder embed_dim ({decoder.embed_dim}) != concept dim ({blt.dim}). "
            "Retrain the decoder/pooler with the current encoder."
        )

    # --- Encode train concepts once (frozen encoder) ---
    # The encoder is frozen, so these tensors depend only on the config: cache
    # them and a resumed run skips straight back to the optimizer loop.
    #
    # The cache key covers exactly what the encoding consumes, NOT the whole
    # config. In particular it excludes --seed: encoding is a no_grad forward
    # pass through eval-mode modules, so the concepts are identical for every
    # seed of a fraction, and the seed governs only the LCM's init, its batch
    # order and the eval-noise draw. Keyed on the full config instead, the 3
    # seeds x 3 fractions grid re-encoded the same corpus nine times -- 4.2
    # GPU-hours and 99 GiB of cache to produce three identical copies per
    # fraction. Anything that does change the tensors (the encoder, the pooler,
    # the corpus slice, the columns) is still in the key.
    encode_fingerprint = config_fingerprint(
        {
            "entropy_model": args.entropy_model,
            "pooler": args.pooler,
            "fraction": args.fraction,
            "max_examples": args.max_examples,
            "eval_examples": args.eval_examples,
            "src_col": args.src_col,
            "tgt_col": args.tgt_col,
            "data_seed": args.data_seed,
        },
        ignore=(),
        extra={"stage": "mt-train-concepts"},
    )
    src_tr, tgt_tr = cached_torch(
        args.embed_cache,
        lambda: (
            torch.stack(
                encode_concepts(
                    blt, [p.source for p in train_pairs], desc="encode src (train)"
                )
            ),
            torch.stack(
                encode_concepts(
                    blt, [p.target for p in train_pairs], desc="encode tgt (train)"
                )
            ),
        ),
        fingerprint=encode_fingerprint,
        resume=args.resume != "never",
        label="train concepts",
    )

    dataset = torch.utils.data.TensorDataset(src_tr, tgt_tr)
    loader = ResumableLoader(
        dataset, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )

    # --- LCM: source concept -> target concept ---
    lcm = BaseLCM(embed_dim=blt.dim, model_dim=2048, n_layers=12, n_heads=16).to(device)
    optim = torch.optim.AdamW(lcm.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    os.makedirs(args.model_dir, exist_ok=True)
    # Fraction + seed in the prefix: runs of the grid would otherwise clobber
    # each other's checkpoints, and a resumed run would pick up another cell's
    # state (or, because the fingerprint covers both, refuse to start at all).
    ckpt = TrainingCheckpointer(
        args.model_dir,
        prefix=run_name,
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(ckpt.load(args.resume, map_location=device), lcm, optim)
    global_step = resume.global_step
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}"
        )

    # Curve for this grid cell. Fraction and seed are both in the name, so a
    # sweep produces one figure per cell rather than overwriting a shared one.
    plot_dir = resolve_plot_dir(args, args.model_dir)
    history = TrainingHistory(
        plot_dir,
        run_name=run_name,
        title=f"BLT-LCM En→Mr (fraction {args.fraction}, seed {args.seed})",
        fingerprint=fingerprint,
        resume=args.resume != "never",
        formats=plot_formats(args),
        loss_label="Concept MSE",
    )

    budget = EpochBudget.from_args(args, history=history, label="concept MSE")

    for epoch in budget.epochs_from(resume.start_epoch):
        lcm.train()
        total, n = 0.0, 0
        start = time.time()
        skip = resume.batches_to_skip(epoch)
        for batch_idx, (src, tgt) in tqdm(
            loader.epoch(epoch, skip=skip),
            desc=f"epoch {budget.describe(epoch)}",
            initial=skip,
            total=len(loader),
        ):
            # [B, 1, dim] source concept (memory) and target concept
            src = src.to(device, non_blocking=True).unsqueeze(1)
            tgt = tgt.to(device, non_blocking=True).unsqueeze(1)
            optim.zero_grad()
            # bf16 needs no GradScaler; the MSE is reduced in fp32 because it is
            # not on autocast's promotion list.
            with torch.autocast(
                device_type=amp_device, dtype=torch.bfloat16, enabled=args.amp
            ):
                pred = lcm(src, tgt)               # [B, dim]
            loss = mse(pred.float(), tgt.squeeze(1))
            loss.backward()
            optim.step()
            step_loss = loss.item()
            total += step_loss
            n += 1
            global_step += 1
            # Sampled, not per-step: the sidecar stays small and the step-loss
            # panel is still dense enough to read.
            if global_step % 50 == 0:
                history.log_step(
                    global_step, step_loss, lr=optim.param_groups[0]["lr"]
                )
            ckpt.maybe_save(
                lcm,
                optim,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
            )
        elapsed = time.time() - start
        avg_loss = total / max(n, 1)
        print(
            f"Epoch {budget.describe(epoch)} | MSE {avg_loss:.5f} "
            f"| {elapsed:.1f}s"
        )
        history.log_epoch(
            epoch + 1, avg_loss, seconds=elapsed, lr=optim.param_groups[0]["lr"]
        )
        budget.observe(epoch, avg_loss)
        ckpt.save_epoch(lcm, optim, epoch=epoch, global_step=global_step)

    print(budget.summary())
    figures = history.plot()

    best_ckpt = ckpt.best_path
    ckpt.save_best(
        lcm,
        optim,
        epoch=args.epochs - 1,
        epoch_completed=True,
        global_step=global_step,
    )
    print(f"Saved LCM to {best_ckpt}")

    # --- Evaluate: English -> concept -> Marathi, across source-noise levels ---
    # Each noise level costs a full generative decode of the eval set, so
    # completed levels are memoized: an interrupted evaluation resumes at the
    # first level it had not finished.
    lcm.eval()
    refs = [p.target for p in eval_pairs]
    clean_srcs = [p.source for p in eval_pairs]
    stages = StageTracker(
        os.path.join(args.model_dir, f"{run_name}_eval_state.json"),
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    def _eval_noise(noise):
        # Per-sentence noise seeds derived from --seed, so (a) each sentence gets
        # an independent corruption rather than the same RNG stream, and (b) the
        # noise realisation changes across seeds, which is what makes the error
        # bars on the noisy conditions meaningful.
        noise_rng = random.Random(f"noise|{args.seed}|{noise}")
        src_texts = [
            add_character_noise(s, noise, seed=noise_rng.randrange(2**31))
            if noise > 0
            else s
            for s in clean_srcs
        ]
        src_emb = torch.stack(
            encode_concepts(blt, src_texts, desc=f"encode src (noise={noise})")
        ).to(device)

        hyps = []
        for i in tqdm(
            range(0, len(src_emb), args.batch_size), desc=f"translate (noise={noise})"
        ):
            batch = src_emb[i : i + args.batch_size].unsqueeze(1)  # [b, 1, dim]
            with torch.no_grad():
                pred = lcm(batch)  # [b, dim] predicted target concept
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                hyps.extend(decoder.decode(pred, max_len=args.max_decode_len))

        # COMET is scored against the CLEAN source: the metric asks "does the
        # output convey the source meaning", and the true meaning is the
        # uncorrupted sentence. Feeding it the noised source would move the
        # target the model is being judged against.
        return compute_all(
            hyps,
            refs,
            srcs=clean_srcs,
            comet_model_name=args.comet_model,
            device=str(device),
        )

    rows = []
    for noise in args.noise_levels:
        metrics = stages.run(f"noise={noise}", lambda noise=noise: _eval_noise(noise))
        print(
            f"seed={args.seed} noise={noise}: "
            + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items() if v == v)
        )
        rows.append(
            {
                "model": "blt_lcm_mt",
                "fraction": args.fraction,
                "noise": noise,
                "seed": args.seed,
                **metrics,
            }
        )

    # Robustness figures: every metric against the source-noise level, plus the
    # same numbers as a drop-in table image.
    if plot_dir and rows:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, run_name)
        figures += plot_noise_curves(
            rows,
            f"{prefix}_noise_robustness",
            title=(
                f"BLT-LCM En→Mr robustness to source noise "
                f"(fraction {args.fraction}, seed {args.seed})"
            ),
            metrics=("BLEU", "chrF++", "TER", "METEOR", "COMET"),
            formats=formats,
        )
        metric_names = ["BLEU", "chrF++", "TER", "METEOR", "COMET"]
        figures += plot_table(
            [
                [f"{r['noise']:.0%}"]
                + [
                    f"{r[m]:.2f}" if isinstance(r.get(m), float) and r[m] == r[m] else "—"
                    for m in metric_names
                ]
                for r in rows
            ],
            ["Source noise"] + metric_names,
            f"{prefix}_metrics_table",
            title=f"BLT-LCM En→Mr metrics (fraction {args.fraction}, seed {args.seed})",
            formats=formats,
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fieldnames = [
            "model", "fraction", "noise", "seed",
            "BLEU", "chrF++", "TER", "METEOR", "COMET",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved metrics to {args.out_csv}")

    # Publish figures, history, metrics CSV and the full hyperparameter set.
    recorder = ResultsRecorder(
        args,
        run_name=run_name,
        script="train_lcm_blt_mt.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, history.json_path, args.out_csv or "")
    clean = next((r for r in rows if r["noise"] == 0.0), rows[0] if rows else {})
    recorder.add_metrics(
        final_train_loss=budget.best,
        best_epoch=budget.best_epoch,
        epochs_run=budget.observed,
        **{f"clean_{k}": v for k, v in clean.items() if isinstance(v, float)},
    )
    recorder.add_info(
        seed=args.seed,
        data_seed=args.data_seed,
        train_pairs=len(train_pairs),
        eval_pairs=len(eval_pairs),
        noise_levels=args.noise_levels,
        **budget.as_dict(),
    )
    recorder.publish()


if __name__ == "__main__":
    main()
