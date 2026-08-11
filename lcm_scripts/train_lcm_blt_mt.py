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
from checkpoint_utils import (
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
    add_resume_args(parser, default_interval_steps=200)
    args = parser.parse_args()

    device = report_device(args.device)
    fingerprint = config_fingerprint(args)
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
        fingerprint=fingerprint,
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
    # Seed in the prefix: multi-seed runs would otherwise clobber each other's
    # checkpoints, and a resumed run would pick up the wrong seed's state.
    ckpt = TrainingCheckpointer(
        args.model_dir,
        prefix=f"lcm_blt_mt_s{args.seed}",
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

    for epoch in range(resume.start_epoch, args.epochs):
        lcm.train()
        total, n = 0.0, 0
        start = time.time()
        skip = resume.batches_to_skip(epoch)
        for batch_idx, (src, tgt) in tqdm(
            loader.epoch(epoch, skip=skip),
            desc=f"epoch {epoch + 1}/{args.epochs}",
            initial=skip,
            total=len(loader),
        ):
            src = src.to(device).unsqueeze(1)  # [B, 1, dim] source concept (memory)
            tgt = tgt.to(device).unsqueeze(1)  # [B, 1, dim] target concept
            optim.zero_grad()
            pred = lcm(src, tgt)               # [B, dim]
            loss = mse(pred, tgt.squeeze(1))
            loss.backward()
            optim.step()
            total += loss.item()
            n += 1
            global_step += 1
            ckpt.maybe_save(
                lcm,
                optim,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
            )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | MSE {total / max(n, 1):.5f} "
            f"| {time.time() - start:.1f}s"
        )
        ckpt.save_epoch(lcm, optim, epoch=epoch, global_step=global_step)

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
        os.path.join(args.model_dir, f"lcm_blt_mt_s{args.seed}_eval_state.json"),
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
        return compute_all(hyps, refs, srcs=clean_srcs, comet_model_name=args.comet_model)

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


if __name__ == "__main__":
    main()
