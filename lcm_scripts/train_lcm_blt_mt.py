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
import time

import torch
from tqdm import tqdm

from blt_loader import BLTLoader
from blt_decoder import load_decoder
from base_lcm import BaseLCM
from eval_metrics import compute_all
from bhashasetu_utils import load_bhashasetu_pairs, add_character_noise


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
    parser.add_argument("--max_decode_len", type=int, default=256)
    parser.add_argument("--comet_model", default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Parallel data (English -> Marathi) ---
    print("Loading BhashaSetu parallel pairs (English -> Marathi)...")
    pairs = load_bhashasetu_pairs(
        fraction=args.fraction,
        max_examples=args.max_examples,
        src_col=args.src_col,
        tgt_col=args.tgt_col,
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
    src_tr = torch.stack(
        encode_concepts(blt, [p.source for p in train_pairs], desc="encode src (train)")
    )
    tgt_tr = torch.stack(
        encode_concepts(blt, [p.target for p in train_pairs], desc="encode tgt (train)")
    )

    dataset = torch.utils.data.TensorDataset(src_tr, tgt_tr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    # --- LCM: source concept -> target concept ---
    lcm = BaseLCM(embed_dim=blt.dim, model_dim=2048, n_layers=12, n_heads=16).to(device)
    optim = torch.optim.AdamW(lcm.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()

    os.makedirs(args.model_dir, exist_ok=True)
    for epoch in range(args.epochs):
        lcm.train()
        total, n = 0.0, 0
        start = time.time()
        for src, tgt in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            src = src.to(device).unsqueeze(1)  # [B, 1, dim] source concept (memory)
            tgt = tgt.to(device).unsqueeze(1)  # [B, 1, dim] target concept
            optim.zero_grad()
            pred = lcm(src, tgt)               # [B, dim]
            loss = mse(pred, tgt.squeeze(1))
            loss.backward()
            optim.step()
            total += loss.item()
            n += 1
        print(
            f"Epoch {epoch + 1}/{args.epochs} | MSE {total / max(n, 1):.5f} "
            f"| {time.time() - start:.1f}s"
        )
        torch.save(
            lcm.state_dict(), f"{args.model_dir}/lcm_blt_mt_epoch{epoch + 1}.pth"
        )
    best_ckpt = f"{args.model_dir}/lcm_blt_mt_best.pth"
    torch.save(lcm.state_dict(), best_ckpt)
    print(f"Saved LCM to {best_ckpt}")

    # --- Evaluate: English -> concept -> Marathi, across source-noise levels ---
    lcm.eval()
    refs = [p.target for p in eval_pairs]
    rows = []
    for noise in args.noise_levels:
        src_texts = [
            add_character_noise(p.source, noise, seed=1234) if noise > 0 else p.source
            for p in eval_pairs
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

        metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)
        print(
            f"noise={noise}: "
            + "  ".join(f"{k}={v:.2f}" for k, v in metrics.items() if v == v)
        )
        rows.append(
            {"model": "blt_lcm_mt", "fraction": args.fraction, "noise": noise, **metrics}
        )

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        fieldnames = [
            "model", "fraction", "noise", "BLEU", "chrF++", "TER", "METEOR", "COMET"
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved metrics to {args.out_csv}")


if __name__ == "__main__":
    main()
