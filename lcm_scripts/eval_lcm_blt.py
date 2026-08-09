"""
Evaluate a trained BLT-LCM model.

Pipeline:
  1. Load trained LCM + BLT encoder (+ learned pooler + byte decoder)
  2. For each document, feed prefix sentences → LCM predicts next concept
  3. Turn each predicted concept back into text. This is done GENERATIVELY by
     default, with the trained BLTDecoder (the analog of LCM's SONAR decoder,
     §2.4.1) — not by nearest-neighbour retrieval. Retrieval is kept only as an
     explicit ``--decode_method retrieval`` baseline.
  4. Compute BLEU/chrF++/METEOR/TER against the ground-truth next sentence

Usage:
  python lcm_scripts/eval_lcm_blt.py \
    --lcm_checkpoint lcm_models/lcm_blt_best.pth \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --pooler lcm_models/blt_pooler.pth \
    --decoder lcm_models/blt_decoder.pth \
    --fraction 0.25
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import torch
from tqdm import tqdm

from base_lcm import BaseLCM
from blt_loader import BLTLoader
from blt_decoder import load_decoder
from embedding_retriever import EmbeddingRetriever
from eval_metrics import compute_all
from checkpoint_utils import (
    ResumableJsonl,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    load_model_state,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLT-LCM with NN retrieval")
    parser.add_argument("--lcm_checkpoint", type=str, required=True)
    parser.add_argument("--entropy_model", type=str, required=True)
    parser.add_argument(
        "--embed_cache",
        type=str,
        default="blt_eval_embeddings_cache.pth",
        help="Cache for the encoded EVAL documents. Deliberately distinct from "
        "train_lcm_blt.py's --embed_cache, which holds the TRAIN split.",
    )
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--max_sent_per_doc", type=int, default=20)
    parser.add_argument(
        "--min_prefix",
        type=int,
        default=2,
        help="Minimum prefix sentences before predicting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional CSV path to save per-sample results",
    )
    parser.add_argument("--comet_model", type=str, default=None)
    parser.add_argument(
        "--pooler",
        type=str,
        default="lcm_models/blt_pooler.pth",
        help="Path to a learned cross-attention pooler (from blt_decoder training). "
        "Must match the pooler used to train the LCM checkpoint.",
    )
    parser.add_argument(
        "--decode_method",
        choices=["generative", "retrieval"],
        default="generative",
        help="How to turn a predicted concept into text. 'generative' uses the "
        "trained BLTDecoder (default, faithful to LCM); 'retrieval' is the "
        "nearest-neighbour baseline.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="lcm_models/blt_decoder.pth",
        help="Path to the trained BLTDecoder checkpoint (for generative decoding).",
    )
    parser.add_argument(
        "--max_decode_len",
        type=int,
        default=256,
        help="Max bytes to generate per sentence during generative decoding.",
    )
    parser.add_argument(
        "--decode_batch_size",
        type=int,
        default=32,
        help="Batch size for generative decoding of predicted concepts.",
    )
    parser.add_argument(
        "--progress_jsonl",
        type=str,
        default=None,
        help="Where to stream per-prediction (hypothesis, reference) rows. "
        "Defaults next to --out_csv, or 'results/eval_lcm_blt_progress.jsonl'. "
        "A rerun replays this file and only decodes what is missing.",
    )
    add_resume_args(parser, training=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    fingerprint = config_fingerprint(args, extra={"stage": "eval_lcm_blt"})

    # --- Load data ---
    from bhashasetu_utils import load_bhashasetu_documents, split_train_eval_documents

    print("Loading documents...")
    all_docs = load_bhashasetu_documents(
        fraction=args.fraction,
        max_sent_per_doc=args.max_sent_per_doc,
        text_col="marathi",
    )
    print(f"Loaded {len(all_docs)} docs total")

    # Split into train (retrieval corpus) and eval (evaluation set).
    # The retriever must NEVER contain eval sentences, otherwise the
    # nearest-neighbour search can return the ground-truth sentence and
    # artificially inflate scores.
    eval_count = min(100, len(all_docs) // 5)
    train_docs, eval_docs = split_train_eval_documents(all_docs, eval_count)
    train_sents = [s for doc in train_docs for s in doc]
    print(
        f"Train docs: {len(train_docs)} ({len(train_sents)} sentences), "
        f"Eval docs: {len(eval_docs)}"
    )

    # --- Load BLT encoder (+ learned pooler) ---
    print("Loading BLT encoder...")
    blt = BLTLoader(
        entropy_model_path=args.entropy_model,
        device=str(device),
        pooler_path=args.pooler,
    )

    # --- Set up the readout (generative decoder or retrieval baseline) ---
    decoder = None
    retriever = None
    if args.decode_method == "generative":
        if not os.path.exists(args.decoder):
            raise FileNotFoundError(
                f"Generative decoding requires a trained BLTDecoder at "
                f"'{args.decoder}'. Train one first:\n"
                f"  python lcm_scripts/blt_decoder.py --entropy_model {args.entropy_model} "
                f"--pooler_save_path {args.pooler}\n"
                f"Or run eval with --decode_method retrieval for the NN baseline."
            )
        print(f"Loading generative BLT decoder from {args.decoder}")
        decoder = load_decoder(args.decoder, device=str(device))
        if decoder.embed_dim != blt.dim:
            raise ValueError(
                f"Decoder embed_dim ({decoder.embed_dim}) != concept dim "
                f"({blt.dim}). Retrain the decoder with the current encoder/pooler."
            )
    else:
        # Retrieval baseline: index TRAIN sentences only (never eval sentences,
        # or NN could return the ground truth and inflate scores).
        print("Encoding retrieval corpus from train documents...")
        retriever = EmbeddingRetriever.from_corpus(
            train_sents, blt, batch_size=64, device=str(device)
        )

    # --- Load LCM ---
    print(f"Loading LCM from {args.lcm_checkpoint}")
    embed_dim = blt.dim
    lcm = BaseLCM(embed_dim=embed_dim, model_dim=2048, n_layers=12, n_heads=16).to(
        device
    )
    # `load_model_state` accepts both the resumable checkpoint payload written
    # by train_lcm_blt.py and the bare state_dict files from older runs.
    lcm.load_state_dict(load_model_state(args.lcm_checkpoint, map_location=device))
    lcm.eval()

    # --- Encode eval-document embeddings ---
    print("Encoding eval document embeddings...")

    def _encode_eval_docs():
        out = []
        for doc_sents in tqdm(eval_docs, desc="Encoding eval docs"):
            embs = blt.encode_sentences_batch(doc_sents)
            out.append(torch.stack([e.detach().cpu() for e in embs]))
        return out

    eval_doc_embeddings = cached_torch(
        args.embed_cache,
        _encode_eval_docs,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=lambda seqs: bool(seqs),
        label="eval document embeddings",
    )

    # --- Predict next-concept embeddings on EVAL documents only ---
    print("Running LCM predictions...")
    pred_embs = []  # list of [embed_dim] tensors (cpu)
    refs = []

    for doc_sents, doc_embs in tqdm(
        zip(eval_docs, eval_doc_embeddings), total=len(eval_docs), desc="Predicting"
    ):
        n = len(doc_sents)
        if n < args.min_prefix + 1:
            continue

        for i in range(args.min_prefix, n):
            prefix_embs = doc_embs[:i].unsqueeze(0).to(device)

            with torch.no_grad():
                pred_emb = lcm(prefix_embs)

            if pred_emb.dim() == 1:
                pred_emb = pred_emb.unsqueeze(0)

            pred_embs.append(pred_emb.squeeze(0).detach().cpu())
            refs.append(doc_sents[i])

    # --- Turn predicted concepts into text ---
    # Generative decoding is byte-at-a-time over thousands of predictions and
    # dominates the runtime here, so each decoded batch is streamed to JSONL as
    # it completes. A rerun replays that file and decodes only what is missing.
    progress_path = args.progress_jsonl or (
        os.path.splitext(args.out_csv)[0] + ".progress.jsonl"
        if args.out_csv
        else "results/eval_lcm_blt_progress.jsonl"
    )
    print(f"\nDecoding {len(pred_embs)} predictions ({args.decode_method})...")
    writer = ResumableJsonl(
        progress_path,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        key="index",
    )
    if pred_embs:
        pred_stack = torch.stack(pred_embs)  # [N, embed_dim]
        for i in tqdm(
            range(0, len(pred_stack), args.decode_batch_size), desc="Decoding"
        ):
            idx = list(range(i, min(i + args.decode_batch_size, len(pred_stack))))
            if all(writer.is_done(j) for j in idx):
                continue
            batch = pred_stack[i : i + args.decode_batch_size]
            if args.decode_method == "generative":
                with torch.no_grad():
                    decoded = decoder.decode(
                        batch.to(device), max_len=args.max_decode_len
                    )
            else:
                decoded = retriever.retrieve(batch)
            for j, hyp in zip(idx, decoded):
                if not writer.is_done(j):
                    writer.append({"index": j, "hypothesis": hyp, "reference": refs[j]})
    writer.close()

    rows = writer.all_records()
    hyps = [r["hypothesis"] for r in rows]
    refs = [r["reference"] for r in rows]

    metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)

    print(f"\n{'=' * 50}")
    print(f"  BLT-LCM Evaluation Results ({args.decode_method} decoding)")
    print(f"{'=' * 50}")
    print(f"  Samples evaluated : {len(hyps)}")
    for k, v in metrics.items():
        if v == v:  # skip NaN
            print(f"  {k:12s}: {v:.2f}")
    print(f"{'=' * 50}")

    # --- Optionally save per-sample CSV ---
    if args.out_csv:
        import csv

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["hypothesis", "reference"])
            for h, r in zip(hyps, refs):
                writer.writerow([h, r])
        print(f"Saved per-sample results to {args.out_csv}")

    return metrics


if __name__ == "__main__":
    main()
