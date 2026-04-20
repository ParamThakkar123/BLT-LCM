"""
Evaluate a trained BLT-LCM model using nearest-neighbor retrieval.

Pipeline:
  1. Load trained LCM + BLT encoder
  2. For each document, feed prefix sentences → LCM predicts next embedding
  3. Retrieve nearest real sentence for each predicted embedding
  4. Compute BLEU/chrF++/METEOR/TER against ground-truth next sentence

Usage:
  python lcm_scripts/eval_lcm_blt.py \
    --lcm_checkpoint lcm_models/lcm_blt_best.pth \
    --entropy_model patching_scratch/entropy_model_marathi.pt \
    --embed_cache blt_embeddings_cache.pth \
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
from embedding_retriever import EmbeddingRetriever
from eval_metrics import compute_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLT-LCM with NN retrieval")
    parser.add_argument("--lcm_checkpoint", type=str, required=True)
    parser.add_argument("--entropy_model", type=str, required=True)
    parser.add_argument("--embed_cache", type=str, default="blt_embeddings_cache.pth")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--max_sent_per_doc", type=int, default=20)
    parser.add_argument("--min_prefix", type=int, default=2,
                        help="Minimum prefix sentences before predicting")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Optional CSV path to save per-sample results")
    parser.add_argument("--comet_model", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- Load data ---
    from train_lcm_blt import prepare_data
    print("Loading documents...")
    docs = prepare_data(args.num_docs, args.max_sent_per_doc, fraction=args.fraction)
    flat_sents = [s for doc in docs for s in doc]
    print(f"Loaded {len(docs)} docs, {len(flat_sents)} total sentences")

    # --- Build retrieval index ---
    print("Loading BLT encoder...")
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    if os.path.exists(args.embed_cache):
        print(f"Building retrieval index from cache: {args.embed_cache}")
        retriever = EmbeddingRetriever.from_cache(flat_sents, args.embed_cache)
    else:
        print("No embedding cache found, encoding corpus from scratch...")
        retriever = EmbeddingRetriever.from_corpus(
            flat_sents, blt, batch_size=64, device=str(device)
        )

    # --- Load LCM ---
    print(f"Loading LCM from {args.lcm_checkpoint}")
    embed_dim = retriever.embeddings.shape[1]
    lcm = BaseLCM(embed_dim=embed_dim, model_dim=2048, n_layers=12, n_heads=16).to(device)
    lcm.load_state_dict(
        torch.load(args.lcm_checkpoint, map_location=device, weights_only=False)
    )
    lcm.eval()

    # --- Encode per-document embeddings ---
    print("Encoding document embeddings for evaluation...")
    doc_embeddings = []
    for doc_sents in tqdm(docs, desc="Encoding docs"):
        embs = blt.encode_sentences_batch(doc_sents)
        doc_embeddings.append(torch.stack([e.detach().cpu() for e in embs]))

    # --- Run evaluation ---
    print("Running LCM predictions + retrieval...")
    hyps = []
    refs = []

    for doc_idx, (doc_sents, doc_embs) in enumerate(
        tqdm(zip(docs, doc_embeddings), total=len(docs), desc="Evaluating")
    ):
        n = len(doc_sents)
        if n < args.min_prefix + 1:
            continue

        for i in range(args.min_prefix, n):
            prefix_embs = doc_embs[:i].unsqueeze(0).to(device)  # [1, i, E]

            with torch.no_grad():
                pred_emb = lcm(prefix_embs)  # [1, E] or [E]

            if pred_emb.dim() == 1:
                pred_emb = pred_emb.unsqueeze(0)

            retrieved = retriever.retrieve(pred_emb)
            hyps.append(retrieved[0])
            refs.append(doc_sents[i])

    print(f"\nEvaluating {len(hyps)} predictions...")
    metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)

    print(f"\n{'='*50}")
    print("  BLT-LCM Evaluation Results (NN Retrieval)")
    print(f"{'='*50}")
    print(f"  Samples evaluated : {len(hyps)}")
    for k, v in metrics.items():
        if v == v:  # skip NaN
            print(f"  {k:12s}: {v:.2f}")
    print(f"{'='*50}")

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
