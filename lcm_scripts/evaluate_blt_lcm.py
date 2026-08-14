"""
Evaluate BLT-LCM model on the test split (last 20% of BhashaSetu dataset) for Machine Translation tasks.

Computes the average MSE loss on next sentence prediction and evaluates generated hypotheses against references using text metrics.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
import re
from blt_loader import BLTLoader
from base_lcm import BaseLCM
import os
from run_blt_patching import text_to_byte_tokens
from tqdm import tqdm

from device_utils import report_device
from checkpoint_utils import (
    ResumableJsonl,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    load_model_state,
)


def prepare_data(is_test=False, max_sent_per_doc=20):
    ds = load_dataset("ParamTh/BhashaSetu", split="train")
    total = len(ds)
    if is_test:
        test_start = int(total * 0.8)  # last 20%
        ds = ds.select(range(test_start, total))
    else:
        train_end = int(total * 0.8)  # first 80%, disjoint from test
        ds = ds.select(range(train_end))
    docs = []
    for row in tqdm(ds, desc="Loading data"):
        text = row.get("marathi", "")
        if text and len(text.strip()) > 0:
            sents = [s.strip() for s in re.split(r"[.।]", text) if s.strip()]
            if len(sents) >= 2:
                docs.append(sents[:max_sent_per_doc])
    return docs


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_seqs):
        self.data = [seq for seq in embeddings_seqs if len(seq) >= 2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def collate(batch):
    srcs, tgts = zip(*batch)
    max_len = max(s.shape[0] for s in srcs)
    emb_dim = srcs[0].shape[1]
    B = len(srcs)
    src_p = torch.zeros(B, max_len, emb_dim, device=srcs[0].device, dtype=srcs[0].dtype)
    tgt_p = torch.zeros(B, max_len, emb_dim, device=srcs[0].device, dtype=srcs[0].dtype)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_p[i, : s.shape[0]] = s
        tgt_p[i, : t.shape[0]] = t
    return src_p, tgt_p


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BLT-LCM on the BhashaSetu test split"
    )
    parser.add_argument(
        "--checkpoint", default="lcm_models/lcm_blt_best.pth", help="LCM checkpoint"
    )
    parser.add_argument(
        "--entropy_model", default="patching_scratch/entropy_model_marathi.pt"
    )
    parser.add_argument(
        "--train_embed_cache",
        default=None,
        help="Optional cache for the encoded retrieval corpus (train split).",
    )
    parser.add_argument(
        "--test_embed_cache",
        default=None,
        help="Optional cache for the encoded test-split document embeddings.",
    )
    parser.add_argument(
        "--progress_jsonl",
        default="outputs/evaluate_blt_lcm_progress.jsonl",
        help="Per-batch retrieval results, streamed as they are produced. A "
        "rerun replays this file and only evaluates the missing batches.",
    )
    add_resume_args(parser, training=False)
    args = parser.parse_args()

    device = report_device()
    fingerprint = config_fingerprint(args, extra={"stage": "evaluate_blt_lcm"})

    # Load BLT loader
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    # Prepare training data for closest sentence lookup
    print("Preparing training data for lookup...")
    train_docs = prepare_data(is_test=False)
    train_sentences = [s for doc in train_docs for s in doc]
    print(f"Loaded {len(train_sentences)} training sentences")

    # Encode training sentences
    print("Encoding training sentences...")

    def _encode_train():
        tokenized_train = [
            text_to_byte_tokens(sent)
            for sent in tqdm(train_sentences, desc="Tokenizing")
        ]
        return torch.stack(blt.encode_tokens_batch(tokenized_train))

    train_embs = cached_torch(
        args.train_embed_cache,
        _encode_train,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        label="retrieval-corpus embeddings",
    ).to(device)
    # Normalized once, up front: cosine similarity against a fixed index is a
    # matmul between unit vectors, and renormalizing the index per query (as
    # cosine_similarity did) re-read the whole thing every time.
    train_embs_normed = F.normalize(train_embs.float(), dim=1)

    # Prepare test data
    print("Preparing test data...")
    test_docs = prepare_data(is_test=True)
    print(f"Loaded {len(test_docs)} test documents")

    # Encode test data
    print("Encoding test data with BLT...")

    def _encode_test():
        flat_sents = []
        doc_indices = []
        for i, sents in enumerate(test_docs):
            for sent in sents:
                flat_sents.append(sent)
                doc_indices.append(i)
        tokenized_batch = [
            text_to_byte_tokens(sent)
            for sent in tqdm(flat_sents, desc="Tokenizing test")
        ]
        embed_list = blt.encode_tokens_batch(tokenized_batch)

        # Reconstruct per-document sequences
        seqs = [[] for _ in range(len(test_docs))]
        for emb, didx in zip(embed_list, doc_indices):
            seqs[didx].append(emb)

        # stack per-document tensors
        for i in range(len(seqs)):
            if len(seqs[i]) == 0:
                seqs[i] = torch.empty((0, blt.dim))
            else:
                seqs[i] = torch.stack(seqs[i], dim=0)
        return seqs

    embeddings_seqs = cached_torch(
        args.test_embed_cache,
        _encode_test,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=lambda s: bool(s),
        label="test document embeddings",
    )

    # Create dataset and dataloader. shuffle=False, so batch index i always
    # covers the same examples — which is what makes per-batch resume sound.
    dataset = EmbeddingDataset(embeddings_seqs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)

    model = BaseLCM(
        embed_dim=embeddings_seqs[0].shape[1], model_dim=2048, n_layers=12, n_heads=16
    )
    # Accepts both the resumable checkpoint payload and older bare state dicts.
    model.load_state_dict(load_model_state(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    writer = ResumableJsonl(
        args.progress_jsonl,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        key="batch",
        flush_every=20,
    )
    if writer.done:
        print(f"Resuming evaluation: {len(writer.done)} batches already scored")

    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if writer.is_done(batch_idx):
                continue
            src = src.to(device)
            tgt = tgt.to(device)
            seq_len = tgt.shape[1]
            if seq_len == 0:
                continue
            # One causal pass yields every next-concept prediction at once.
            preds = model.forward_all(src)  # [B, L, E]

            # Per-position MSE, computed in one reduction. The old loop called
            # mse() and .item() once per position, so every position cost a
            # device sync.
            per_pos = (preds - tgt).pow(2).mean(dim=(0, 2))  # [L]
            batch_loss = float(per_pos.sum())
            batch_n = seq_len

            # Nearest training sentence for every (position, batch) prediction,
            # as a single matmul. This used to be B x L separate
            # cosine_similarity calls against the whole index, each followed by
            # an argmax and an .item() sync -- 160 of them per batch at B=8,
            # L=20, and each one re-read the entire [N, E] index.
            #
            # Iteration order is position-major to match the previous loop, so
            # the hypothesis order in the output is unchanged.
            flat = F.normalize(preds.transpose(0, 1).reshape(-1, preds.shape[-1]), dim=1)
            best = (flat @ train_embs_normed.T).argmax(dim=1).cpu().tolist()
            batch_hyps = [train_sentences[i] for i in best]

            writer.append(
                {
                    "batch": batch_idx,
                    "loss_sum": batch_loss,
                    "n": batch_n,
                    "hyps": batch_hyps,
                }
            )
    writer.close()

    # Reassemble in batch order; `all_records` sorts by the resume key, so a run
    # stitched together from several attempts still yields the original order.
    rows = writer.all_records()
    hyps = [h for r in rows for h in r["hyps"]]
    total_loss = sum(r["loss_sum"] for r in rows)
    n = sum(r["n"] for r in rows)

    # Collect refs: for each test doc, the next sentences
    refs = []
    for doc in test_docs:
        refs.extend(doc[1:])  # Skip first, as no prediction for it
    refs = refs[: len(hyps)]  # Trim if necessary

    avg_loss = total_loss / n if n > 0 else float("inf")
    print(f"Test MSE Loss: {avg_loss:.4f}")

    # Save hyp and ref
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hyp.txt", "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")
    with open("outputs/ref.txt", "w", encoding="utf-8") as f:
        for r in refs:
            f.write(r + "\n")

    # Run eval_runner
    os.makedirs("results", exist_ok=True)
    os.system(
        "python lcm_scripts/eval_runner.py --hyp_file outputs/hyp.txt --ref_file outputs/ref.txt --out_csv results/mt_eval_results.csv --comet_model wmt22-comet-da"
    )

    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Test MSE Loss: {avg_loss:.4f}\n")
        f.write(f"Number of predictions: {n}\n")


if __name__ == "__main__":
    main()
