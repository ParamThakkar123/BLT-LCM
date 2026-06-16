"""Train/evaluate a BPE-embedding + BaseLCM baseline on BhashaSetu.

This baseline mirrors the SONAR-LCM experiment, but replaces SONAR sentence
embeddings with sentence vectors built from a learned SentencePiece BPE embedding
lookup. Documents are modeled as sequences of sentence embeddings; BaseLCM learns
next-sentence embedding prediction and evaluation decodes with nearest-neighbor
retrieval over the training sentence index.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from base_lcm import BaseLCM
from bhashasetu_utils import DEFAULT_DATASET, DEFAULT_NOISE_LEVELS, add_character_noise, load_bhashasetu_documents
from embedding_retriever import EmbeddingRetriever
from eval_metrics import compute_bleu, compute_chrf, compute_ter

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover - dependency checked at runtime
    spm = None

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class BPESentenceEncoder(nn.Module):
    """Mean-pool learned BPE token embeddings into sentence embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, pad_id: int = PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(ids)
        mask_f = mask.unsqueeze(-1).to(emb.dtype)
        pooled = (emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)

    def encode_sentences(self, sentences: list[str], sp, max_len: int, device) -> torch.Tensor:
        ids, mask = encode_batch(sentences, sp, max_len)
        with torch.no_grad():
            return self(ids.to(device), mask.to(device)).detach().cpu()


class BPEDocumentDataset(Dataset):
    def __init__(self, docs: list[list[str]], sp, max_len: int):
        self.docs = []
        for doc in docs:
            if len(doc) < 2:
                continue
            encoded = []
            for sent in doc:
                ids = [BOS_ID] + sp.encode(sent, out_type=int)[: max_len - 2] + [EOS_ID]
                encoded.append(torch.tensor(ids, dtype=torch.long))
            self.docs.append(encoded)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        return doc[:-1], doc[1:]


def collate(batch):
    src_docs, tgt_docs = zip(*batch)
    max_sents = max(len(doc) for doc in src_docs)
    max_tokens = max(tok.numel() for doc in src_docs + tgt_docs for tok in doc)
    src = torch.full((len(batch), max_sents, max_tokens), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_sents, max_tokens), PAD_ID, dtype=torch.long)
    src_mask = torch.zeros((len(batch), max_sents, max_tokens), dtype=torch.bool)
    tgt_mask = torch.zeros((len(batch), max_sents, max_tokens), dtype=torch.bool)
    for i, (src_doc, tgt_doc) in enumerate(zip(src_docs, tgt_docs)):
        for j, sent in enumerate(src_doc):
            src[i, j, : sent.numel()] = sent
            src_mask[i, j, : sent.numel()] = True
        for j, sent in enumerate(tgt_doc):
            tgt[i, j, : sent.numel()] = sent
            tgt_mask[i, j, : sent.numel()] = True
    return src, src_mask, tgt, tgt_mask


def encode_doc_tensor(encoder: BPESentenceEncoder, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, tok_len = ids.shape
    embs = encoder(ids.reshape(bsz * seq_len, tok_len), mask.reshape(bsz * seq_len, tok_len))
    return embs.reshape(bsz, seq_len, -1)



def encode_batch(sentences: list[str], sp, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = []
    for sent in sentences:
        ids = [BOS_ID] + sp.encode(sent, out_type=int)[: max_len - 2] + [EOS_ID]
        encoded.append(ids)
    width = max(len(x) for x in encoded)
    ids = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(encoded), width), dtype=torch.bool)
    for i, row in enumerate(encoded):
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, : len(row)] = True
    return ids, mask


def train_sentencepiece(docs: list[list[str]], out_dir: str, vocab_size: int) -> str:
    if spm is None:
        raise RuntimeError("sentencepiece is required. Install project dependencies first.")
    os.makedirs(out_dir, exist_ok=True)
    corpus = os.path.join(out_dir, "bpe_lcm_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for doc in docs:
            for sent in doc:
                f.write(sent.replace("\n", " ") + "\n")
    prefix = os.path.join(out_dir, "bhashasetu_bpe_lcm")
    spm.SentencePieceTrainer.train(
        input=corpus,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        character_coverage=1.0,
    )
    return prefix + ".model"


def train_epoch(model, encoder, loader, optim, device):
    mse = nn.MSELoss()
    model.train()
    encoder.train()
    total = 0.0
    steps = 0
    for src_ids, src_mask, tgt_ids, tgt_mask in tqdm(loader, desc="train"):
        src_ids, src_mask = src_ids.to(device), src_mask.to(device)
        tgt_ids, tgt_mask = tgt_ids.to(device), tgt_mask.to(device)
        optim.zero_grad()
        src = encode_doc_tensor(encoder, src_ids, src_mask)
        tgt = encode_doc_tensor(encoder, tgt_ids, tgt_mask)
        pred = model(src, tgt)
        loss = mse(pred, tgt)
        loss.backward()
        optim.step()
        total += float(loss.item())
        steps += 1
    return total / max(1, steps)


def build_retriever(train_docs, sp, encoder, args, device) -> EmbeddingRetriever:
    flat_sents = [sent for doc in train_docs for sent in doc]
    chunks = []
    for i in tqdm(range(0, len(flat_sents), args.encode_batch_size), desc="BPE encode retriever"):
        chunks.append(encoder.encode_sentences(flat_sents[i : i + args.encode_batch_size], sp, args.max_len, device))
    return EmbeddingRetriever(flat_sents, torch.cat(chunks, dim=0))


def evaluate(model, sp, encoder, docs, retriever, args, noise: float, device) -> dict[str, float | int]:
    hyps, refs = [], []
    model.eval()
    for doc_idx, doc in enumerate(tqdm(docs, desc=f"eval noise={noise:.2f}")):
        if len(doc) < args.min_prefix + 1:
            continue
        noisy_doc = [add_character_noise(s, noise, seed=doc_idx * 1000 + i) for i, s in enumerate(doc)] if noise else doc
        embs = encoder.encode_sentences(noisy_doc, sp, args.max_len, device)
        for i in range(args.min_prefix, len(doc)):
            with torch.no_grad():
                pred = model(embs[:i].unsqueeze(0).to(device))
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            hyps.append(retriever.retrieve(pred)[0])
            refs.append(doc[i])
    return {"num_predictions": len(hyps), "BLEU": compute_bleu(hyps, refs), "chrF++": compute_chrf(hyps, refs), "TER": compute_ter(hyps, refs)}


def maybe_init_wandb(args):
    if not args.wandb:
        return None
    import wandb

    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, dir=args.log_dir, config=vars(args))


def main():
    p = argparse.ArgumentParser(description="Train BPE embedding + LCM baseline")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--num_docs", type=int, default=500)
    p.add_argument("--eval_docs", type=int, default=100)
    p.add_argument("--max_sent_per_doc", type=int, default=20)
    p.add_argument("--text_col", default="marathi")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--min_prefix", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_dir", default="runs/lcm_bpe")
    p.add_argument("--log_dir", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="BLT-LCM")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.log_dir and args.out_dir == "runs/lcm_bpe":
        args.out_dir = args.log_dir
    if args.log_dir is None:
        args.log_dir = args.out_dir
    if args.wandb_name is None:
        args.wandb_name = Path(args.out_dir).name

    os.makedirs(args.out_dir, exist_ok=True)
    run = maybe_init_wandb(args)
    device = torch.device(args.device)

    docs = load_bhashasetu_documents(args.dataset, args.split, args.fraction, args.num_docs + args.eval_docs, args.max_sent_per_doc, args.text_col)
    train_docs = docs[: args.num_docs]
    eval_docs = docs[args.num_docs : args.num_docs + args.eval_docs] or train_docs[: min(args.eval_docs, len(train_docs))]
    if not train_docs or not eval_docs:
        raise RuntimeError("No BhashaSetu documents available for BPE-LCM training/evaluation")

    sp_model = train_sentencepiece(train_docs, args.out_dir, args.vocab_size)
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    encoder = BPESentenceEncoder(sp.get_piece_size(), args.embed_dim).to(device)
    dataset = BPEDocumentDataset(train_docs, sp, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = BaseLCM(embed_dim=args.embed_dim, model_dim=args.model_dim, n_layers=args.n_layers, n_heads=args.n_heads).to(device)
    optim = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        loss = train_epoch(model, encoder, loader, optim, device)
        print(f"epoch={epoch + 1} train_loss={loss:.4f}")
        if run:
            run.log({"train/loss": loss, "epoch": epoch + 1})
        torch.save({"lcm": model.state_dict(), "encoder": encoder.state_dict(), "args": vars(args)}, os.path.join(args.out_dir, f"lcm_bpe_fraction{args.fraction}_epoch{epoch + 1}.pth"))

    retriever = build_retriever(train_docs, sp, encoder, args, device)
    rows = []
    for noise in args.noise_levels:
        metrics = evaluate(model, sp, encoder, eval_docs, retriever, args, noise, device)
        row = {"model": "bpe_lcm", "fraction": args.fraction, "noise": noise, **metrics}
        rows.append(row)
        print(row)
        if run:
            run.log({f"eval/{k}_noise_{noise}": v for k, v in metrics.items() if isinstance(v, (int, float))})

    out_csv = os.path.join(args.out_dir, f"metrics_fraction{args.fraction}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fraction", "noise", "num_predictions", "BLEU", "chrF++", "TER"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
