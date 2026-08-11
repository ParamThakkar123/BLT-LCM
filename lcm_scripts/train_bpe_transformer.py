"""Train/evaluate a BPE + Transformer translation baseline on BhashaSetu.

The script trains a SentencePiece BPE tokenizer, a compact PyTorch
encoder-decoder Transformer, then reports BLEU, chrF++ and TER for the requested
BhashaSetu fraction. It is designed for reproducible subset studies (25%, 50%,
80%) and noisy-input benchmarking (0%, 10%, 20%).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Sequence

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from bhashasetu_utils import (
    DEFAULT_DATASET,
    DEFAULT_NOISE_LEVELS,
    ParallelExample,
    add_character_noise,
    load_bhashasetu_pairs,
)
from eval_metrics import compute_bleu, compute_chrf, compute_ter
from device_utils import report_device
from checkpoint_utils import (
    ResumableLoader,
    StageTracker,
    TrainingCheckpointer,
    add_resume_args,
    config_fingerprint,
    seed_everything,
)

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover - dependency checked at runtime
    spm = None


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


class TranslationDataset(Dataset):
    def __init__(
        self, pairs: Sequence[ParallelExample], sp, max_len: int, noise: float = 0.0
    ):
        self.pairs = list(pairs)
        self.sp = sp
        self.max_len = max_len
        self.noise = noise

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode(self, text: str) -> list[int]:
        ids = (
            [BOS_ID] + self.sp.encode(text, out_type=int)[: self.max_len - 2] + [EOS_ID]
        )
        return ids

    def __getitem__(self, idx: int):
        ex = self.pairs[idx]
        src = (
            add_character_noise(ex.source, self.noise, seed=idx)
            if self.noise
            else ex.source
        )
        return torch.tensor(self._encode(src)), torch.tensor(self._encode(ex.target))


def collate(batch):
    srcs, tgts = zip(*batch)
    max_src = max(x.numel() for x in srcs)
    max_tgt = max(x.numel() for x in tgts)
    src = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : s.numel()] = s
        tgt[i, : t.numel()] = t
    return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class BPETransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD_ID)
        tgt_pad = tgt_in.eq(PAD_ID)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_in.size(1), device=tgt_in.device
        )
        src_e = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_e = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        y = self.transformer(
            src_e,
            tgt_e,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(y)


def train_sentencepiece(
    pairs: Sequence[ParallelExample],
    out_dir: str,
    vocab_size: int,
    resume: bool = True,
) -> str:
    if spm is None:
        raise RuntimeError(
            "sentencepiece is required. Install project dependencies first."
        )
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "bhashasetu_bpe") + ".model"
    if resume and os.path.exists(model_path):
        # The saved model checkpoints were trained against this vocabulary;
        # retraining it on resume would shift every token id underneath them.
        print(f"[resume] reusing existing SentencePiece model {model_path}")
        return model_path
    corpus_path = os.path.join(out_dir, "bpe_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for ex in pairs:
            f.write(ex.source.replace("\n", " ") + "\n")
            f.write(ex.target.replace("\n", " ") + "\n")
    prefix = os.path.join(out_dir, "bhashasetu_bpe")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
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


def greedy_decode(model, sp, src, max_len: int, device) -> list[str]:
    model.eval()
    hyps: list[str] = []
    with torch.no_grad():
        for row in src:
            src_one = row.unsqueeze(0).to(device)
            ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
            for _ in range(max_len - 1):
                logits = model(src_one, ys)
                next_id = int(logits[:, -1].argmax(-1).item())
                ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=1)
                if next_id == EOS_ID:
                    break
            ids = [
                i for i in ys.squeeze(0).tolist() if i not in (BOS_ID, EOS_ID, PAD_ID)
            ]
            hyps.append(sp.decode(ids))
    return hyps


def evaluate(model, sp, pairs, args, noise: float, device) -> dict[str, float]:
    ds = TranslationDataset(pairs, sp, args.max_len, noise=noise)
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate
    )
    hyps, refs = [], []
    for src, tgt in tqdm(loader, desc=f"eval noise={noise:.2f}"):
        hyps.extend(greedy_decode(model, sp, src, args.max_len, device))
        refs.extend(
            [
                sp.decode(
                    [i for i in row.tolist() if i not in (BOS_ID, EOS_ID, PAD_ID)]
                )
                for row in tgt
            ]
        )
    return {
        "BLEU": compute_bleu(hyps, refs),
        "chrF++": compute_chrf(hyps, refs),
        "TER": compute_ter(hyps, refs),
    }


def main():
    p = argparse.ArgumentParser(description="Train BPE + Transformer on BhashaSetu")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--eval_examples", type=int, default=2000)
    p.add_argument("--src_col", default=None)
    p.add_argument("--tgt_col", default=None)
    p.add_argument("--out_dir", default="runs/bpe_transformer")
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dim_ff", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS)
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    add_resume_args(p, default_interval_steps=200)
    args = p.parse_args()

    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading BhashaSetu fraction={args.fraction}")

    wandb_module = None
    if args.wandb:
        from experiment_config import setup_wandb

        wandb_module = setup_wandb(
            args.out_dir,
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config={},
        )
    pairs = load_bhashasetu_pairs(
        args.dataset,
        args.split,
        args.fraction,
        args.max_examples,
        args.src_col,
        args.tgt_col,
    )
    if len(pairs) < 2:
        raise RuntimeError(
            "Need at least two parallel examples. Check src/tgt columns."
        )
    split = max(1, int(len(pairs) * 0.95))
    train_pairs, eval_pairs = pairs[:split], pairs[split : split + args.eval_examples]
    if not eval_pairs:
        eval_pairs = train_pairs[: min(args.eval_examples, len(train_pairs))]

    sp_model = train_sentencepiece(
        train_pairs, args.out_dir, args.vocab_size, resume=args.resume != "never"
    )
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    train_ds = TranslationDataset(train_pairs, sp, args.max_len)
    loader = ResumableLoader(
        train_ds,
        batch_size=args.batch_size,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate,
    )

    device = report_device(args.device)
    model = BPETransformer(
        sp.get_piece_size(),
        args.d_model,
        args.nhead,
        args.num_layers,
        args.dim_ff,
        args.dropout,
    ).to(device)
    if wandb_module is not None:
        try:
            wandb_module.watch(model, log="all", log_freq=100)
        except Exception:
            pass
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    ckpt = TrainingCheckpointer(
        args.out_dir,
        prefix=f"bpe_transformer_fraction{args.fraction}",
        suffix=".pt",
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(ckpt.load(args.resume, map_location=device), model, opt)
    global_step = resume.global_step
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}"
        )

    for epoch in range(resume.start_epoch, args.epochs):
        model.train()
        total, steps = 0.0, 0
        skip = resume.batches_to_skip(epoch)
        for batch_idx, (src, tgt) in tqdm(
            loader.epoch(epoch, skip=skip),
            desc=f"epoch {epoch + 1}",
            initial=skip,
            total=len(loader),
        ):
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad()
            logits = model(src, tgt[:, :-1])
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
            global_step += 1
            ckpt.maybe_save(
                model,
                opt,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
            )
        avg_loss = total / max(1, steps)
        print(f"epoch={epoch + 1} train_loss={avg_loss:.4f}")
        if wandb_module is not None:
            try:
                wandb_module.log({"train/loss": avg_loss}, step=epoch + 1)
            except Exception:
                pass
        ckpt.save_epoch(model, opt, epoch=epoch, global_step=global_step)

    # Greedy decoding the eval set is the slowest part of this script and runs
    # once per noise level, so completed levels are memoized.
    stages = StageTracker(
        os.path.join(args.out_dir, f"eval_state_fraction{args.fraction}.json"),
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )
    rows = []
    for noise in args.noise_levels:
        metrics = stages.run(
            f"noise={noise}",
            lambda noise=noise: evaluate(model, sp, eval_pairs, args, noise, device),
        )
        row = {
            "model": "bpe_transformer",
            "fraction": args.fraction,
            "noise": noise,
            **metrics,
        }
        rows.append(row)
        print(row)
        if wandb_module is not None:
            try:
                wandb_module.log(
                    {
                        f"eval/noise_{noise}/BLEU": metrics["BLEU"],
                        f"eval/noise_{noise}/chrF++": metrics["chrF++"],
                        f"eval/noise_{noise}/TER": metrics["TER"],
                    }
                )
            except Exception:
                pass

    out_csv = os.path.join(args.out_dir, f"metrics_fraction{args.fraction}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "fraction", "noise", "BLEU", "chrF++", "TER"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")

    if wandb_module is not None:
        try:
            wandb_module.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
