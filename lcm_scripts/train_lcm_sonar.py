"""Train and benchmark the SONAR-embedding + BaseLCM baseline on BhashaSetu.

The baseline encodes Marathi sentence documents with ``SonarLoader`` (XLM-R based
SONAR-like embeddings), trains BaseLCM for next-sentence embedding prediction,
and decodes predictions through nearest-neighbor retrieval. Metrics are BLEU,
chrF++ and TER at clean, 10% noisy and 20% noisy evaluation settings.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from base_lcm import BaseLCM
from checkpoint_utils import (
    ResumableLoader,
    ResumePoint,
    StageTracker,
    TrainingCheckpointer,
    add_resume_args,
    cached_torch,
    config_fingerprint,
)
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


class EmbeddingSequenceDataset(Dataset):
    def __init__(self, seqs: list[torch.Tensor]):
        self.seqs = [s for s in seqs if s.dim() == 2 and s.shape[0] >= 2]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq[:-1], seq[1:]


def collate(batch):
    srcs, tgts = zip(*batch)
    max_len = max(s.shape[0] for s in srcs)
    dim = srcs[0].shape[1]
    src = torch.zeros(
        len(batch), max_len, dim, device=srcs[0].device, dtype=srcs[0].dtype
    )
    tgt = torch.zeros(
        len(batch), max_len, dim, device=srcs[0].device, dtype=srcs[0].dtype
    )
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : s.shape[0]] = s
        tgt[i, : t.shape[0]] = t
    return src, tgt


def encode_docs(
    docs: list[list[str]], encoder: SonarLoader, batch_size: int
) -> list[torch.Tensor]:
    flat = [s for doc in docs for s in doc]
    encoded = []
    for i in tqdm(range(0, len(flat), batch_size), desc="SONAR encode"):
        encoded.extend(
            [e.cpu() for e in encoder.encode_sentences(flat[i : i + batch_size])]
        )
    seqs, offset = [], 0
    for doc in docs:
        seqs.append(torch.stack(encoded[offset : offset + len(doc)]))
        offset += len(doc)
    return seqs


def train_epoch(
    model,
    loader: ResumableLoader,
    optim,
    device,
    epoch: int,
    ckpt: TrainingCheckpointer,
    resume: ResumePoint,
    global_step: int,
) -> tuple[float, int]:
    """Run one epoch, checkpointing as it goes. Returns (avg_loss, global_step)."""
    model.train()
    total = 0.0
    steps = 0
    skip = resume.batches_to_skip(epoch)
    for batch_idx, (src, tgt) in tqdm(
        loader.epoch(epoch, skip=skip), desc="train", initial=skip, total=len(loader)
    ):
        src, tgt = src.to(device), tgt.to(device)
        optim.zero_grad()
        pred = model(src, tgt)
        # Mask loss over padded (zero) positions so padding doesn't bias
        # the model toward predicting the origin.
        mask = (tgt.abs().sum(dim=-1, keepdim=True) > 0).float()
        loss = ((pred - tgt).pow(2) * mask).sum() / mask.sum().clamp(min=1)
        loss.backward()
        optim.step()
        total += float(loss.item())
        steps += 1
        global_step += 1
        ckpt.maybe_save(
            model,
            optim,
            epoch=epoch,
            batch_in_epoch=batch_idx,
            global_step=global_step,
        )
    return total / max(1, steps), global_step


def evaluate(
    model, docs, encoder, retriever, args, noise: float, device
) -> dict[str, float]:
    hyps, refs = [], []
    model.eval()
    for doc_idx, doc in enumerate(tqdm(docs, desc=f"eval noise={noise:.2f}")):
        if len(doc) < args.min_prefix + 1:
            continue
        noisy_doc = (
            [
                add_character_noise(s, noise, seed=doc_idx * 1000 + i)
                for i, s in enumerate(doc)
            ]
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
        "BLEU": compute_bleu(hyps, refs),
        "chrF++": compute_chrf(hyps, refs),
        "TER": compute_ter(hyps, refs),
    }


def main():
    p = argparse.ArgumentParser(description="Train SONAR embedding + LCM baseline")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--eval_docs", type=int, default=100)
    p.add_argument("--max_sent_per_doc", type=int, default=20)
    p.add_argument("--text_col", default="marathi")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--encode_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model_dim", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=16)
    p.add_argument("--min_prefix", type=int, default=2)
    p.add_argument("--out_dir", default="runs/lcm_sonar")
    p.add_argument(
        "--log_dir",
        default=None,
        help="Alias for --out_dir used by Slurm/W&B launchers",
    )
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="BLT-LCM")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument(
        "--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS)
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--embed_cache",
        default=None,
        help="Optional path for the cached SONAR encodings of the train split. "
        "Set it so a resumed run skips re-encoding the corpus.",
    )
    add_resume_args(p, default_interval_steps=200)
    args = p.parse_args()

    if args.log_dir and args.out_dir == "runs/lcm_sonar":
        args.out_dir = args.log_dir
    if args.log_dir is None:
        args.log_dir = args.out_dir
    if args.wandb_name is None:
        args.wandb_name = os.path.basename(args.out_dir.rstrip(os.sep))

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            dir=args.log_dir,
            config=vars(args),
        )

    os.makedirs(args.out_dir, exist_ok=True)
    docs = load_bhashasetu_documents(
        args.dataset, args.split, args.fraction, args.max_sent_per_doc, args.text_col
    )
    train_docs, eval_docs = split_train_eval_documents(docs, args.eval_docs)

    device = torch.device(args.device)
    fingerprint = config_fingerprint(args)
    encoder = SonarLoader(device=str(device))
    # SONAR encoding is frozen and config-determined, so a resumed run reloads it.
    seqs = cached_torch(
        args.embed_cache,
        lambda: encode_docs(train_docs, encoder, args.encode_batch_size),
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=lambda s: bool(s),
        label="SONAR document encodings",
    )
    flat_sents = [s for doc in train_docs for s in doc]
    flat_embs = torch.cat([s for s in seqs if s.shape[0] > 0], dim=0)
    retriever = EmbeddingRetriever(flat_sents, flat_embs)

    dataset = EmbeddingSequenceDataset(seqs)
    loader = ResumableLoader(
        dataset,
        batch_size=args.batch_size,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate,
    )
    if not seqs:
        raise RuntimeError(
            "No valid documents found after encoding. Check dataset filters."
        )
    embed_dim = seqs[0].shape[1]
    model = BaseLCM(
        embed_dim=embed_dim,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt = TrainingCheckpointer(
        args.out_dir,
        prefix=f"lcm_sonar_fraction{args.fraction}",
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(ckpt.load(args.resume, map_location=device), model, optim)
    global_step = resume.global_step
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}"
        )

    for epoch in range(resume.start_epoch, args.epochs):
        loss, global_step = train_epoch(
            model, loader, optim, device, epoch, ckpt, resume, global_step
        )
        print(f"epoch={epoch + 1} train_loss={loss:.4f}")
        if wandb_run:
            wandb_run.log({"train/loss": loss, "epoch": epoch + 1})
        ckpt.save_epoch(model, optim, epoch=epoch, global_step=global_step)

    # Each noise level is a full retrieval decode of the eval set; memoize the
    # finished ones so an interrupted evaluation picks up where it stopped.
    stages = StageTracker(
        os.path.join(args.out_dir, f"eval_state_fraction{args.fraction}.json"),
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
            **metrics,
        }
        rows.append(row)
        print(row)
        if wandb_run:
            wandb_run.log({f"eval/{k}_noise_{noise}": v for k, v in metrics.items()})
    out_csv = os.path.join(args.out_dir, f"metrics_fraction{args.fraction}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "fraction", "noise", "BLEU", "chrF++", "TER"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
