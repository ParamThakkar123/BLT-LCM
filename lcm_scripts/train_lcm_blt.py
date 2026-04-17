"""
Train BLT-LCM: use BLT-derived embeddings (from BLTLoader) to train same LCM architecture.

Usage:
  python lcm_scripts/train_lcm_blt.py --entropy_model ../patching_scratch/entropy_model_marathi.pt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from blt_loader import BLTLoader
from base_lcm import BaseLCM
from eval_metrics import compute_all
from experiment_config import setup_logging
import os


def prepare_data(num_docs=500, max_sent_per_doc=20):
    from datasets import load_dataset

    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    docs = []
    cur = 0
    for row in ds:
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            sents = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
            if len(sents) > 1:
                docs.append(sents[:max_sent_per_doc])
                cur += 1
        if cur >= num_docs:
            break
    return docs


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_seqs):
        self.data = embeddings_seqs

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
    src_p = torch.zeros(B, max_len, emb_dim)
    tgt_p = torch.zeros(B, max_len, emb_dim)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_p[i, : s.shape[0]] = s
        tgt_p[i, : t.shape[0]] = t
    return src_p, tgt_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_model", type=str, required=True)
    parser.add_argument("--num_docs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--eval_hyp",
        type=str,
        default=None,
        help="Optional hypothesis file to evaluate after each epoch",
    )
    parser.add_argument(
        "--eval_ref",
        type=str,
        default=None,
        help="Optional reference file to evaluate after each epoch",
    )
    parser.add_argument(
        "--comet_model",
        type=str,
        default=None,
        help="Optional COMET model name or checkpoint path",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    writer = None
    if args.log_dir:
        writer = setup_logging(args.log_dir)
    wandb_module = None
    if args.wandb:
        from experiment_config import setup_wandb

        wandb_module = setup_wandb(
            args.log_dir or ".",
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config={},
        )
    print("Preparing data list...")
    docs = prepare_data(args.num_docs)

    print("Loading BLT loader...")
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    print("Encoding with BLT (this may take a while)...")
    embeddings_seqs = []
    for sents in tqdm(docs):
        emb = blt.encode_sentences(sents)
        embeddings_seqs.append(emb.cpu())

    dataset = EmbeddingDataset(embeddings_seqs)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    model = BaseLCM(
        embed_dim=embeddings_seqs[0].shape[1], model_dim=2048, n_layers=12, n_heads=16
    ).to(device)
    if wandb_module is not None:
        try:
            wandb_module.watch(model, log="all", log_freq=100)
        except Exception:
            pass
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    mse = torch.nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        start = time.time()
        global_step = epoch * 1000000
        for src, tgt in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            optim.zero_grad()
            losses = []
            seq_len = tgt.shape[1]
            for i in range(seq_len):
                prefix = src[:, : i + 1] if i + 1 <= src.shape[1] else src
                pred = model(prefix, tgt[:, i : i + 1])
                losses.append(mse(pred, tgt[:, i]))
            loss = torch.stack(losses).mean()
            loss.backward()
            optim.step()
            total += loss.item()
            n += 1
            if writer is not None:
                global_step += 1
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                try:
                    lr = optim.param_groups[0]["lr"]
                    writer.add_scalar("train/lr", lr, global_step)
                except Exception:
                    pass
                try:
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm += param_norm * param_norm
                    total_norm = total_norm**0.5
                    writer.add_scalar("train/grad_norm", total_norm, global_step)
                except Exception:
                    pass

        elapsed = time.time() - start
        print(f"Epoch {epoch + 1} avg loss: {total / n:.4f} time: {elapsed:.1f}s")
        torch.save(model.state_dict(), f"lcm_models/lcm_blt_epoch{epoch + 1}.pth")

        if writer is not None:
            writer.add_scalar("train/loss", total / n if n > 0 else 0.0, epoch + 1)
        if wandb_module is not None:
            try:
                wandb_module.log(
                    {"train/loss": total / n if n > 0 else 0.0}, step=epoch + 1
                )
            except Exception:
                pass

        if args.eval_hyp and args.eval_ref and writer is not None:
            with open(args.eval_hyp, encoding="utf-8") as f:
                hyps = [l.strip() for l in f if l.strip()]
            with open(args.eval_ref, encoding="utf-8") as f:
                refs = [l.strip() for l in f if l.strip()]
            metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)
            for k, v in metrics.items():
                writer.add_scalar(f"eval/{k}", v, epoch + 1)
            # checkpoint best by preferred metric
            monitor_score = None
            for cand in ["chrF++", "BLEU", "METEOR"]:
                val = metrics.get(cand, float("nan"))
                try:
                    if not (val is None) and (val == val):
                        monitor_score = float(val)
                        break
                except Exception:
                    continue
            if monitor_score is not None:
                best_path = f"lcm_models/lcm_blt_best.pth"
                prev_best = getattr(main, "_best_score", None)
                if prev_best is None or monitor_score > prev_best:
                    torch.save(model.state_dict(), best_path)
                    setattr(main, "_best_score", monitor_score)
        else:
            monitor_score = -(total / n if n > 0 else float("inf"))
            best_path = f"lcm_models/lcm_blt_best.pth"
            prev_best = getattr(main, "_best_score", None)
            if prev_best is None or monitor_score > prev_best:
                torch.save(model.state_dict(), best_path)
    # close writers
    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass
    if wandb_module is not None:
        try:
            wandb_module.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
