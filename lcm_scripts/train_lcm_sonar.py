"""
Train LCM using SONAR-like embeddings (SonarLite implemented in this repo).

Usage:
  python lcm_scripts/train_lcm_sonar.py --num_docs 1000 --epochs 3
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset

from sonar_module import SonarLite
from base_lcm import BaseLCM
from eval_metrics import compute_all
from experiment_config import setup_logging
import os


def prepare_data(num_docs=1000, max_sent_per_doc=20):
    # load Marathi sentences from ParamTh/BhashaSetu
    from datasets import load_dataset

    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    docs = []
    cur = 0
    for row in ds:
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            # simple sentence split by punctuation; better: SaT Capped in full pipeline
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
        # seq is tensor [num_sent, embed_dim]
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
    parser.add_argument("--wandb_entity", type=str, default=None)
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
            project=args.wandb_project or "blt-lcm",
            name=args.wandb_name,
            entity=args.wandb_entity,
            config={"project": "blt-lcm"},
        )
    print("Preparing data...")
    docs = prepare_data(num_docs=args.num_docs)

    print("Building SONAR-lite encoder...")
    sonar = SonarLite(device=device)

    print("Encoding sentences (this may take a while)...")
    embeddings_seqs = []
    for sents in tqdm(docs):
        emb = sonar.encode_sentences(sents)
        embeddings_seqs.append(emb.cpu())

    dataset = EmbeddingDataset(embeddings_seqs)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )

    print("Building LCM model...")
    model = BaseLCM(
        embed_dim=1024, model_dim=2048, n_layers=12, n_heads=16, max_seq_len=256
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
        global_step = (
            epoch * 1000000
        )  # large base to keep epoch steps distinct if desired
        for src, tgt in tqdm(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            optim.zero_grad()
            # predict for each position i: using prefix up to i predict tgt at i
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
            # per-batch logging
            if writer is not None:
                # increment step
                global_step += 1
                writer.add_scalar("train/step_loss", loss.item(), global_step)
                # learning rate
                try:
                    lr = optim.param_groups[0]["lr"]
                    writer.add_scalar("train/lr", lr, global_step)
                except Exception:
                    pass
                # gradient norm
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
        torch.save(model.state_dict(), f"lcm_models/lcm_sonar_epoch{epoch + 1}.pth")

        # Log epoch-level training loss
        if writer is not None:
            writer.add_scalar("train/loss", total / n if n > 0 else 0.0, epoch + 1)
            # parameter histograms
            try:
                for name, param in model.named_parameters():
                    writer.add_histogram(
                        f"params/{name}", param.clone().cpu().data.numpy(), epoch + 1
                    )
            except Exception:
                pass
        if wandb_module is not None:
            try:
                wandb_module.log(
                    {"train/loss": total / n if n > 0 else 0.0}, step=epoch + 1
                )
            except Exception:
                pass

        # Optional evaluation using supplied hypothesis/reference files
        if args.eval_hyp and args.eval_ref:
            with open(args.eval_hyp, encoding="utf-8") as f:
                hyps = [l.strip() for l in f if l.strip()]
            with open(args.eval_ref, encoding="utf-8") as f:
                refs = [l.strip() for l in f if l.strip()]
            metrics = compute_all(hyps, refs, comet_model_name=args.comet_model)
            for k, v in metrics.items():
                if writer is not None:
                    writer.add_scalar(f"eval/{k}", v, epoch + 1)
            # also write a CSV summary in the log dir
            try:
                out_csv = os.path.join(args.log_dir, "eval_summary.csv")
                header = sorted(metrics.keys())
                header = ["epoch"] + header
                if not os.path.exists(out_csv):
                    with open(out_csv, "w", encoding="utf-8") as f:
                        f.write(",".join(header) + "\n")
                with open(out_csv, "a", encoding="utf-8") as f:
                    vals = [str(metrics[k]) for k in sorted(metrics.keys())]
                    f.write(str(epoch + 1) + "," + ",".join(vals) + "\n")
            except Exception:
                pass
            # checkpoint best by preferred metric (chrF++ then BLEU then METEOR)
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
                # higher is better
                best_path = f"lcm_models/lcm_sonar_best.pth"
                prev_best = getattr(main, "_best_score", None)
                if prev_best is None or monitor_score > prev_best:
                    torch.save(model.state_dict(), best_path)
                    setattr(main, "_best_score", monitor_score)
                    # upload model artifact to W&B
                    try:
                        if wandb_module is not None:
                            art = wandb_module.Artifact("lcm_sonar_model", type="model")
                            art.add_file(best_path)
                            wandb_module.run.log_artifact(art)
                    except Exception:
                        pass
        # Log example reconstructions (first doc first sentence)
        try:
            if (
                writer is not None
                and len(embeddings_seqs) > 0
                and len(embeddings_seqs[0]) > 0
            ):
                sample_emb = embeddings_seqs[0][0].unsqueeze(0).to(device)
                recon = sonar.decode_embeddings(sample_emb.cpu(), max_len=128)[0]
                writer.add_text("examples/reconstruction", recon, epoch + 1)
                if wandb_module is not None:
                    try:
                        wandb_module.log(
                            {"examples/reconstruction": recon}, step=epoch + 1
                        )
                        # also save a small text artifact
                        art = wandb_module.Artifact("recon_text", type="reconstruction")
                        tmp = os.path.join(
                            args.log_dir or ".", f"recon_epoch{epoch + 1}.txt"
                        )
                        with open(tmp, "w", encoding="utf-8") as f:
                            f.write(recon)
                        art.add_file(tmp)
                        wandb_module.run.log_artifact(art)
                    except Exception:
                        pass
        except Exception:
            pass
        else:
            # no external eval: checkpoint by negative training loss (higher better)
            monitor_score = -(total / n if n > 0 else float("inf"))
            best_path = f"lcm_models/lcm_sonar_best.pth"
            prev_best = getattr(main, "_best_score", None)
            if prev_best is None or monitor_score > prev_best:
                torch.save(model.state_dict(), best_path)
                setattr(main, "_best_score", monitor_score)


if __name__ == "__main__":
    main()
