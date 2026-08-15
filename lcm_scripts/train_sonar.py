"""
Pretrain SonarLite auto-encoder on Marathi sentences (auto-encoding objective).

Saves checkpoint to lcm_models/sonar.pth

Usage:
  python lcm_scripts/train_sonar.py --num_samples 2000 --epochs 3
"""

import os
from dotenv import load_dotenv
load_dotenv()

import argparse
import time
import torch
from tqdm import tqdm

from datasets import load_dataset

from sonar_module import SonarLite, text_to_byte_tokens
from experiment_config import setup_logging
from device_utils import report_device
from plot_utils import (
    TrainingHistory,
    add_plot_args,
    plot_formats,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from train_control import EpochBudget, add_epoch_control_args
from checkpoint_utils import (
    ResumableLoader,
    TrainingCheckpointer,
    add_resume_args,
    config_fingerprint,
    seed_everything,
)


def stream_sentences(num_sentences=20000):
    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    cnt = 0
    for row in ds:
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            # simple split
            for s in text.replace("\n", " ").split("."):
                s = s.strip()
                if s:
                    yield s
                    cnt += 1
                    if cnt >= num_sentences:
                        return


class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, max_len=256):
        self.sentences = sentences
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        # support either plain sentence or (src, tgt) tuple
        if isinstance(s, tuple) or isinstance(s, list):
            src, tgt = s
            src_toks = text_to_byte_tokens(src)[: self.max_len]
            tgt_toks = text_to_byte_tokens(tgt)[: self.max_len]
            return torch.tensor(src_toks, dtype=torch.long), torch.tensor(
                tgt_toks, dtype=torch.long
            )
        else:
            toks = text_to_byte_tokens(s)[: self.max_len]
            return torch.tensor(toks, dtype=torch.long)


def collate_fn(batch):
    # batch elements may be (src) or (src, tgt)
    first = batch[0]
    if isinstance(first, tuple) or isinstance(first, list):
        srcs = [b[0] for b in batch]
        tgts = [b[1] for b in batch]
        src_lens = [s.numel() for s in srcs]
        tgt_lens = [t.numel() for t in tgts]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)
        B = len(batch)
        src_out = torch.full((B, max_src), 0, dtype=torch.long)
        src_mask = torch.zeros((B, max_src), dtype=torch.bool)
        tgt_out = torch.full((B, max_tgt), 0, dtype=torch.long)
        tgt_mask = torch.zeros((B, max_tgt), dtype=torch.bool)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src_out[i, : s.numel()] = s
            src_mask[i, : s.numel()] = 1
            tgt_out[i, : t.numel()] = t
            tgt_mask[i, : t.numel()] = 1
        return (src_out, src_mask, tgt_out, tgt_mask)
    else:
        lens = [b.numel() for b in batch]
        maxl = max(lens)
        B = len(batch)
        out = torch.full((B, maxl), 0, dtype=torch.long)
        mask = torch.zeros((B, maxl), dtype=torch.bool)
        for i, b in enumerate(batch):
            out[i, : b.numel()] = b
            mask[i, : b.numel()] = 1
        return out, mask


def train(args):
    device = report_device(args.device)
    os.makedirs(args.model_dir, exist_ok=True)
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)

    writer = None
    if hasattr(args, "log_dir") and args.log_dir:
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

    print("Loading sentences...")
    # support optional parallel dataset for translation objective
    if args.parallel_dataset:
        ds_iter = load_dataset(args.parallel_dataset, split="train", streaming=True)
        pairs = []
        cnt = 0
        for row in ds_iter:
            src = row.get(args.src_col, "")
            tgt = row.get(args.tgt_col, "")
            if src and tgt and len(src.strip()) > 2 and len(tgt.strip()) > 2:
                pairs.append((src.strip(), tgt.strip()))
                cnt += 1
                if cnt >= args.num_samples:
                    break
        print(f"Loaded {len(pairs)} parallel pairs from {args.parallel_dataset}")
        ds = SentenceDataset(pairs, max_len=args.max_len)
    else:
        s_iter = stream_sentences(args.num_samples)
        sents = list(s_iter)
        print(f"Loaded {len(sents)} sentences")
        ds = SentenceDataset(sents, max_len=args.max_len)
    loader = ResumableLoader(
        ds,
        batch_size=args.batch_size,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = SonarLite(device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt = TrainingCheckpointer(
        args.model_dir,
        prefix="sonar",
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

    # Training hyperparams for SONAR-style objectives
    noise_prob = args.noise_prob
    lambda_mse = args.lambda_mse
    freeze_after = args.freeze_encoder_after

    def add_noise_tokens(tokens, mask, prob):
        # tokens: [B, L] long
        if prob <= 0:
            return tokens, mask
        B, L = tokens.shape
        noisy = tokens.clone()
        # sample positions to corrupt (do not corrupt PAD positions)
        rand = torch.rand((B, L), device=tokens.device)
        corrupt = (rand < prob) & (mask.bool())
        # replace with UNK token
        noisy[corrupt] = 3
        return noisy, mask

    history = TrainingHistory(
        resolve_plot_dir(args, args.model_dir),
        run_name="sonar_lite",
        title="SonarLite auto-encoder",
        fingerprint=fingerprint,
        resume=args.resume != "never",
        formats=plot_formats(args),
        loss_label="Reconstruction + λ·MSE",
    )

    budget = EpochBudget.from_args(args, history=history, label="AE loss")

    for epoch in budget.epochs_from(resume.start_epoch):
        model.train()
        total_loss = 0.0
        steps = 0
        start = time.time()
        skip = resume.batches_to_skip(epoch)
        for batch_idx, batch in tqdm(
            loader.epoch(epoch, skip=skip), initial=skip, total=len(loader)
        ):
            # batch can be (tokens, mask) for monolingual or
            # (src_out, src_mask, tgt_out, tgt_mask) for parallel
            if isinstance(batch, tuple) and len(batch) == 2:
                tokens, mask = batch
                tokens = tokens.to(device)
                mask = mask.to(device)

                B, L = tokens.shape
                tgt_in = torch.full((B, L + 1), 0, dtype=torch.long, device=device)
                tgt_in[:, 0] = 1
                tgt_in[:, 1 : L + 1] = tokens

                tgt_out = torch.full((B, L + 1), 0, dtype=torch.long, device=device)
                tgt_out[:, :L] = tokens
                tgt_out[:, L] = 2

                # Optionally freeze encoder after certain epoch
                if freeze_after is not None and epoch >= freeze_after:
                    for n, p in model.named_parameters():
                        if (
                            n.startswith("encoder")
                            or n.startswith("pool")
                            or n.startswith("post_norm")
                            or n.startswith("tok_emb")
                        ):
                            p.requires_grad = False

                with torch.no_grad():
                    _, emb_clean = model(tokens, mask, tgt_in=None)

                if args.robust_update:
                    emb_normed, median, iqr = model.normalize_bottleneck(emb_clean)
                    with torch.no_grad():
                        model.update_running_robust(median, iqr, momentum=0.0)
                    emb_teacher = emb_normed
                else:
                    emb_teacher = emb_clean

                noisy_tokens, noisy_mask = add_noise_tokens(tokens, mask, noise_prob)

                logits, emb_noisy = model(noisy_tokens, noisy_mask, tgt_in)
                logits = logits[:, : L + 1, :]

                rec_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt_out.view(-1), ignore_index=0
                )

                if args.robust_update or model.robust_enabled:
                    emb_student = model.apply_running_robust(emb_noisy)
                else:
                    emb_student = emb_noisy

                mse_loss = torch.nn.functional.mse_loss(
                    emb_student, emb_teacher.detach()
                )

                loss = rec_loss + lambda_mse * mse_loss

            elif isinstance(batch, tuple) and len(batch) == 4:
                src_tokens, src_mask, tgt_tokens, tgt_mask = batch
                src_tokens = src_tokens.to(device)
                src_mask = src_mask.to(device)
                tgt_tokens = tgt_tokens.to(device)
                tgt_mask = tgt_mask.to(device)

                B, Ls = src_tokens.shape
                _, Lt = tgt_tokens.shape
                # build tgt_in/out from tgt_tokens
                tgt_in = torch.full((B, Lt + 1), 0, dtype=torch.long, device=device)
                tgt_in[:, 0] = 1
                tgt_in[:, 1 : Lt + 1] = tgt_tokens

                tgt_out = torch.full((B, Lt + 1), 0, dtype=torch.long, device=device)
                tgt_out[:, :Lt] = tgt_tokens
                tgt_out[:, Lt] = 2

                # translation objective: encode source, decode target
                logits, emb_src = model(src_tokens, src_mask, tgt_in)
                logits = logits[:, : Lt + 1, :]
                trans_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), tgt_out.view(-1), ignore_index=0
                )

                # teacher embedding: encode target (no grad)
                with torch.no_grad():
                    _, emb_tgt = model(tgt_tokens, tgt_mask, tgt_in=None)

                # optional robust update
                if args.robust_update:
                    emb_normed, median, iqr = model.normalize_bottleneck(emb_tgt)
                    with torch.no_grad():
                        model.update_running_robust(median, iqr, momentum=0.0)
                    emb_tgt_use = emb_normed
                else:
                    emb_tgt_use = emb_tgt

                if args.robust_update or model.robust_enabled:
                    emb_src_use = model.apply_running_robust(emb_src)
                else:
                    emb_src_use = emb_src

                mse_loss = torch.nn.functional.mse_loss(
                    emb_src_use, emb_tgt_use.detach()
                )

                # total loss: translation + MSE regularizer
                loss = trans_loss + lambda_mse * mse_loss

            else:
                # other cases (teacher vectors) not handled here.
                continue

            opt.zero_grad()
            loss.backward()
            opt.step()

            step_loss = loss.item()
            total_loss += step_loss
            steps += 1
            global_step += 1
            if global_step % 50 == 0:
                history.log_step(
                    global_step, step_loss, lr=opt.param_groups[0]["lr"]
                )
            ckpt.maybe_save(
                model,
                opt,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
            )

        elapsed = time.time() - start
        avg_loss = total_loss / steps if steps > 0 else 0.0
        print(
            f"Epoch {budget.describe(epoch)} avg loss: {avg_loss:.4f} "
            f"time: {elapsed:.1f}s"
        )
        history.log_epoch(
            epoch + 1, avg_loss, seconds=elapsed, lr=opt.param_groups[0]["lr"]
        )
        budget.observe(epoch, avg_loss)
        path = ckpt.save_epoch(model, opt, epoch=epoch, global_step=global_step)
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch + 1)
        if wandb_module is not None:
            try:
                wandb_module.log({"train/loss": avg_loss}, step=epoch + 1)
                # upload epoch checkpoint as artifact
                art = wandb_module.Artifact(f"sonar_epoch{epoch + 1}", type="model")
                art.add_file(path)
                try:
                    # prefer run.log_artifact if available
                    if hasattr(wandb_module, "run") and wandb_module.run is not None:
                        wandb_module.run.log_artifact(art)
                    elif hasattr(wandb_module, "log_artifact"):
                        wandb_module.log_artifact(art)
                except Exception:
                    pass
            except Exception:
                pass

    print(budget.summary())
    figures = history.plot()

    recorder = ResultsRecorder(
        args, run_name="sonar_lite", script="train_sonar.py", fingerprint=fingerprint
    )
    recorder.add_source(*figures, history.json_path)
    recorder.add_metrics(
        final_loss=budget.best,
        best_epoch=budget.best_epoch,
        epochs_run=budget.observed,
    )
    recorder.add_info(
        objective="translation" if args.parallel_dataset else "denoising auto-encoder",
        num_samples=args.num_samples,
        noise_prob=args.noise_prob,
        lambda_mse=args.lambda_mse,
        **budget.as_dict(),
    )
    recorder.publish()

    if writer is not None:
        try:
            writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        "--parallel_dataset",
        type=str,
        default=None,
        help="optional datasets id for parallel data",
    )
    parser.add_argument(
        "--src_col",
        type=str,
        default="source",
        help="source column name for parallel dataset",
    )
    parser.add_argument(
        "--tgt_col",
        type=str,
        default="target",
        help="target column name for parallel dataset",
    )
    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.1,
        help="token corruption prob for denoising AE",
    )
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=1.0,
        help="weight for bottleneck MSE distillation",
    )
    parser.add_argument(
        "--robust_update",
        action="store_true",
        help="update robust median/IQR from teacher embeddings each batch",
    )
    parser.add_argument(
        "--freeze_encoder_after",
        type=int,
        default=None,
        help="epoch after which encoder is frozen (None=no freeze)",
    )
    parser.add_argument("--model_dir", type=str, default="lcm_models")
    add_resume_args(parser, default_interval_steps=200)
    add_plot_args(parser)
    add_epoch_control_args(parser)
    add_results_args(parser)
    args = parser.parse_args()
    train(args)
