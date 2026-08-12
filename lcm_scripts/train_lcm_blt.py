"""
Train BLT-LCM: use BLT-derived embeddings (from BLTLoader) to train same LCM architecture.

Usage:
  python lcm_scripts/train_lcm_blt.py --entropy_model patching_scratch/entropy_model_marathi.pt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from dotenv import load_dotenv
load_dotenv()

import argparse
import re
import multiprocessing
import time

import torch
from tqdm import tqdm
from datasets import load_dataset
from run_blt_patching import text_to_byte_tokens
from blt_loader import BLTLoader
from base_lcm import BaseLCM
from diffusion_lcm import OneTowerDiffusionLCM, TwoTowerDiffusionLCM
from quant_lcm import QuantLCM
from eval_metrics import compute_all
from experiment_config import setup_logging
from device_utils import report_device
from checkpoint_utils import (
    ResumableLoader,
    TrainingCheckpointer,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    seed_everything,
)


def prepare_data(num_docs=500, max_sent_per_doc=20, fraction=1.0):
    ds = load_dataset("ParamTh/BhashaSetu", split="train")
    total = len(ds)
    num_to_select = int(total * fraction)
    ds = ds.shuffle(seed=42).select(range(num_to_select))
    docs = []
    buf = []
    for row in tqdm(ds, desc="Loading docs"):
        text = row.get("marathi", "")
        if text and len(text.strip()) > 0:
            sents = [
                s.strip()
                for s in re.split(r"[.αÑñ]", text.replace("\n", " "))
                if s.strip()
            ]
            buf.extend(sents)
            while len(buf) >= max_sent_per_doc:
                docs.append(buf[:max_sent_per_doc])
                buf = buf[max_sent_per_doc:]
                if len(docs) >= num_docs:
                    return docs
    if buf and len(docs) < num_docs:
        docs.append(buf)
    return docs


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_seqs, eot_embedding=None):
        # LCM §2.3.1: every training document is suffixed with the encoded
        # "End of text." concept, so the model learns to emit it and the
        # inference-time stop criterion has something real to match against.
        data = []
        for seq in embeddings_seqs:
            if len(seq) < 2:
                continue
            if eot_embedding is not None:
                seq = torch.cat([seq, eot_embedding.unsqueeze(0).to(seq.dtype)], dim=0)
            data.append(seq)
        self.data = data

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
    parser.add_argument(
        "--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY")
    )
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
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="Fraction of dataset to use"
    )
    parser.add_argument(
        "--embed_cache",
        type=str,
        default="blt_embeddings_cache.pth",
        help="Path to load/save precomputed embeddings (torch.save format)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="lcm_models",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--lcm_variant",
        choices=["base", "two_tower", "one_tower", "quant"],
        default="base",
        help="Which LCM to train. 'base' is the MSE Base-LCM (LCM §2.3.1); "
        "'two_tower'/'one_tower' are the diffusion variants (§2.3.4/§2.3.3) that "
        "the paper finds outperform it; 'quant' is the RVQ model (§2.3.5).",
    )
    parser.add_argument("--model_dim", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of discretized diffusion timesteps T (diffusion variants).",
    )
    parser.add_argument(
        "--noise_schedule",
        choices=["cosine", "quadratic", "sigmoid"],
        default="cosine",
        help="Noise schedule for the diffusion variants (LCM §2.3.2).",
    )
    parser.add_argument(
        "--n_codebooks",
        type=int,
        default=64,
        help="RVQ codebooks for --lcm_variant quant (paper uses 64).",
    )
    parser.add_argument(
        "--units_per_codebook",
        type=int,
        default=8192,
        help="Units per RVQ codebook for --lcm_variant quant (paper uses 8192).",
    )
    parser.add_argument(
        "--quant_target",
        choices=["discrete", "continuous"],
        default="discrete",
        help="Quant-LCM-d (cross-entropy over units) or Quant-LCM-c (MSE).",
    )
    parser.add_argument(
        "--quant_fit_samples",
        type=int,
        default=200_000,
        help="Concept vectors used to fit the RVQ codebooks.",
    )
    add_resume_args(parser, default_interval_steps=1000)
    args = parser.parse_args()

    device = report_device(args.device)
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)
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
    t0 = time.time()
    print("Preparing data list...")
    docs = prepare_data(args.num_docs, fraction=args.fraction)
    print(f"Data loading: {time.time() - t0:.1f}s  ({len(docs)} docs)")

    print("Loading BLT loader...")
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    t1 = time.time()
    print("Encoding with BLT (this may take a while)...")
    torch.set_float32_matmul_precision(
        "high"
    )  # Enable TensorFloat32 for better performance
    def _encode_all():
        flat_sents = []
        doc_indices = []
        for i, sents in enumerate(docs):
            for sent in sents:
                flat_sents.append(sent)
                doc_indices.append(i)
        print(f"Encoding {len(flat_sents)} sentences from {len(docs)} documents")
        tokenized = [text_to_byte_tokens(sent) for sent in flat_sents]
        # The whole corpus goes in at once so the loader can length-sort across
        # all of it and size each forward by padded byte count. Slicing into
        # fixed-size batches here would cap sorting at one slice, and every
        # slice would be padded out to its own longest sentence.
        embeds = blt.encode_tokens_to_tensor(tokenized, show_progress=True).cpu()

        # Reconstruct per-document sequences
        seqs = [[] for _ in range(len(docs))]
        for row, didx in enumerate(doc_indices):
            seqs[didx].append(embeds[row])

        # stack per-document tensors
        for i in range(len(seqs)):
            if len(seqs[i]) == 0:
                seqs[i] = torch.empty((0, blt.dim))
            else:
                seqs[i] = torch.stack(seqs[i], dim=0)

        print(f"Encoding: {time.time() - t1:.1f}s")
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"Peak VRAM after encoding: {peak_vram:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)
        return seqs

    def _cache_is_usable(seqs):
        """A cache is only useful if it holds at least one trainable sequence."""
        return bool(seqs) and any(
            getattr(s, "shape", (0,))[0] >= 2 for s in seqs if hasattr(s, "shape")
        )

    # Encoding the corpus can dominate a short run and is fully determined by the
    # config, so a resumed run reloads it instead of paying for it again.
    embeddings_seqs = cached_torch(
        args.embed_cache,
        _encode_all,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=_cache_is_usable,
        label="BLT embeddings",
    )

    # The stop concept must be the encoder's own embedding of "End of text.",
    # not a learned free parameter (LCM §2.3.1).
    eot_embedding = blt.encode_sentences_batch(["End of text."])[0].detach().cpu()

    dataset = EmbeddingDataset(embeddings_seqs, eot_embedding=eot_embedding)
    dataloader = ResumableLoader(
        dataset,
        batch_size=args.batch_size,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate,
    )

    embed_dim = embeddings_seqs[0].shape[1]
    max_concepts = max(int(s.shape[0]) for s in embeddings_seqs) + 2
    max_seq_len = max(128, max_concepts)

    if args.lcm_variant == "base":
        model = BaseLCM(
            embed_dim=embed_dim,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=max_seq_len,
        ).to(device)
    elif args.lcm_variant == "two_tower":
        model = TwoTowerDiffusionLCM(
            embed_dim=embed_dim,
            model_dim=args.model_dim,
            context_layers=max(1, args.n_layers // 3),
            denoiser_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=max_seq_len,
            timesteps=args.diffusion_steps,
            schedule=args.noise_schedule,
        ).to(device)
    elif args.lcm_variant == "one_tower":
        model = OneTowerDiffusionLCM(
            embed_dim=embed_dim,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=max_seq_len,
            timesteps=args.diffusion_steps,
            schedule=args.noise_schedule,
        ).to(device)
    else:  # quant
        model = QuantLCM(
            embed_dim=embed_dim,
            model_dim=args.model_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seq_len=max_seq_len,
            n_codebooks=args.n_codebooks,
            units_per_codebook=args.units_per_codebook,
            target=args.quant_target,
        ).to(device)
    print(f"LCM variant: {args.lcm_variant}")

    # Fit the robust scaler once, on a sample of the actual training concepts
    # (LCM Eq. 4). Without this the scaler is the identity and the model cannot
    # map predictions back into the encoder's coordinate scale.
    scaler_sample = torch.cat(
        [s for s in embeddings_seqs if s.shape[0] > 0], dim=0
    )
    if scaler_sample.shape[0] > 200_000:
        idx = torch.randperm(scaler_sample.shape[0])[:200_000]
        scaler_sample = scaler_sample[idx]
    scaler_sample = scaler_sample.to(device)
    model.fit_normalizer(scaler_sample)
    print(f"Fitted robust scaler on {scaler_sample.shape[0]} concept vectors")

    if args.lcm_variant == "base":
        model.set_eot_embedding(eot_embedding)
    elif args.lcm_variant == "quant":
        # RVQ codebooks must be fitted before the model can encode targets.
        print(f"Fitting {args.n_codebooks} RVQ codebooks...")
        model.fit_quantizer(scaler_sample[: args.quant_fit_samples])
    if wandb_module is not None:
        try:
            wandb_module.watch(model, log="all", log_freq=100)
        except Exception:
            pass
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt = TrainingCheckpointer(
        args.model_dir,
        # Variant in the prefix so training two variants into the same
        # --model_dir does not have them overwrite each other's checkpoints.
        prefix="lcm_blt" if args.lcm_variant == "base" else f"lcm_blt_{args.lcm_variant}",
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(
        ckpt.load(args.resume, map_location=device), model, optim
    )
    # keep a single global step counter across epochs so logging steps are monotonic
    global_step = resume.global_step
    best_score = resume.best_score
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}, global step {global_step}"
        )

    for epoch in range(resume.start_epoch, args.epochs):
        model.train()
        total = 0.0
        n = 0
        start = time.time()
        skip = resume.batches_to_skip(epoch)
        # don't reset `global_step` here — it must be monotonically increasing for loggers
        for batch_idx, (src, tgt) in tqdm(
            dataloader.epoch(epoch, skip=skip),
            initial=skip,
            total=len(dataloader),
        ):
            src = src.to(device)
            tgt = tgt.to(device)
            seq_len = tgt.shape[1]
            if seq_len == 0:
                continue
            optim.zero_grad()
            if args.lcm_variant == "base":
                # One causal pass covers every position: output at t predicts
                # the concept at t+1. The old loop re-ran the model once per
                # position, which was O(L^2) forward passes for one gradient.
                preds = model.forward_all(src)
                # Mask padded positions so they don't pull predictions to zero.
                valid = (tgt.abs().sum(dim=-1, keepdim=True) > 0).float()
                loss = (
                    ((preds - tgt).pow(2) * valid).sum()
                    / valid.sum().clamp(min=1)
                    / tgt.shape[-1]
                )
            else:
                # Diffusion and quantized variants own their objective; they
                # take the full document (context + target) rather than a
                # pre-shifted (src, tgt) pair.
                doc = torch.cat([src, tgt[:, -1:]], dim=1)
                loss = model.loss(doc)
            loss.backward()
            optim.step()
            total += loss.item()
            n += 1
            global_step += 1

            ckpt.maybe_save(
                model,
                optim,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
                best_score=best_score,
            )
            if writer is not None:
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

            if wandb_module is not None:
                try:
                    lr = optim.param_groups[0]["lr"]
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2).item()
                            total_norm += param_norm * param_norm
                    total_norm = total_norm**0.5
                    wandb_module.log(
                        {
                            "train/step_loss": loss.item(),
                            "train/lr": lr,
                            "train/grad_norm": total_norm,
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        elapsed = time.time() - start
        avg_loss = total / n if n > 0 else float("nan")
        print(f"Epoch {epoch + 1} avg loss: {avg_loss:.4f} time: {elapsed:.1f}s")
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"Peak VRAM epoch {epoch + 1}: {peak_vram:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)
        ckpt.save_epoch(
            model,
            optim,
            epoch=epoch,
            global_step=global_step,
            best_score=best_score,
        )

        # per-epoch logging
        if writer is not None:
            writer.add_scalar("train/loss", total / n if n > 0 else 0.0, epoch + 1)
        if wandb_module is not None:
            try:
                # use monotonic global_step for wandb logging
                wandb_module.log(
                    {"train/loss": total / n if n > 0 else 0.0}, step=global_step
                )
            except Exception:
                pass

        # optional evaluation and best-checkpointing
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
            # also send eval metrics to wandb if enabled, using monotonic step
            if wandb_module is not None:
                try:
                    wandb_module.log(metrics, step=global_step)
                except Exception:
                    pass
        else:
            monitor_score = -(total / n if n > 0 else float("inf"))

        # `best_score` rides along in the checkpoint, so a resumed run keeps
        # comparing against the best epoch of the *whole* run rather than
        # treating the first epoch after the restart as an automatic winner.
        if monitor_score is not None and (
            best_score is None or monitor_score > best_score
        ):
            best_score = monitor_score
            ckpt.save_best(
                model,
                optim,
                epoch=epoch,
                batch_in_epoch=0,
                epoch_completed=True,
                global_step=global_step,
                best_score=best_score,
            )
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
