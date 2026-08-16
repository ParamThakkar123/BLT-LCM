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
import multiprocessing
import time

import torch
from tqdm import tqdm
from datasets import load_dataset
from run_blt_patching import text_to_byte_tokens
from bhashasetu_utils import split_sentences
from blt_loader import BLTLoader
from base_lcm import BaseLCM
from diffusion_lcm import OneTowerDiffusionLCM, TwoTowerDiffusionLCM
from quant_lcm import QuantLCM
from eval_metrics import compute_all
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
    DEFAULT_FINGERPRINT_IGNORE,
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
            # Shared with every other loader in the repo. The separator class
            # used to be a mangled literal ("[.αÑñ]") that had lost the
            # Devanagari danda in an encoding round-trip, so Marathi text was
            # only ever split on the ASCII full stop -- which most sentences do
            # not contain. Documents came out as paragraph-sized "sentences",
            # which is both wrong and far more expensive to encode.
            sents = split_sentences(text)
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run the forward/backward in bfloat16 autocast on CUDA. Roughly "
        "halves step time on Ampere and newer at a small numerical cost; the "
        "loss itself is still reduced in fp32.",
    )
    parser.add_argument(
        "--scaler_fit_samples",
        type=int,
        default=50_000,
        help="Concept vectors used to fit the robust scaler (LCM Eq. 4). "
        "torch.quantile sorts its input, so this is a transient VRAM spike; "
        "50k is statistically ample for a per-dimension median and IQR.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Steps between per-step scalar logs (TensorBoard / W&B). The "
        "gradient-norm computation walks every parameter, so logging it every "
        "step is a real cost on a 12-layer, 2048-wide model.",
    )
    add_resume_args(parser, default_interval_steps=1000)
    add_plot_args(parser)
    add_epoch_control_args(parser)
    add_results_args(parser)
    args = parser.parse_args()

    device = report_device(args.device)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    if args.amp and amp_device != "cuda":
        print("  [amp] no CUDA device; running in fp32")
        args.amp = False
    seed_everything(args.ckpt_seed)
    # --batch_size is a VRAM knob, not part of what the run computes: the
    # auto-setup driver halves it and retries after a CUDA OOM, and that retry
    # has to resume the interrupted run (and reuse the cached embeddings, which
    # a frozen encoder produces identically at any batch size) instead of
    # aborting on a fingerprint mismatch.
    fingerprint = config_fingerprint(
        args, ignore=DEFAULT_FINGERPRINT_IGNORE | {"batch_size"}
    )
    # Fraction in the run name: one encode pass per fraction, so the training
    # curve and the published results directory must not be shared between them.
    # The checkpoints are separated by --model_dir instead (the eval scripts
    # glob for a fixed `lcm_blt_best.pth` inside a per-run directory).
    run_name = f"lcm_blt_{args.lcm_variant}_fraction{args.fraction:g}"
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
    # Training-curve record. It lives next to the checkpoints and carries the
    # run fingerprint, so a resumed run continues the same curve instead of
    # drawing one that starts at the restart.
    history = TrainingHistory(
        resolve_plot_dir(args, args.model_dir),
        run_name=run_name,
        title=f"BLT-LCM ({args.lcm_variant}, fraction {args.fraction:g})",
        fingerprint=fingerprint,
        resume=args.resume != "never",
        formats=plot_formats(args),
        loss_label="MSE loss" if args.lcm_variant == "base" else "Loss",
    )
    t0 = time.time()
    print("Preparing data list...")
    docs = prepare_data(args.num_docs, fraction=args.fraction)
    print(f"Data loading: {time.time() - t0:.1f}s  ({len(docs)} docs)")

    print("Loading BLT loader...")
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    t1 = time.time()
    print("Encoding with BLT (this may take a while)...")
    # TF32 is enabled for every script by report_device() above.
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
    # torch.quantile sorts its input, so fitting on the full 200k x 1024 sample
    # committed ~800 MiB plus a sort buffer on the GPU right before the model
    # was allocated. The median and IQR of 50k samples are indistinguishable
    # from those of 200k at this dimensionality, so the extra spike bought
    # nothing. Subsample on the CPU and move only what is kept.
    if scaler_sample.shape[0] > args.scaler_fit_samples:
        idx = torch.randperm(scaler_sample.shape[0])[: args.scaler_fit_samples]
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

    # Epoch budget: a fixed --epochs, or --train_until_plateau to keep going
    # while the loss improves. Seeded from the history so a resumed run keeps
    # the patience counter the original process had reached.
    budget = EpochBudget.from_args(args, history=history, label="MSE loss")

    for epoch in budget.epochs_from(resume.start_epoch):
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
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            seq_len = tgt.shape[1]
            if seq_len == 0:
                continue
            optim.zero_grad()
            # bf16 has fp32's exponent range, so no GradScaler is needed. The
            # MSE reduction stays in fp32 below: unlike cross_entropy it is not
            # on autocast's promotion list, and summing many squared residuals
            # in bf16 loses accuracy for no speed gain.
            with torch.autocast(
                device_type=amp_device, dtype=torch.bfloat16, enabled=args.amp
            ):
                if args.lcm_variant == "base":
                    # One causal pass covers every position: output at t
                    # predicts the concept at t+1. The old loop re-ran the model
                    # once per position, which was O(L^2) forwards per gradient.
                    preds = model.forward_all(src)
                else:
                    # Diffusion and quantized variants own their objective; they
                    # take the full document (context + target) rather than a
                    # pre-shifted (src, tgt) pair.
                    doc = torch.cat([src, tgt[:, -1:]], dim=1)
                    loss = model.loss(doc)

            if args.lcm_variant == "base":
                preds = preds.float()
                # Mask padded positions so they don't pull predictions to zero.
                valid = (tgt.abs().sum(dim=-1, keepdim=True) > 0).float()
                loss = (
                    ((preds - tgt).pow(2) * valid).sum()
                    / valid.sum().clamp(min=1)
                    / tgt.shape[-1]
                )
            loss.backward()
            optim.step()
            # Accumulated on the device and read back once per epoch. Calling
            # .item() here forced a host sync on every single step purely to
            # maintain a running average that is only printed at epoch end.
            total = total + loss.detach()
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
            # Step logging. The gradient norm walks every parameter and used to
            # be computed TWICE per step when both loggers were on -- once for
            # TensorBoard and again for W&B -- and on every step regardless. It
            # is now computed once, only on the steps that actually log.
            log_now = (
                writer is not None or wandb_module is not None or history.enabled
            ) and (global_step % max(args.log_every, 1) == 0)
            if log_now:
                step_loss = loss.item()
                lr = optim.param_groups[0]["lr"]
                # One fused norm over all gradients instead of a per-parameter
                # .item() sync each.
                grads = [
                    p.grad.detach() for p in model.parameters() if p.grad is not None
                ]
                total_norm = (
                    float(torch.linalg.vector_norm(torch.stack(
                        [torch.linalg.vector_norm(g) for g in grads]
                    )))
                    if grads
                    else 0.0
                )

                history.log_step(
                    global_step, step_loss, lr=lr, grad_norm=total_norm
                )
                if writer is not None:
                    writer.add_scalar("train/step_loss", step_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    writer.add_scalar("train/grad_norm", total_norm, global_step)
                if wandb_module is not None:
                    try:
                        wandb_module.log(
                            {
                                "train/step_loss": step_loss,
                                "train/lr": lr,
                                "train/grad_norm": total_norm,
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

        elapsed = time.time() - start
        # The single host sync for the whole epoch's running loss.
        avg_loss = float(total) / n if n > 0 else float("nan")
        print(
            f"Epoch {budget.describe(epoch)} avg loss: {avg_loss:.4f} "
            f"time: {elapsed:.1f}s"
        )
        peak_vram = None
        if device.type == "cuda":
            peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"Peak VRAM epoch {epoch + 1}: {peak_vram:.2f} GB")
            torch.cuda.reset_peak_memory_stats(device)
        history.log_epoch(
            epoch + 1,
            avg_loss,
            seconds=elapsed,
            lr=optim.param_groups[0]["lr"],
            peak_vram_gb=peak_vram,
        )
        # Drives --train_until_plateau: the loop ends when this stops improving.
        budget.observe(epoch, avg_loss)
        # End-of-epoch snapshot goes into the rolling `_last` checkpoint only.
        # No per-epoch `_epoch{N}` files: the run keeps exactly two checkpoints,
        # `_last` (the resume target) and `_best`.
        ckpt.save(
            model,
            optim,
            epoch=epoch,
            batch_in_epoch=0,
            epoch_completed=True,
            global_step=global_step,
            best_score=best_score,
        )
        print(f"[checkpoint] epoch {epoch + 1} -> {ckpt.last_path}", flush=True)

        # per-epoch logging
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss if n > 0 else 0.0, epoch + 1)
        if wandb_module is not None:
            try:
                # use monotonic global_step for wandb logging
                wandb_module.log(
                    {"train/loss": avg_loss if n > 0 else 0.0}, step=global_step
                )
            except Exception:
                pass

        # optional evaluation and best-checkpointing. The eval files are an
        # explicit request, so it runs whenever either sink -- TensorBoard or
        # the plotted history -- is there to receive the numbers.
        if args.eval_hyp and args.eval_ref and (writer is not None or history.enabled):
            with open(args.eval_hyp, encoding="utf-8") as f:
                hyps = [l.strip() for l in f if l.strip()]
            with open(args.eval_ref, encoding="utf-8") as f:
                refs = [l.strip() for l in f if l.strip()]
            metrics = compute_all(
                hyps, refs, comet_model_name=args.comet_model, device=str(device)
            )
            history.log_eval(epoch + 1, metrics)
            if writer is not None:
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
            monitor_score = -(avg_loss if n > 0 else float("inf"))

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

    print(budget.summary())

    # Training curve + diagnostics dashboard for the whole run.
    figures = history.plot()

    # Collect figures, loss history and the full hyperparameter set into the
    # repository and (with --push_results) push them.
    recorder = ResultsRecorder(
        args,
        run_name=run_name,
        script="train_lcm_blt.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, history.json_path)
    recorder.add_metrics(
        final_train_loss=budget.best,
        best_epoch=budget.best_epoch,
        epochs_run=budget.observed,
        best_score=best_score,
    )
    recorder.add_info(
        lcm_variant=args.lcm_variant,
        documents=len(docs),
        concepts=int(sum(int(s.shape[0]) for s in embeddings_seqs)),
        embed_dim=embed_dim,
        checkpoint=ckpt.best_path,
        **budget.as_dict(),
    )
    recorder.publish()

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
