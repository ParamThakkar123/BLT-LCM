"""
Fine-tune BLT-LCM: load a pre-trained LCM model and fine-tune on BLT-derived embeddings.

Usage:
  python lcm_scripts/finetune_lcm.py --checkpoint lcm_models/lcm_blt_best.pth --entropy_model ../patching_scratch/entropy_model_marathi.pt
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import time
import torch
from tqdm import tqdm

from blt_loader import BLTLoader
from base_lcm import BaseLCM
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
    ResumableLoader,
    TrainingCheckpointer,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    load_model_state,
    seed_everything,
)

from peft import LoraConfig, get_peft_model

PEFT_AVAILABLE = True


def prepare_data(fraction=1.0, max_sent_per_doc=20):
    from bhashasetu_utils import load_bhashasetu_documents

    return load_bhashasetu_documents(
        fraction=fraction,
        max_sent_per_doc=max_sent_per_doc,
        text_col="marathi",
    )

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
    src_p = torch.zeros(B, max_len, emb_dim, device=srcs[0].device, dtype=srcs[0].dtype)
    tgt_p = torch.zeros(B, max_len, emb_dim, device=srcs[0].device, dtype=srcs[0].dtype)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_p[i, : s.shape[0]] = s
        tgt_p[i, : t.shape[0]] = t
    return src_p, tgt_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained LCM model checkpoint",
    )
    parser.add_argument("--entropy_model", type=str, required=True)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of BhashaSetu documents for fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Fine-tuning learning rate (lower than training)",
    )
    parser.add_argument(
        "--freeze_prenet",
        action="store_true",
        help="Freeze PreNet layers during fine-tuning",
    )
    parser.add_argument(
        "--freeze_postnet",
        action="store_true",
        help="Freeze PostNet layers during fine-tuning",
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Number of transformer layers to freeze from the bottom",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA (parameter-efficient) fine-tuning of the pretrained LCM",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["linear"],
        help="Target modules for LoRA (e.g., linear for all Linear layers)",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="TensorBoard log dir")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation",
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
        "--model_dir",
        type=str,
        default="lcm_models",
        help="Directory for fine-tuning checkpoints",
    )
    parser.add_argument(
        "--embed_cache",
        type=str,
        default=None,
        help="Optional path for the cached BLT encodings, so a resumed run "
        "skips re-encoding the corpus.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run the forward/backward in bfloat16 autocast on CUDA. Roughly "
        "halves step time on Ampere and newer at a small numerical cost; the "
        "loss itself is still reduced in fp32.",
    )
    add_resume_args(parser, default_interval_steps=200)
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

    print("Preparing fine-tuning data list...")
    docs = prepare_data(args.fraction)

    print("Loading BLT loader...")
    blt = BLTLoader(entropy_model_path=args.entropy_model, device=str(device))

    print("Encoding with BLT for fine-tuning...")

    def _encode_all():
        # The whole corpus goes in as one call so the loader can length-sort
        # across all of it and size each forward by padded byte count. Encoding
        # document by document capped that sorting at one document and padded
        # every forward out to that document's longest sentence.
        flat: list[str] = []
        owners: list[int] = []
        for i, sents in enumerate(docs):
            for sent in sents:
                flat.append(sent)
                owners.append(i)

        embeds = blt.encode_sentences(flat).detach().cpu()

        grouped: list[list] = [[] for _ in docs]
        for row, owner in enumerate(owners):
            grouped[owner].append(embeds[row])
        return [
            torch.stack(rows) if rows else torch.empty(0, blt.dim) for rows in grouped
        ]

    embeddings_seqs = cached_torch(
        args.embed_cache,
        _encode_all,
        fingerprint=fingerprint,
        resume=args.resume != "never",
        validate=lambda s: bool(s),
        label="BLT embeddings",
    )

    dataset = EmbeddingDataset(embeddings_seqs)
    dataloader = ResumableLoader(
        dataset,
        batch_size=args.batch_size,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate,
    )

    # Load pre-trained model. `load_model_state` unwraps both the resumable
    # checkpoint payload and the bare state_dict files older runs produced.
    max_concepts = max(int(s.shape[0]) for s in embeddings_seqs) + 2
    model = BaseLCM(
        embed_dim=embeddings_seqs[0].shape[1],
        model_dim=2048,
        n_layers=12,
        n_heads=16,
        max_seq_len=max(128, max_concepts),
    ).to(device)
    try:
        model.load_state_dict(load_model_state(args.checkpoint, map_location=device))
    except RuntimeError as e:
        raise RuntimeError(
            f"Could not load {args.checkpoint} into BaseLCM. Checkpoints written "
            f"before BaseLCM became decoder-only (with a fitted robust scaler and "
            f"a buffered EOT concept) have an incompatible parameter layout and "
            f"must be retrained with train_lcm_blt.py.\n\nUnderlying error: {e}"
        ) from e
    print(f"Loaded pre-trained model from {args.checkpoint}")
    if not model.scaler.is_fitted:
        sample = torch.cat([s for s in embeddings_seqs if s.shape[0] > 0], dim=0)
        model.fit_normalizer(sample[:200_000].to(device))
        print("Checkpoint had no fitted scaler; fitted one on the fine-tuning data")

    # Apply LoRA (parameter-efficient fine-tuning) if enabled.
    # Note: this fine-tunes the *pretrained* BaseLCM checkpoint, so LoRA adapters
    # are appropriate. 4-bit "QLoRA" was intentionally removed: BaseLCM is a
    # custom fp32 module (not a HF from_pretrained model with a bitsandbytes
    # 4-bit config), so no genuine 4-bit quantization was ever applied.
    if args.lora:
        # task_type=None: BaseLCM has a custom forward(src_embs, tgt_embs) rather
        # than a HF input_ids interface, so the generic PeftModel wrapper (which
        # forwards *args positionally) is required; a task-specific wrapper would
        # inject input_ids and break the forward pass.
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=None,
        )
        model = get_peft_model(model, lora_config)
        print(f"Applied LoRA with rank {args.lora_rank}")

    # Freeze layers if specified
    if args.freeze_prenet:
        for param in model.prenet.parameters():
            param.requires_grad = False
        print("Froze PreNet layers")

    if args.freeze_postnet:
        for param in model.postnet.parameters():
            param.requires_grad = False
        print("Froze PostNet layers")

    if args.freeze_layers > 0:
        for i in range(args.freeze_layers):
            for param in model.layers[i].parameters():
                param.requires_grad = False
        print(f"Froze first {args.freeze_layers} transformer layers")

    if wandb_module is not None:
        try:
            wandb_module.watch(model, log="all", log_freq=100)
        except Exception as e:
            logging.warning(f"wandb.watch failed: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt = TrainingCheckpointer(
        args.model_dir,
        prefix="lcm_finetuned",
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(ckpt.load(args.resume, map_location=device), model, optim)
    # One monotonic counter for the whole run. It used to restart from
    # `epoch * 1000000` each epoch and only advance when a TensorBoard writer
    # was attached, which made step numbers meaningless for resume.
    global_step = resume.global_step
    best_score = resume.best_score
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}, global step {global_step}"
        )

    history = TrainingHistory(
        resolve_plot_dir(args, args.model_dir),
        run_name="lcm_finetuned",
        title="BLT-LCM fine-tuning" + (" (LoRA)" if args.lora else ""),
        fingerprint=fingerprint,
        resume=args.resume != "never",
        formats=plot_formats(args),
        loss_label="MSE loss",
    )

    budget = EpochBudget.from_args(args, history=history, label="MSE loss")

    for epoch in budget.epochs_from(resume.start_epoch):
        model.train()
        total = 0.0
        n = 0
        start = time.time()
        skip = resume.batches_to_skip(epoch)
        for batch_idx, (src, tgt) in tqdm(
            dataloader.epoch(epoch, skip=skip), initial=skip, total=len(dataloader)
        ):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optim.zero_grad()
            # Single causal pass; see train_lcm_blt.py for the rationale.
            # PEFT wrappers forward unknown attributes to the wrapped module, so
            # this reaches BaseLCM.forward_all through the LoRA-injected layers.
            # bf16 needs no GradScaler; the MSE is reduced in fp32 below.
            with torch.autocast(
                device_type=amp_device, dtype=torch.bfloat16, enabled=args.amp
            ):
                preds = model.forward_all(src)
            preds = preds.float()
            valid = (tgt.abs().sum(dim=-1, keepdim=True) > 0).float()
            loss = (
                ((preds - tgt).pow(2) * valid).sum()
                / valid.sum().clamp(min=1)
                / tgt.shape[-1]
            )
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
            # The history is sampled; TensorBoard keeps its per-step cadence.
            sample_history = history.enabled and global_step % 50 == 0
            if writer is not None or sample_history:
                step_loss = loss.item()
                lr = optim.param_groups[0]["lr"]
                try:
                    # One fused norm over all gradients, as in train_lcm_blt.py.
                    # The per-parameter .norm().item() loop this replaces forced
                    # a device sync for every single parameter tensor.
                    grads = [
                        p.grad.detach()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                    total_norm = (
                        float(
                            torch.linalg.vector_norm(
                                torch.stack(
                                    [torch.linalg.vector_norm(g) for g in grads]
                                )
                            )
                        )
                        if grads
                        else 0.0
                    )
                except Exception as e:
                    logging.warning(f"Grad norm logging failed: {e}")
                    total_norm = None
                if sample_history:
                    history.log_step(
                        global_step, step_loss, lr=lr, grad_norm=total_norm
                    )
                if writer is not None:
                    writer.add_scalar("finetune/step_loss", step_loss, global_step)
                    writer.add_scalar("finetune/lr", lr, global_step)
                    if total_norm is not None:
                        writer.add_scalar(
                            "finetune/grad_norm", total_norm, global_step
                        )

        elapsed = time.time() - start
        avg_loss = total / n if n > 0 else float("nan")
        print(
            f"Fine-tune Epoch {budget.describe(epoch)} avg loss: {avg_loss:.4f} "
            f"time: {elapsed:.1f}s"
        )
        history.log_epoch(
            epoch + 1, avg_loss, seconds=elapsed, lr=optim.param_groups[0]["lr"]
        )
        budget.observe(epoch, avg_loss)
        ckpt.save_epoch(
            model, optim, epoch=epoch, global_step=global_step, best_score=best_score
        )

        if writer is not None:
            writer.add_scalar("finetune/loss", total / n if n > 0 else 0.0, epoch + 1)
        if wandb_module is not None:
            try:
                wandb_module.log(
                    {"finetune/loss": total / n if n > 0 else 0.0}, step=epoch + 1
                )
            except Exception as e:
                logging.warning(f"wandb epoch log failed: {e}")

        # The eval files are an explicit request, so the pass runs whenever
        # either sink -- TensorBoard or the plotted history -- can receive it.
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
        else:
            monitor_score = -(total / n if n > 0 else float("inf"))

        # `best_score` travels in the checkpoint, so a resumed run keeps
        # comparing against the best epoch of the whole run rather than treating
        # the first epoch after a restart as an automatic winner.
        if monitor_score is not None and (
            best_score is None or monitor_score > best_score
        ):
            best_score = monitor_score
            ckpt.save_best(
                model,
                optim,
                epoch=epoch,
                epoch_completed=True,
                global_step=global_step,
                best_score=best_score,
            )

    print(budget.summary())
    figures = history.plot()

    recorder = ResultsRecorder(
        args,
        run_name="lcm_finetuned",
        script="finetune_lcm.py",
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
        base_checkpoint=args.checkpoint,
        lora=bool(args.lora),
        documents=len(docs),
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
