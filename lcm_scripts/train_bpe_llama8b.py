"""Fine-tune and benchmark a BPE + Llama-8B translation baseline.

The script uses a Llama-family 8B causal language model (default
``meta-llama/Meta-Llama-3-8B-Instruct``) with its byte-level/BPE tokenizer and
LoRA/QLoRA adapters. It evaluates generated translations with BLEU, chrF++ and
TER at clean, 10% noisy and 20% noisy inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

import torch
from torch.utils.data import Dataset
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
from plot_utils import (
    TrainingHistory,
    add_plot_args,
    plot_formats,
    plot_noise_curves,
    plot_table,
    resolve_plot_dir,
)
from results_sync import ResultsRecorder, add_results_args
from train_control import EpochBudget, add_epoch_control_args
from checkpoint_utils import (
    StageTracker,
    add_resume_args,
    config_fingerprint,
    seed_everything,
)


def build_prompt(source: str) -> str:
    return f"Translate the following sentence into Marathi.\nSource: {source}\nMarathi:"


class LlamaTranslationDataset(Dataset):
    def __init__(self, pairs: list[ParallelExample], tokenizer, max_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ex = self.pairs[idx]
        prompt = build_prompt(ex.source)
        full = prompt + " " + ex.target + self.tokenizer.eos_token
        enc = self.tokenizer(
            full, truncation=True, max_length=self.max_len, padding="max_length"
        )
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_len)[
            "input_ids"
        ]
        labels = enc["input_ids"].copy()
        labels[: len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
        labels = [
            (-100 if tok == self.tokenizer.pad_token_id else lab)
            for tok, lab in zip(enc["input_ids"], labels)
        ]
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def generate(
    model, tokenizer, sources: list[str], max_new_tokens: int, device: str
) -> list[str]:
    hyps = []
    model.eval()
    for src in tqdm(sources, desc="generate"):
        inputs = tokenizer(build_prompt(src), return_tensors="pt", truncation=True).to(
            device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        hyps.append(text.strip().split("\n")[0].strip())
    return hyps


def main():
    p = argparse.ArgumentParser(description="Fine-tune BPE + Llama 8B on BhashaSetu")
    p.add_argument("--model_name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--fraction", type=float, default=0.25)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--eval_examples", type=int, default=500)
    p.add_argument("--src_col", default=None)
    p.add_argument("--tgt_col", default=None)
    p.add_argument("--out_dir", default="runs/bpe_llama8b")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument(
        "--qlora",
        action="store_true",
        help="Load the 8B model in 4-bit for memory efficiency",
    )
    p.add_argument(
        "--noise_levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS)
    )
    p.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    add_resume_args(p, default_interval_steps=200)
    add_plot_args(p)
    add_epoch_control_args(p)
    add_results_args(p)
    args = p.parse_args()

    # The Trainer picks the training device itself, so report once up front
    # rather than at the generation stage half a run later.
    report_device()
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    os.makedirs(args.out_dir, exist_ok=True)
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if args.qlora
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

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
    if wandb_module is not None:
        try:
            wandb_module.watch(model, log="all", log_freq=100)
        except Exception:
            pass

    train_ds = LlamaTranslationDataset(train_pairs, tokenizer, args.max_len)
    # HF Trainer owns the loop here, so --train_until_plateau is enforced by a
    # callback that watches the logged training loss and asks the Trainer to
    # stop, rather than by an epoch generator we control.
    budget = EpochBudget.from_args(args, label="cross-entropy")
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=budget.cap if args.train_until_plateau else args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        # Step-based saving so a killed run loses at most `save_interval_steps`
        # of work; "epoch" could throw away hours on an 8B fine-tune.
        save_strategy="steps",
        save_steps=args.save_interval_steps,
        save_total_limit=args.max_checkpoints or None,
        seed=args.ckpt_seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds)

    if args.train_until_plateau:
        from transformers import TrainerCallback

        class PlateauStopper(TrainerCallback):
            """Stop once the logged training loss stops improving.

            The Trainer has no eval loop configured here, so the signal is the
            same running training loss it prints; `budget` applies the same
            --patience / --min_delta rule the other scripts use.
            """

            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: A002
                if not logs or "loss" not in logs:
                    return control
                budget.observe(int(state.epoch or 0), float(logs["loss"]))
                if budget.since_improvement >= budget.patience and (
                    (state.epoch or 0) >= budget.min_epochs
                ):
                    budget.stop_reason = (
                        f"plateau: no improvement > {budget.min_delta:g} in "
                        f"{budget.since_improvement} logged steps"
                    )
                    print(f"[epochs] {budget.stop_reason}; stopping", flush=True)
                    control.should_training_stop = True
                return control

        trainer.add_callback(PlateauStopper())

    # HF Trainer does its own checkpointing; hand it the resume decision.
    # `True` makes it pick the newest checkpoint-* under out_dir, restoring
    # optimizer, scheduler, RNG and the dataloader position.
    if args.resume == "never":
        resume_from = None
    elif args.resume == "auto":
        resume_from = any(
            d.startswith("checkpoint-") for d in os.listdir(args.out_dir)
        ) or None
    else:
        resume_from = args.resume
    if resume_from:
        print(f"[resume] continuing Trainer run from {args.out_dir}")
    trainer.train(resume_from_checkpoint=resume_from)

    # HF Trainer keeps its own scalar log, so the curve is transcribed from
    # `state.log_history` rather than recorded in a loop we own. A resumed
    # Trainer run restores that history too, so the curve spans the whole run.
    plot_dir = resolve_plot_dir(args, args.out_dir)
    history = TrainingHistory(
        plot_dir,
        run_name=f"bpe_llama8b_fraction{args.fraction}",
        title=f"BPE + Llama-8B LoRA (fraction {args.fraction})",
        fingerprint=fingerprint,
        # The Trainer log is authoritative and already complete; merging it with
        # an earlier sidecar could double-count steps.
        resume=False,
        formats=plot_formats(args),
        loss_label="Cross-entropy",
    )
    epoch_losses: dict[int, list[float]] = {}
    for entry in getattr(trainer.state, "log_history", []) or []:
        if "loss" not in entry:
            continue
        step = int(entry.get("step", 0))
        history.log_step(
            step, float(entry["loss"]), lr=entry.get("learning_rate"),
            grad_norm=entry.get("grad_norm"),
        )
        # `epoch` is fractional while an epoch is in flight; bucket by the
        # epoch each log line falls inside so the epoch panel is meaningful
        # even for a fractional --epochs run.
        bucket = int(math.ceil(float(entry.get("epoch", 0)) or 1e-9))
        epoch_losses.setdefault(max(bucket, 1), []).append(float(entry["loss"]))
    for epoch, losses in sorted(epoch_losses.items()):
        history.log_epoch(epoch, sum(losses) / len(losses))
    figures = history.plot()
    if args.train_until_plateau:
        print(budget.summary())

    if wandb_module is not None:
        try:
            wandb_module.log({"train/epochs": args.epochs}, step=1)
        except Exception:
            pass

    model.save_pretrained(
        os.path.join(args.out_dir, f"llama8b_lora_fraction{args.fraction}")
    )
    tokenizer.save_pretrained(
        os.path.join(args.out_dir, f"llama8b_lora_fraction{args.fraction}")
    )

    rows = []
    refs = [ex.target for ex in eval_pairs]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Autoregressive generation over the eval set with an 8B model is the most
    # expensive stage here, so each completed noise level is memoized.
    stages = StageTracker(
        os.path.join(args.out_dir, f"eval_state_fraction{args.fraction}.json"),
        fingerprint=fingerprint,
        resume=args.resume != "never",
    )

    def _eval_noise(noise):
        sources = [
            add_character_noise(ex.source, noise, seed=i)
            for i, ex in enumerate(eval_pairs)
        ]
        hyps = generate(model, tokenizer, sources, args.max_new_tokens, device)
        return {
            "BLEU": compute_bleu(hyps, refs),
            "chrF++": compute_chrf(hyps, refs),
            "TER": compute_ter(hyps, refs),
        }

    for noise in args.noise_levels:
        metrics = stages.run(f"noise={noise}", lambda noise=noise: _eval_noise(noise))
        row = {
            "model": "bpe_llama8b",
            "fraction": args.fraction,
            "noise": noise,
            **metrics,
        }
        print(row)
        rows.append(row)
        if wandb_module is not None:
            try:
                wandb_module.log(
                    {
                        f"eval/noise_{noise}/BLEU": row["BLEU"],
                        f"eval/noise_{noise}/chrF++": row["chrF++"],
                        f"eval/noise_{noise}/TER": row["TER"],
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

    if plot_dir and rows:
        formats = plot_formats(args)
        prefix = os.path.join(plot_dir, f"bpe_llama8b_fraction{args.fraction}")
        figures += plot_noise_curves(
            rows,
            f"{prefix}_noise_robustness",
            title=(
                f"BPE + Llama-8B robustness to source noise "
                f"(fraction {args.fraction})"
            ),
            formats=formats,
        )
        figures += plot_table(
            [
                [f"{r['noise']:.0%}", f"{r['BLEU']:.2f}", f"{r['chrF++']:.2f}", f"{r['TER']:.2f}"]
                for r in rows
            ],
            ["Source noise", "BLEU ↑", "chrF++ ↑", "TER ↓"],
            f"{prefix}_metrics_table",
            title=f"BPE + Llama-8B metrics (fraction {args.fraction})",
            formats=formats,
        )

    recorder = ResultsRecorder(
        args,
        run_name=f"bpe_llama8b_fraction{args.fraction}",
        script="train_bpe_llama8b.py",
        fingerprint=fingerprint,
    )
    recorder.add_source(*figures, history.json_path, out_csv)
    clean = next((r for r in rows if r["noise"] == 0.0), rows[0] if rows else {})
    recorder.add_metrics(
        final_train_loss=budget.best,
        **{f"clean_{k}": v for k, v in clean.items() if isinstance(v, float)},
    )
    recorder.add_info(
        base_model=args.model_name,
        qlora=bool(args.qlora),
        lora_rank=args.lora_rank,
        train_pairs=len(train_pairs),
        eval_pairs=len(eval_pairs),
        noise_levels=args.noise_levels,
        **budget.as_dict(),
    )
    recorder.publish()

    if wandb_module is not None:
        try:
            wandb_module.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
