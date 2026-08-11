"""
Training script for Base-LCM
"""

import os

import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

from base_lcm import BaseLCM
from data_loader import LCMDataset, collate_fn
from blt_loader import BLTLoader
from device_utils import report_device
from checkpoint_utils import (
    ResumableLoader,
    TrainingCheckpointer,
    add_resume_args,
    cached_torch,
    config_fingerprint,
    seed_everything,
)


def train_base_lcm(args):
    device = report_device()
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)

    # Load BLT
    blt_loader = BLTLoader(
        entropy_model_path="../patching_scratch/entropy_model_marathi.pt",
        device=str(device),
    )

    # Create model. The robust scaler and EOT concept are installed after the
    # corpus has been encoded, below.
    model = BaseLCM(
        embed_dim=1024, model_dim=2048, n_layers=12, n_heads=16, max_seq_len=128
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # Load real Marathi data from BhashaSetu
    from datasets import load_dataset

    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    marathi_texts = []
    for row in ds:
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            marathi_texts.append(text)
        if len(marathi_texts) >= 1000:  # Small subset for testing
            break

    # Split into train (80%) and validation (20%) to prevent data leakage
    split = int(len(marathi_texts) * 0.8)
    train_texts = marathi_texts[:split]
    val_texts = marathi_texts[split:]

    def segment_texts(texts, max_sents=10):
        sentences_list = []
        for text in texts[:100]:
            sents = [s.strip() for s in text.split(".") if s.strip()]
            if len(sents) > 1:
                sentences_list.append(sents[:max_sents])
        return sentences_list

    train_sentences_list = segment_texts(train_texts)
    val_sentences_list = segment_texts(val_texts)

    # Encode all sentences. The BLT encoder is frozen, so this is pure with
    # respect to the config and a resumed run reloads it from the cache.
    train_sentences = [s for seq in train_sentences_list for s in seq]
    val_sentences = [s for seq in val_sentences_list for s in seq]
    all_embeddings = cached_torch(
        args.embed_cache,
        lambda: blt_loader.encode_sentences(train_sentences + val_sentences).cpu(),
        fingerprint=fingerprint,
        resume=args.resume != "never",
        label="BLT embeddings",
    )
    train_embs = all_embeddings[: len(train_sentences)]
    val_embs = all_embeddings[len(train_sentences) :]

    # Create dataset from embeddings
    class RealDataset(torch.utils.data.Dataset):
        def __init__(self, sentences_list, all_embeddings, embed_dim=1024):
            self.data = []
            idx = 0
            for seq_sents in sentences_list:
                seq_embs = all_embeddings[idx : idx + len(seq_sents)]
                idx += len(seq_sents)
                if len(seq_embs) > 1:
                    self.data.append((seq_embs[:-1], seq_embs[1:]))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # LCM Eq. (4): fit the PreNet/PostNet scaler once on the real concepts, and
    # install the encoded end-of-text concept as the stop signal.
    model.fit_normalizer(all_embeddings[:200_000].to(device))
    model.set_eot_embedding(blt_loader.encode_sentences(["End of text."])[0])

    dataset = RealDataset(train_sentences_list, train_embs)
    dataloader = ResumableLoader(
        dataset,
        batch_size=4,
        seed=args.ckpt_seed,
        shuffle=True,
        collate_fn=collate_fn,
    )

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt = TrainingCheckpointer(
        args.model_dir,
        prefix="base_lcm",
        fingerprint=fingerprint,
        max_keep=args.max_checkpoints,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
    )
    resume = ckpt.restore(
        ckpt.load(args.resume, map_location=device), model, optimizer
    )
    global_step = resume.global_step
    if resume.resumed:
        print(
            f"Resuming at epoch {resume.start_epoch + 1}/{args.epochs}, "
            f"batch {resume.start_batch}"
        )

    model.train()
    for epoch in range(resume.start_epoch, args.epochs):
        total_loss = 0
        n_batches = 0
        skip = resume.batches_to_skip(epoch)
        for batch_idx, (src, tgt) in tqdm(
            dataloader.epoch(epoch, skip=skip), initial=skip, total=len(dataloader)
        ):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # One causal pass predicts every next concept.
            preds = model.forward_all(src)
            valid = (tgt.abs().sum(dim=-1, keepdim=True) > 0).float()
            loss = (
                ((preds - tgt).pow(2) * valid).sum()
                / valid.sum().clamp(min=1)
                / tgt.shape[-1]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            ckpt.maybe_save(
                model,
                optimizer,
                epoch=epoch,
                batch_in_epoch=batch_idx,
                global_step=global_step,
            )

        avg = total_loss / n_batches if n_batches else float("nan")
        print(f"Epoch {epoch + 1}, Loss: {avg:.4f}")

        ckpt.save_epoch(model, optimizer, epoch=epoch, global_step=global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="lcm_models")
    parser.add_argument(
        "--embed_cache",
        type=str,
        default=None,
        help="Optional path for the cached BLT encodings, so a resumed run "
        "skips re-encoding the corpus.",
    )
    add_resume_args(parser, default_interval_steps=100)
    args = parser.parse_args()
    train_base_lcm(args)
