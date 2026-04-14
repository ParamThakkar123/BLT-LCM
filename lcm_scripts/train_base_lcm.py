"""
Training script for Base-LCM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from base_lcm import BaseLCM
from data_loader import LCMDataset, collate_fn
from blt_loader import BLTLoader


def train_base_lcm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BLT
    blt_loader = BLTLoader(
        entropy_model_path="../patching_scratch/entropy_model_marathi.pt",
        device=str(device),
    )

    # Create model
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

    # Segment into sentences (simple split by .)
    sentences_list = []
    for text in marathi_texts[:100]:
        sents = [s.strip() for s in text.split(".") if s.strip()]
        if len(sents) > 1:
            sentences_list.append(sents[:10])  # Max 10 sentences

    # Encode all sentences
    all_sentences = [s for seq in sentences_list for s in seq]
    all_embeddings = blt_loader.encode_sentences(all_sentences)

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

    dataset = RealDataset(sentences_list, all_embeddings)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            # For each position, predict next
            losses = []
            for i in range(tgt.shape[1]):
                pred = model(
                    src[:, : i + 1] if i > 0 else src[:, :1], tgt[:, i : i + 1]
                )
                loss = criterion(pred, tgt[:, i])
                losses.append(loss)

            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

        # Save model
        torch.save(model.state_dict(), f"lcm_models/base_lcm_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train_base_lcm(args)
