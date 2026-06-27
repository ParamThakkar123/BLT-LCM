"""
Data loader for LCM training
Loads pre-segmented sentences and encodes them
"""

import json

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset for next-embedding prediction over precomputed embedding sequences."""

    def __init__(self, embeddings_seqs, min_seq_len=2):
        self.data = [seq for seq in embeddings_seqs if len(seq) >= min_seq_len]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def collate_embeddings(batch):
    src_batch, tgt_batch = zip(*batch)
    max_len = max(src.shape[0] for src in src_batch)
    embed_dim = src_batch[0].shape[1]

    padded_src = torch.zeros(len(src_batch), max_len, embed_dim)
    padded_tgt = torch.zeros(len(tgt_batch), max_len, embed_dim)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, : src.shape[0]] = src
        padded_tgt[i, : tgt.shape[0]] = tgt

    return padded_src, padded_tgt


# Backward-compatible alias for existing training scripts/tests.
collate = collate_embeddings


class LCMDataset(Dataset):
    def __init__(self, jsonl_file, sonar_loader, max_seq_len=128):
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                sentences = [
                    s["marathi_text"] for s in item.get("sentences", [])
                ]  # Assume sentences key
                if len(sentences) > 1:
                    self.data.append(sentences[:max_seq_len])

        self.sonar_loader = sonar_loader
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentences = self.data[idx]
        embeddings = self.sonar_loader.encode_sentences(sentences)
        return embeddings[:-1], embeddings[1:]  # src, tgt


def collate_fn(batch):
    return collate_embeddings(batch)
