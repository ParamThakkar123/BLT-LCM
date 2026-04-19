"""
Evaluate BLT-LCM model on the test split (last 20% of BhashaSetu dataset) for Machine Translation tasks.

Computes the average MSE loss on next sentence prediction and evaluates generated hypotheses against references using text metrics.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import re
from blt_loader import BLTLoader
from base_lcm import BaseLCM
import os
from run_blt_patching import text_to_byte_tokens
from tqdm import tqdm


def prepare_data(is_test=False, max_sent_per_doc=20):
    ds = load_dataset("ParamTh/BhashaSetu", split="train")
    total = len(ds)
    if is_test:
        test_start = int(total * 0.8)  # last 20%
        ds = ds.select(range(test_start, total))
    docs = []
    for row in tqdm(ds, desc="Loading data"):
        text = row.get("marathi", "")
        if text and len(text.strip()) > 0:
            sents = [s.strip() for s in re.split(r"[.।]", text) if s.strip()]
            if len(sents) >= 2:
                docs.append(sents[:max_sent_per_doc])
    return docs


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_seqs):
        self.data = [seq for seq in embeddings_seqs if len(seq) >= 2]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load BLT loader
    entropy_model_path = "patching_scratch/entropy_model_marathi.pt"
    blt = BLTLoader(entropy_model_path=entropy_model_path, device=str(device))

    # Prepare training data for closest sentence lookup (use 10% to save time)
    print("Preparing training data for lookup...")
    train_docs = prepare_data(is_test=False)
    train_sentences = []
    for doc in train_docs[: int(len(train_docs) * 0.1)]:  # 10%
        train_sentences.extend(doc)
    print(f"Loaded {len(train_sentences)} training sentences")

    # Encode training sentences
    print("Encoding training sentences...")
    tokenized_train = [
        text_to_byte_tokens(sent) for sent in tqdm(train_sentences, desc="Tokenizing")
    ]
    train_embs = blt.encode_tokens_batch(tokenized_train)
    train_embs = torch.stack(train_embs).to(device)

    # Prepare test data
    print("Preparing test data...")
    test_docs = prepare_data(is_test=True)
    print(f"Loaded {len(test_docs)} test documents")

    # Encode test data
    print("Encoding test data with BLT...")
    flat_sents = []
    doc_indices = []
    for i, sents in enumerate(test_docs):
        for sent in sents:
            flat_sents.append(sent)
            doc_indices.append(i)
    tokenized_batch = [
        text_to_byte_tokens(sent) for sent in tqdm(flat_sents, desc="Tokenizing test")
    ]
    embed_list = blt.encode_tokens_batch(tokenized_batch)

    # Reconstruct per-document sequences
    embeddings_seqs = [[] for _ in range(len(test_docs))]
    for emb, didx in zip(embed_list, doc_indices):
        embeddings_seqs[didx].append(emb)

    # stack per-document tensors
    for i in range(len(embeddings_seqs)):
        if len(embeddings_seqs[i]) == 0:
            embeddings_seqs[i] = torch.empty((0, blt.model.dim))
        else:
            embeddings_seqs[i] = torch.stack(embeddings_seqs[i], dim=0)

    # Create dataset and dataloader
    dataset = EmbeddingDataset(embeddings_seqs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate)

    # Load model
    checkpoint_path = "lcm_models/lcm_blt_best.pth"
    model = BaseLCM(
        embed_dim=embeddings_seqs[0].shape[1], model_dim=2048, n_layers=12, n_heads=16
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    mse = torch.nn.MSELoss()
    total_loss = 0.0
    n = 0
    hyps = []
    refs = []

    print("Evaluating model on test set...")
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            seq_len = tgt.shape[1]
            if seq_len == 0:
                continue
            for i in range(seq_len):
                prefix = src[:, : i + 1] if i + 1 <= src.shape[1] else src
                pred = model(prefix, tgt[:, i : i + 1])
                total_loss += mse(pred, tgt[:, i]).item()
                n += 1

                # Find closest training sentence for each in batch
                pred_emb = pred.squeeze(1)  # [B, E]
                for b in range(pred_emb.shape[0]):
                    p_emb = pred_emb[b]  # [E]
                    similarities = torch.cosine_similarity(
                        p_emb.unsqueeze(0), train_embs, dim=1
                    )
                    best_idx = torch.argmax(similarities).item()
                    hyp = train_sentences[best_idx]
                    hyps.append(hyp)

    # Collect refs: for each test doc, the next sentences
    for doc in test_docs:
        refs.extend(doc[1:])  # Skip first, as no prediction for it
    refs = refs[: len(hyps)]  # Trim if necessary

    avg_loss = total_loss / n if n > 0 else float("inf")
    print(f"Test MSE Loss: {avg_loss:.4f}")

    # Save hyp and ref
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/hyp.txt", "w", encoding="utf-8") as f:
        for h in hyps:
            f.write(h + "\n")
    with open("outputs/ref.txt", "w", encoding="utf-8") as f:
        for r in refs:
            f.write(r + "\n")

    # Run eval_runner
    os.makedirs("results", exist_ok=True)
    os.system(
        "python lcm_scripts/eval_runner.py --hyp_file outputs/hyp.txt --ref_file outputs/ref.txt --out_csv results/mt_eval_results.csv --comet_model wmt22-comet-da"
    )

    # Save results
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Test MSE Loss: {avg_loss:.4f}\n")
        f.write(f"Number of predictions: {n}\n")


if __name__ == "__main__":
    main()
