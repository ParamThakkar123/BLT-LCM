"""
BLT Entropy-Based Patching for Marathi Sentences
=================================================
Implements the entropy-based patching algorithm from the BLT paper
(Byte Latent Transformer, Meta 2024) on Marathi text from BhashaSetu.

Standalone implementation using pure PyTorch (no xformers dependency).

Two modes:
  1. train_and_patch: Train a small byte-level entropy model from scratch, then patch
  2. patch_only: Load a previously trained model and patch

Usage:
    python run_blt_patching.py --num_sentences 10000 --train_epochs 3
    python run_blt_patching.py --load_model entropy_model_marathi.pt --mode patch_only
"""

import argparse
import json
import math
import os
import sys
import time
from datasets import load_dataset
from tqdm import tqdm

# Fix Windows console encoding for Marathi/Devanagari output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Constants (from BLT codebase: bytelatent/tokenizers/constants.py)
# ============================================================
OFFSET = 4  # Byte values 0-255 map to token IDs 4-259
VOCAB_SIZE = 260  # 256 bytes + 4 special tokens
DEFAULT_THRESHOLD = 1.335442066192627  # Default from BLT PatcherArgs


# ============================================================
# Byte encoding utilities
# ============================================================
def text_to_byte_tokens(text: str) -> list[int]:
    """Convert text to BLT byte-level token IDs (UTF-8 bytes + OFFSET)."""
    return [b + OFFSET for b in text.encode("utf-8")]


def byte_tokens_to_text(tokens: list[int]) -> str:
    """Convert BLT byte-level token IDs back to text."""
    byte_values = [t - OFFSET for t in tokens if OFFSET <= t < OFFSET + 256]
    return bytes(byte_values).decode("utf-8", errors="replace")


# ============================================================
# Model components (mirrors BLT's LMTransformer but with SDPA)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seqlen=512, theta=10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seqlen)
        emb = torch.outer(t, freqs)
        # Store cos/sin: [1, 1, max_seqlen, head_dim//2]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len):
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings. q,k: [batch, heads, seq_len, head_dim]"""
    d = q.shape[-1] // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim_multiplier, max_seqlen, rope_theta):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

        # Attention projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # FFN (SwiGLU)
        hidden_dim = int(2 * 4 * dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.rotary = RotaryEmbedding(self.head_dim, max_seqlen, rope_theta)

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        # Self-attention with RoPE
        h = self.attention_norm(x)
        q = self.wq(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seqlen)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # SDPA causal attention (no xformers needed)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        x = x + self.wo(attn_out)

        # SwiGLU FFN
        h = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class ByteEntropyModel(nn.Module):
    """
    Small byte-level causal transformer for next-byte prediction.
    Used to compute per-byte entropy for BLT patching.
    """

    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        dim=256,
        n_heads=4,
        n_layers=4,
        max_seqlen=512,
        ffn_dim_multiplier=1.3,
        rope_theta=10000.0,
    ):
        super().__init__()
        self.max_length = max_seqlen
        self.vocab_size = vocab_size
        self.dim = dim  # hidden width; used by the BLT local encoder / pooler

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(dim, n_heads, ffn_dim_multiplier, max_seqlen, rope_theta)
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens, return_hidden=False):
        """Run the byte transformer.

        By default returns next-byte logits (used only to compute entropy for
        patch-boundary placement). With ``return_hidden=True`` it returns the
        final normalized byte hidden states ``[B, T, dim]`` instead — these are
        the local-encoder features that BLT pools into patch representations
        (BLT §3.2.2). The logits are never used as patch/concept embeddings.
        """
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if return_hidden:
            return h
        return self.output(h)


# ============================================================
# Entropy computation (from BLT: bytelatent/data/patcher.py)
# ============================================================
def compute_entropy(logits):
    """
    Compute Shannon entropy from logits.
    logits: [..., vocab_size]
    Returns: [...] entropy in nats
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


@torch.no_grad()
def compute_entropies_for_tokens(tokens_2d, entropy_model, batch_size=1, device="cuda"):
    """
    Compute per-byte entropies using the entropy model.
    tokens_2d: [batch, seq_len] tensor of byte token IDs
    Returns: [batch, seq_len] tensor of entropy values
    """
    all_entropies = []
    max_length = entropy_model.max_length
    batch_numel = max_length * batch_size
    flat = tokens_2d.flatten()
    splits = torch.split(flat, batch_numel)

    for split in splits:
        actual_len = split.numel()
        pad_size = (max_length - (actual_len % max_length)) % max_length
        if pad_size > 0:
            pad = torch.zeros(pad_size, dtype=split.dtype, device=split.device)
            split = torch.cat([split, pad])
        split = split.reshape(-1, max_length).to(device)

        logits = entropy_model(split)
        # Flatten and remove padding
        logits = logits.reshape(-1, logits.shape[-1])[:actual_len]
        ent = compute_entropy(logits)
        all_entropies.append(ent.cpu())

    return torch.cat(all_entropies).reshape(tokens_2d.shape)


# ============================================================
# Entropy-based patching (from BLT: bytelatent/data/patcher.py)
# ============================================================
def entropy_patch_sentence(entropies_list, threshold=DEFAULT_THRESHOLD):
    """
    Given per-byte entropy values for a single sentence,
    find patch boundaries where entropy > threshold.

    Follows BLT convention: positions 0 and 1 are always patch starts,
    then any position i >= 2 where entropy[i] > threshold starts a new patch.

    Returns:
        boundaries: list of byte indices where patches start
        patch_lengths: list of byte-lengths per patch
    """
    seq_len = len(entropies_list)
    if seq_len == 0:
        return [], []

    # BLT convention: first two positions always start patches
    boundaries = [0]
    if seq_len > 1:
        boundaries.append(1)

    # From position 2 onward, start new patch where entropy > threshold
    # (BLT skips position 0's entropy, checks positions 1..end after the first two forced starts)
    for i in range(2, seq_len):
        if entropies_list[i] > threshold:
            boundaries.append(i)

    # Compute patch lengths from boundaries
    patch_lengths = []
    for i in range(len(boundaries)):
        if i + 1 < len(boundaries):
            patch_lengths.append(boundaries[i + 1] - boundaries[i])
        else:
            patch_lengths.append(seq_len - boundaries[i])

    return boundaries, patch_lengths


# ============================================================
# Training dataset
# ============================================================
class ByteTextDataset(Dataset):
    """Chunks Marathi text into byte sequences for next-byte prediction training."""

    def __init__(self, texts, max_len=512):
        self.samples = []
        for text in texts:
            tokens = text_to_byte_tokens(text)
            # Need at least 2 tokens for (input, target) pair
            if len(tokens) < 2:
                continue
            # Chunk long sequences
            for i in range(0, len(tokens), max_len):
                chunk = tokens[i : i + max_len + 1]  # +1 for target
                if len(chunk) >= 2:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    xs, ys = zip(*batch)
    max_len = max(len(x) for x in xs)
    x_padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    y_padded = torch.full((len(ys), max_len), -100, dtype=torch.long)  # -100 = ignore
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_padded[i, : len(x)] = x
        y_padded[i, : len(y)] = y
    return x_padded, y_padded


# ============================================================
# Training loop
# ============================================================
def train_entropy_model(
    marathi_texts,
    vocab_size=VOCAB_SIZE,
    dim=256,
    n_heads=4,
    n_layers=4,
    max_seqlen=512,
    ffn_dim_multiplier=1.3,
    epochs=3,
    lr=3e-4,
    batch_size=32,
    device="cuda",
):
    print(f"\n{'='*60}")
    print(f"Training Byte-Level Entropy Model")
    print(f"  Architecture: {n_layers} layers, dim={dim}, heads={n_heads}")
    print(f"  Max sequence length: {max_seqlen}")
    print(f"  Training on: {len(marathi_texts)} Marathi sentences")
    print(f"{'='*60}\n")

    torch.set_default_dtype(torch.float32)
    model = ByteEntropyModel(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seqlen=max_seqlen,
        ffn_dim_multiplier=ffn_dim_multiplier,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    dataset = ByteTextDataset(marathi_texts, max_len=max_seqlen)
    print(f"  Training chunks: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(loader)} | "
                    f"Loss: {total_loss/n_batches:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - start
        avg_loss = total_loss / n_batches
        print(
            f"  Epoch {epoch+1}/{epochs} DONE | "
            f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )

    model.eval()
    return model


# ============================================================
# Main patching pipeline
# ============================================================
def run_patching(marathi_texts, entropy_model, threshold, device="cuda"):
    """
    Run BLT entropy-based patching on a list of Marathi sentences.

    For each sentence:
      1. Encode text as UTF-8 bytes -> token IDs
      2. Run entropy model to get per-byte entropy
      3. Place patch boundaries where entropy > threshold
      4. Record all patch info

    Returns list of dicts (one per sentence) with full patching results.
    """
    results = []
    entropy_model.eval()
    total_start = time.time()

    for idx, text in enumerate(marathi_texts):
        tokens = text_to_byte_tokens(text)
        if len(tokens) == 0:
            continue

        # Compute per-byte entropies
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)
        entropies = compute_entropies_for_tokens(
            tokens_tensor, entropy_model, batch_size=1, device=device
        )
        entropies_list = entropies[0].tolist()

        # Find patch boundaries
        boundaries, patch_lengths = entropy_patch_sentence(entropies_list, threshold)

        # Build detailed patch records
        patches = []
        for i, (start, length) in enumerate(zip(boundaries, patch_lengths)):
            end = start + length
            patch_byte_tokens = tokens[start:end]
            patch_text = byte_tokens_to_text(patch_byte_tokens)
            patch_entropies = entropies_list[start:end]

            patches.append(
                {
                    "patch_index": i,
                    "byte_start": start,
                    "byte_end": end,
                    "length_bytes": length,
                    "patch_text": patch_text,
                    "entropy_at_boundary": round(entropies_list[start], 6),
                    "mean_entropy_in_patch": round(
                        sum(patch_entropies) / len(patch_entropies), 6
                    ),
                }
            )

        result = {
            "sentence_index": idx,
            "marathi_text": text,
            "num_bytes": len(tokens),
            "num_patches": len(patches),
            "avg_patch_length": round(len(tokens) / max(len(patches), 1), 2),
            "threshold_used": threshold,
            "entropy_per_byte": [round(e, 6) for e in entropies_list],
            "patch_boundaries": boundaries,
            "patch_lengths": patch_lengths,
            "patches": patches,
        }
        results.append(result)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - total_start
            print(
                f"  Processed {idx+1}/{len(marathi_texts)} sentences "
                f"({elapsed:.1f}s elapsed)"
            )

    elapsed = time.time() - total_start
    print(f"  Patching complete: {len(results)} sentences in {elapsed:.1f}s")
    return results


# ============================================================
# Entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="BLT Entropy-Based Patching on Marathi (BhashaSetu)"
    )
    parser.add_argument(
        "--mode",
        choices=["train_and_patch", "patch_only"],
        default="train_and_patch",
        help="train_and_patch: train from scratch then patch. patch_only: load saved model.",
    )
    parser.add_argument(
        "--num_sentences", type=int, default=10000, help="Number of Marathi sentences"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Entropy threshold for patch boundaries",
    )

    # Model architecture
    parser.add_argument("--dim", type=int, default=256, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--max_seqlen", type=int, default=512, help="Max sequence length")
    parser.add_argument("--ffn_dim_multiplier", type=float, default=1.3)

    # Training
    parser.add_argument("--train_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    # I/O
    parser.add_argument(
        "--output", type=str, default="blt_marathi_patched.jsonl", help="Output JSONL path"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="entropy_model_marathi.pt",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load a previously trained model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    # ---- Load dataset (streaming to avoid downloading entire 7.8GB) ----
    print("Loading BhashaSetu dataset from HuggingFace (streaming)...")

    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    marathi_texts = []
    for row in tqdm(ds, desc="Loading dataset", unit="rows", total=args.num_sentences):
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            marathi_texts.append(text)
        if len(marathi_texts) >= args.num_sentences:
            break
    print(f"Selected {len(marathi_texts)} Marathi sentences")

    # ---- Build / load entropy model ----
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading entropy model from {args.load_model}...")
        checkpoint = torch.load(args.load_model, map_location=args.device)
        # Load config from checkpoint if available
        cfg = checkpoint.get("config", {})
        model = ByteEntropyModel(
            vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
            dim=cfg.get("dim", args.dim),
            n_heads=cfg.get("n_heads", args.n_heads),
            n_layers=cfg.get("n_layers", args.n_layers),
            max_seqlen=cfg.get("max_seqlen", args.max_seqlen),
            ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", args.ffn_dim_multiplier),
        ).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Model loaded successfully.")

    elif args.mode == "train_and_patch":
        model = train_entropy_model(
            marathi_texts,
            vocab_size=VOCAB_SIZE,
            dim=args.dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            max_seqlen=args.max_seqlen,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            epochs=args.train_epochs,
            lr=args.lr,
            batch_size=args.train_batch_size,
            device=args.device,
        )
        # Save model with config for easy reloading
        if args.save_model:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": VOCAB_SIZE,
                        "dim": args.dim,
                        "n_heads": args.n_heads,
                        "n_layers": args.n_layers,
                        "max_seqlen": args.max_seqlen,
                        "ffn_dim_multiplier": args.ffn_dim_multiplier,
                    },
                },
                args.save_model,
            )
            print(f"Saved trained model to {args.save_model}")
    else:
        raise ValueError(
            "patch_only mode requires --load_model pointing to a saved checkpoint"
        )

    # ---- Run entropy-based patching ----
    print(f"\nRunning BLT entropy-based patching (threshold={args.threshold})...")
    results = run_patching(
        marathi_texts, model, args.threshold, device=args.device
    )

    # ---- Save JSONL output ----
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} patched sentences to {args.output}")

    # ---- Summary statistics ----
    total_patches = sum(r["num_patches"] for r in results)
    total_bytes = sum(r["num_bytes"] for r in results)
    avg_patches_per_sent = total_patches / len(results)
    avg_patch_len = total_bytes / total_patches if total_patches > 0 else 0

    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"  Sentences processed : {len(results)}")
    print(f"  Total bytes         : {total_bytes:,}")
    print(f"  Total patches       : {total_patches:,}")
    print(f"  Avg patches/sentence: {avg_patches_per_sent:.1f}")
    print(f"  Avg bytes/patch     : {avg_patch_len:.1f}")

    # Show a few example patches
    print(f"\n{'='*60}")
    print(f"Example Patches (first 3 sentences)")
    print(f"{'='*60}")
    for ex in results[:3]:
        print(f"\n  Sentence {ex['sentence_index']}: \"{ex['marathi_text'][:60]}...\"")
        print(f"  Bytes: {ex['num_bytes']} | Patches: {ex['num_patches']} | Avg len: {ex['avg_patch_length']}")
        for p in ex["patches"][:6]:
            print(
                f"    [{p['patch_index']:2d}] bytes[{p['byte_start']:3d}:{p['byte_end']:3d}] "
                f"len={p['length_bytes']:3d} "
                f"entropy={p['entropy_at_boundary']:.3f} "
                f"\"{p['patch_text'][:40]}\""
            )
        if len(ex["patches"]) > 6:
            print(f"    ... ({len(ex['patches']) - 6} more patches)")


if __name__ == "__main__":
    main()
