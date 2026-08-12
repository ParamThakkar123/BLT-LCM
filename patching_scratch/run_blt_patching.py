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
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lcm_scripts"))

from device_utils import report_device
from checkpoint_utils import (  # noqa: E402
    ResumableJsonl,
    ResumableLoader,
    TrainingCheckpointer,
    add_resume_args,
    config_fingerprint,
    seed_everything,
)

# ============================================================
# Constants (from BLT codebase: bytelatent/tokenizers/constants.py)
# ============================================================
OFFSET = 4  # Byte values 0-255 map to token IDs 4-259
VOCAB_SIZE = 260  # 256 bytes + 4 special tokens
DEFAULT_THRESHOLD = 1.335442066192627  # Global constraint theta_g, BLT PatcherArgs
DEFAULT_THRESHOLD_ADD = 0.0  # Approx. monotonicity constraint theta_r (BLT §2.3)


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


def sliding_window_causal_mask(seq_len, window, device):
    """Boolean attend-mask for causal attention limited to `window` positions.

    BLT §4.2 trains the entropy model with sliding window attention of 512
    bytes; unbounded causal attention would let the entropy at position i depend
    on arbitrarily distant context, which is what produces the "entropy drift"
    on repetitive text described in §4.4.

    Returns [seq_len, seq_len] with True where attention is allowed:
    position i attends to j for max(0, i-window+1) <= j <= i.
    """
    idx = torch.arange(seq_len, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)  # i - j
    return (delta >= 0) & (delta < window)


class TransformerBlock(nn.Module):
    def __init__(
        self, dim, n_heads, ffn_dim_multiplier, max_seqlen, rope_theta, attn_window=None
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # None => unbounded causal attention (previous behaviour).
        self.attn_window = attn_window

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

        # SDPA causal attention (no xformers needed). With a finite window we
        # must pass an explicit mask, since is_causal only expresses "attend to
        # everything up to i".
        if self.attn_window is None or self.attn_window >= seqlen:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = sliding_window_causal_mask(seqlen, self.attn_window, x.device)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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

    Reference configuration (BLT §4.2) is 100M parameters, 14 layers, dim 512
    and a 512-byte sliding attention window. The defaults here are a deliberate
    scaled-down variant (dim 256, 4 layers) that trains on a single GPU; per
    BLT Fig. 8, patching quality improves with both entropy-model size and
    context length, with diminishing returns past ~50M parameters. Pass the
    paper's values explicitly to reproduce it.
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
        attn_window=None,
    ):
        super().__init__()
        self.max_length = max_seqlen
        self.vocab_size = vocab_size
        self.dim = dim  # hidden width; used by the BLT local encoder / pooler
        self.attn_window = attn_window

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    ffn_dim_multiplier,
                    max_seqlen,
                    rope_theta,
                    attn_window=attn_window,
                )
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
def compute_entropies_for_tokens(
    tokens_2d, entropy_model, batch_size=1, device="cuda", context_overlap=None
):
    """Per-byte next-byte entropies under the entropy model.

    ``tokens_2d``: [batch, seq_len] byte token IDs.

    Returns [batch, seq_len] where::

        out[b, t] = H( x_{t+1} | x_{<=t} )

    i.e. the entropy of the model's *next-byte* distribution after reading
    position t. Note this is offset by one from the paper's ``H(x_i)``
    (BLT Eq. 1), which is the entropy of the distribution over x_i given
    x_{<i} and therefore lives at ``out[i-1]``. ``entropy_patch_sentence``
    performs that shift; do not compare ``out[i]`` against a threshold
    directly.

    Each row is processed independently. Rows longer than the model's context
    are covered with *overlapping* windows so that every scored position keeps
    at least ``context_overlap`` bytes of real left context — the previous
    implementation flattened the whole batch and re-chunked it into disjoint
    windows, which both destroyed context at window edges and let neighbouring
    rows leak into each other.
    """
    max_length = entropy_model.max_length
    if context_overlap is None:
        context_overlap = max_length // 2
    context_overlap = max(0, min(context_overlap, max_length - 1))

    rows = []
    for row in tokens_2d:
        seq_len = row.numel()
        ent = torch.empty(seq_len, dtype=torch.float32)
        start = 0
        while start < seq_len:
            # Window covers [win_start, win_end); we only *keep* scores for
            # [start, win_end) so every kept position had left context.
            win_start = max(0, start - context_overlap) if start > 0 else 0
            win_end = min(seq_len, win_start + max_length)
            window = row[win_start:win_end].unsqueeze(0).to(device)
            logits = entropy_model(window)[0]
            scores = compute_entropy(logits).cpu()
            keep_from = start - win_start
            ent[start:win_end] = scores[keep_from:]
            if win_end >= seq_len:
                break
            # Everything up to win_end is now scored, so the next window starts
            # there and reaches back ``context_overlap`` bytes for its context;
            # each window therefore contributes ``stride`` new positions.
            #
            # This used to read `win_start + stride if win_start + stride >
            # start else start + 1`. With the default overlap of max_length // 2,
            # win_start is start - overlap and stride is max_length - overlap, so
            # win_start + stride == start exactly -- never greater. The guard
            # fired every time and the window crawled forward ONE BYTE per
            # iteration, running a full max_length forward pass per byte: ~600
            # forwards for an 850-byte sequence instead of 3.
            start = win_end
        rows.append(ent)
    return torch.stack(rows).reshape(tokens_2d.shape)


# ============================================================
# Entropy-based patching (from BLT: bytelatent/data/patcher.py)
# ============================================================
def entropy_patch_sentence(
    entropies_list,
    threshold=DEFAULT_THRESHOLD,
    mode="global",
    threshold_add=DEFAULT_THRESHOLD_ADD,
):
    """Place patch boundaries from per-byte next-byte entropies.

    ``entropies_list`` must be the output of :func:`compute_entropies_for_tokens`,
    i.e. ``entropies_list[t] = H(x_{t+1} | x_{<=t})``. The paper's quantity
    ``H(x_k)`` — the entropy of the distribution over byte k given everything
    before it (BLT Eq. 1) — is therefore ``entropies_list[k-1]``, and that is
    what gets compared against the threshold here. Reading
    ``entropies_list[k]`` instead (as this function previously did) shifts
    every boundary one byte early, so the high-entropy byte lands in the
    *middle* of a patch rather than starting one.

    Two segmentation rules from BLT §2.3 are supported:

    ``mode="global"``    Global constraint: start a patch at k when
                         ``H(x_k) > threshold``.
    ``mode="monotonic"`` Approximate monotonicity constraint: start a patch at
                         k when ``H(x_k) - H(x_{k-1}) > threshold_add``. This
                         detects points that break the locally-decreasing
                         entropy within a patch, and (per §4.4) is less prone
                         to drift when the entropy model's context changes.

    Positions 0 and 1 always start patches, matching BLT's convention that the
    first two byte positions are forced patch starts.

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

    if mode not in ("global", "monotonic"):
        raise ValueError(f"unknown patching mode {mode!r}; expected global|monotonic")

    # From position 2 onward. H(x_k) == entropies_list[k - 1].
    for k in range(2, seq_len):
        h_k = entropies_list[k - 1]
        if mode == "global":
            is_start = h_k > threshold
        else:
            is_start = (h_k - entropies_list[k - 2]) > threshold_add
        if is_start:
            boundaries.append(k)

    # Compute patch lengths from boundaries
    patch_lengths = []
    for i in range(len(boundaries)):
        if i + 1 < len(boundaries):
            patch_lengths.append(boundaries[i + 1] - boundaries[i])
        else:
            patch_lengths.append(seq_len - boundaries[i])

    return boundaries, patch_lengths


def split_on_newlines(tokens):
    """Split a byte-token sequence into segments after each newline byte.

    BLT §4.4 resets the entropy model's context at newlines for the large
    BLT-Entropy run, because repeated structured content (multiple-choice
    options, boilerplate) otherwise drives entropy down and yields runaway
    patch sizes. Scoring each line with a fresh context removes that drift.
    """
    newline = ord("\n") + OFFSET
    segments, current = [], []
    for tok in tokens:
        current.append(tok)
        if tok == newline:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


@torch.no_grad()
def entropies_with_context_reset(tokens, entropy_model, device="cuda"):
    """Next-byte entropies computed with the context reset at each newline.

    Returns a flat list aligned 1:1 with ``tokens`` (BLT §4.4).
    """
    out = []
    for segment in split_on_newlines(tokens):
        t = torch.tensor([segment], dtype=torch.long)
        seg_ent = compute_entropies_for_tokens(t, entropy_model, device=device)[0]
        out.extend(seg_ent.tolist())
    return out


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
    attn_window=None,
    epochs=3,
    lr=3e-4,
    batch_size=32,
    device="cuda",
    resume="auto",
    fingerprint=None,
    ckpt_dir="patching_scratch",
    save_interval_steps=200,
    save_interval_seconds=0.0,
    max_checkpoints=5,
    seed=42,
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
        attn_window=attn_window,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    dataset = ByteTextDataset(marathi_texts, max_len=max_seqlen)
    print(f"  Training chunks: {len(dataset)}")

    loader = ResumableLoader(
        dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # The cosine schedule's position depends on how many steps have been taken,
    # so it rides in the checkpoint alongside the model and optimizer.
    ckpt = TrainingCheckpointer(
        ckpt_dir,
        prefix="entropy_model_marathi",
        suffix=".pt",
        fingerprint=fingerprint,
        max_keep=max_checkpoints,
        save_interval_steps=save_interval_steps,
        save_interval_seconds=save_interval_seconds,
    )
    rp = ckpt.restore(
        ckpt.load(resume, map_location=device), model, optimizer, scheduler
    )
    global_step = rp.global_step
    if rp.resumed:
        print(
            f"  [resume] continuing at epoch {rp.start_epoch + 1}/{epochs}, "
            f"batch {rp.start_batch}"
        )

    model.train()
    for epoch in range(rp.start_epoch, epochs):
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        skip = rp.batches_to_skip(epoch)
        for step, (x, y) in loader.epoch(epoch, skip=skip):
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
            global_step += 1

            ckpt.maybe_save(
                model,
                optimizer,
                scheduler,
                epoch=epoch,
                batch_in_epoch=step,
                global_step=global_step,
            )

            if (step + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(loader)} | "
                    f"Loss: {total_loss/n_batches:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - start
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch+1}/{epochs} DONE | "
            f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )
        ckpt.save_epoch(
            model, optimizer, scheduler, epoch=epoch, global_step=global_step
        )

    model.eval()
    return model


# ============================================================
# Main patching pipeline
# ============================================================
def run_patching(
    marathi_texts,
    entropy_model,
    threshold,
    device="cuda",
    output=None,
    resume="auto",
    fingerprint=None,
    mode="global",
    threshold_add=DEFAULT_THRESHOLD_ADD,
    reset_context_on_newline=False,
):
    """
    Run BLT entropy-based patching on a list of Marathi sentences.

    For each sentence:
      1. Encode text as UTF-8 bytes -> token IDs
      2. Run entropy model to get per-byte entropy
      3. Place patch boundaries where entropy > threshold
      4. Record all patch info

    Records stream straight to ``output`` as JSONL rather than accumulating in
    memory, so an interrupted scan of a large corpus resumes at the first
    sentence it had not yet written instead of restarting from zero. Returns the
    full list of per-sentence results (including any replayed from a prior run).
    """
    entropy_model.eval()
    total_start = time.time()

    writer = ResumableJsonl(
        output or "blt_marathi_patched.jsonl",
        fingerprint=fingerprint,
        resume=resume != "never",
        key="sentence_index",
    )
    if writer.done:
        print(f"  Resuming patching: {len(writer.done)} sentences already written")

    for idx, text in enumerate(marathi_texts):
        if writer.is_done(idx):
            continue
        tokens = text_to_byte_tokens(text)
        if len(tokens) == 0:
            continue

        # Compute per-byte entropies (next-byte convention; see the docstring of
        # compute_entropies_for_tokens for the alignment)
        if reset_context_on_newline:
            entropies_list = entropies_with_context_reset(
                tokens, entropy_model, device=device
            )
        else:
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)
            entropies_list = compute_entropies_for_tokens(
                tokens_tensor, entropy_model, device=device
            )[0].tolist()

        # Find patch boundaries
        boundaries, patch_lengths = entropy_patch_sentence(
            entropies_list, threshold, mode=mode, threshold_add=threshold_add
        )

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
            "patching_mode": mode,
            "threshold_add": threshold_add,
            "entropy_per_byte": [round(e, 6) for e in entropies_list],
            "patch_boundaries": boundaries,
            "patch_lengths": patch_lengths,
            "patches": patches,
        }
        writer.append(result)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - total_start
            print(
                f"  Processed {idx+1}/{len(marathi_texts)} sentences "
                f"({elapsed:.1f}s elapsed)"
            )

    writer.close()
    results = writer.all_records()
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
        help="Global-constraint entropy threshold theta_g for patch boundaries",
    )
    parser.add_argument(
        "--patching_mode",
        choices=["global", "monotonic"],
        default="global",
        help="BLT §2.3 segmentation rule. 'global': H(x_k) > theta_g. "
        "'monotonic': H(x_k) - H(x_k-1) > theta_r (approximate monotonicity).",
    )
    parser.add_argument(
        "--threshold_add",
        type=float,
        default=DEFAULT_THRESHOLD_ADD,
        help="theta_r for --patching_mode monotonic",
    )
    parser.add_argument(
        "--reset_context_on_newline",
        action="store_true",
        help="Recompute entropies with the context reset at each newline "
        "(BLT §4.4), which avoids runaway patch sizes on repetitive text.",
    )
    parser.add_argument(
        "--attn_window",
        type=int,
        default=None,
        help="Sliding-window attention span in bytes for the entropy model "
        "(BLT §4.2 uses 512). Default None = unbounded causal attention.",
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
    add_resume_args(parser, default_interval_steps=200)

    args = parser.parse_args()
    report_device(args.device)
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)

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
            attn_window=cfg.get("attn_window", args.attn_window),
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
            attn_window=args.attn_window,
            epochs=args.train_epochs,
            lr=args.lr,
            batch_size=args.train_batch_size,
            device=args.device,
            resume=args.resume,
            fingerprint=fingerprint,
            ckpt_dir=os.path.dirname(os.path.abspath(args.save_model)) or ".",
            save_interval_steps=args.save_interval_steps,
            save_interval_seconds=args.save_interval_seconds,
            max_checkpoints=args.max_checkpoints,
            seed=args.ckpt_seed,
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
                        "attn_window": args.attn_window,
                    },
                },
                args.save_model,
            )
            print(f"Saved trained model to {args.save_model}")
    else:
        raise ValueError(
            "patch_only mode requires --load_model pointing to a saved checkpoint"
        )

    # ---- Run entropy-based patching (streams straight to args.output) ----
    print(f"\nRunning BLT entropy-based patching (threshold={args.threshold})...")
    results = run_patching(
        marathi_texts,
        model,
        args.threshold,
        device=args.device,
        output=args.output,
        resume=args.resume,
        fingerprint=fingerprint,
    )

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
