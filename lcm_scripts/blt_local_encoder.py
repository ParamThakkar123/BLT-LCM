"""BLT local encoder, hash n-gram embeddings and latent transformer.

Implements the byte-side of the Byte Latent Transformer (Pagnoni et al., 2024):

``HashNGramEmbedder``     BLT §3.2.1 / Eqs. (2)-(4) and Appendix C. Augments each
                          byte embedding with rolling-polynomial-hashed n-gram
                          embeddings for n in 3..8.
``PatchCrossAttentionPooler``
                          BLT §3.2.2. Perceiver-style cross-attention that pools
                          the byte hidden states of one patch into a single
                          patch representation, with patch masking so a query
                          only sees its own bytes.
``BLTLocalEncoder``       BLT §3.2. A *learnable* encoder E with l_E transformer
                          layers using local block-causal attention, each
                          followed by a cross-attention pooling step. This is a
                          separate network from the entropy model, which BLT
                          uses only to place patch boundaries.
``BLTLatentTransformer``  BLT §3.1. The global model over the sequence of patch
                          representations, with block-causal attention.
``BLTSentenceEncoder``    Composes the three into ``text -> concept vector``,
                          which is the interface the LCM consumes.

The entropy model stays frozen and is used solely for boundary placement; every
module here is trainable.
"""

from typing import List, NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A 10-digit prime, per BLT Appendix C.
HASH_BASE_PRIME = 1000000007

# BLT §4.8 uses 500,000 hashes with a single hash function over n = 3..8.
PAPER_NGRAM_SIZES = (3, 4, 5, 6, 7, 8)
PAPER_HASH_VOCAB = 500_000

# That configuration is very large for a single accelerator: at encoder width
# 256 it is 6 x 500_000 x 256 x 4B = 2.9 GiB of *parameters*, and AdamW keeps a
# gradient plus two moments, so ~11.4 GiB is committed before any activation --
# enough to OOM a 16 GiB GPU during optimizer.step().
#
# BLT Table 8 shows the per-n vocabulary matters more than covering every n, and
# that the smaller n are the more impactful ones ("3,4,5 @ 100k" scores 0.837 on
# the train distribution vs 0.826 for the full "3..8 @ 400k"). The default here
# is therefore the memory-feasible subset; pass PAPER_NGRAM_SIZES /
# PAPER_HASH_VOCAB explicitly to reproduce the paper's configuration on hardware
# that can hold it.
DEFAULT_NGRAM_SIZES = (3, 4, 5)
DEFAULT_HASH_VOCAB = 100_000


def hash_embedding_bytes(
    dim: int,
    ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
    hash_vocab_size: int = DEFAULT_HASH_VOCAB,
    bytes_per_param: int = 4,
    optimizer_multiplier: int = 4,
) -> int:
    """Bytes the hash tables commit, including optimizer state.

    ``optimizer_multiplier=4`` accounts for parameters + gradient + AdamW's two
    moments. Use this to sanity-check a configuration before training rather
    than discovering it at the first optimizer.step().
    """
    params = len(tuple(ngram_sizes)) * hash_vocab_size * dim * bytes_per_param
    return params * optimizer_multiplier


def rolling_poly_hash(ngram: Sequence[int], base: int = HASH_BASE_PRIME) -> int:
    """BLT Appendix C, Eq. (23): sum_j b_{i-j+1} * a^{j-1}.

    ``ngram`` is ordered oldest-first, so ``ngram[-1]`` is byte b_i and gets
    exponent 0.
    """
    h = 0
    for j, byte in enumerate(reversed(ngram)):
        h += byte * pow(base, j, 2**61 - 1)
    return h


def _rolling_poly_hash_tensor(
    tokens: torch.Tensor, n: int, base: int = HASH_BASE_PRIME
) -> torch.Tensor:
    """Vectorised rolling polynomial hash for every position of a batch.

    ``tokens``: [B, T] byte token ids. Returns [B, T] hash values. Positions
    i < n-1 have fewer than n preceding bytes; BLT omits byte-grams of size n
    when i < n, so those positions are marked with -1 and skipped by the caller.
    """
    B, T = tokens.shape
    device = tokens.device
    # Modulus keeps the accumulator inside int64 while staying collision-sparse.
    mod = 2**61 - 1
    powers = torch.tensor(
        [pow(base, j, mod) for j in range(n)], dtype=torch.int64, device=device
    )
    # Window the sequence: windows[b, i, j] = tokens[b, i-n+1+j]
    padded = F.pad(tokens, (n - 1, 0), value=0)
    windows = padded.unfold(dimension=1, size=n, step=1)  # [B, T, n]
    # powers[j] must multiply the byte at offset (n-1-j) from the window start,
    # so that the most recent byte b_i carries exponent 0.
    weights = powers.flip(0)
    h = (windows.to(torch.int64) * weights).sum(dim=-1) % mod
    if n > 1:
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = h.masked_fill(positions < (n - 1), -1)
    return h


class HashNGramEmbedder(nn.Module):
    """Hash n-gram embeddings (BLT §3.2.1).

    For each position i and each n, the byte n-gram g_{i,n} = {b_{i-n+1}..b_i}
    is hashed into a per-n embedding table, and the results are summed onto the
    byte embedding::

        e_i = x_i + sum_{n=3..8} E^hash_n( Hash(g_{i,n}) )

    then normalised by the number of n-gram sizes plus one, per Eq. (3).

    BLT Table 8 shows this is the single largest ablation effect in the
    architecture, and §7 calls the hash embeddings "vital to bringing the
    performance of BLT to match those of tokenizer based models".
    """

    def __init__(
        self,
        dim: int,
        ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
        hash_vocab_size: int = DEFAULT_HASH_VOCAB,
        base: int = HASH_BASE_PRIME,
    ):
        super().__init__()
        self.dim = dim
        self.ngram_sizes = tuple(ngram_sizes)
        self.hash_vocab_size = hash_vocab_size
        self.base = base
        self.tables = nn.ModuleDict(
            {
                str(n): nn.Embedding(hash_vocab_size, dim)
                for n in self.ngram_sizes
            }
        )
        for n in self.ngram_sizes:
            nn.init.normal_(self.tables[str(n)].weight, std=0.02)

    @property
    def param_bytes(self) -> int:
        """Parameter bytes in the hash tables (excluding optimizer state)."""
        return len(self.ngram_sizes) * self.hash_vocab_size * self.dim * 4

    def memory_summary(self) -> str:
        gib = 1024**3
        return (
            f"hash n-grams n={list(self.ngram_sizes)} x {self.hash_vocab_size:,} "
            f"@ dim {self.dim}: {self.param_bytes / gib:.2f} GiB params, "
            f"~{self.param_bytes * 4 / gib:.2f} GiB with AdamW state"
        )

    def forward(self, tokens: torch.Tensor, byte_emb: torch.Tensor) -> torch.Tensor:
        """``tokens``: [B, T] ids. ``byte_emb``: [B, T, dim]. Returns [B, T, dim]."""
        out = byte_emb
        for n in self.ngram_sizes:
            h = _rolling_poly_hash_tensor(tokens, n, self.base)  # [B, T], -1 = skip
            valid = h >= 0
            idx = torch.where(valid, h % self.hash_vocab_size, torch.zeros_like(h))
            emb = self.tables[str(n)](idx)
            out = out + emb * valid.unsqueeze(-1).to(emb.dtype)
        # Eq. (3): normalise by the number of n-gram sizes plus one (the byte
        # embedding itself), so the scale does not grow with |ngram_sizes|.
        return out / (len(self.ngram_sizes) + 1)


class PatchCrossAttentionPooler(nn.Module):
    """Perceiver-style cross-attention pooling of bytes into a patch vector.

    Forward inputs:
        patch_hidden: [P, L, dim] byte hidden states for P padded patches.
        key_mask:     [P, L] bool, True for real bytes, False for padding.

    Returns:
        patch_reps:   [P, concept_dim] one representation per patch.

    The single query per patch is initialized from the masked mean of that
    patch's byte hidden states (BLT initializes the patch query by pooling the
    patch's byte embeddings), then cross-attends over the patch's bytes only.
    Padding is masked out of the attention, so patch length never leaks into
    the pooled vector.
    """

    def __init__(self, dim: int, concept_dim: Optional[int] = None, n_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            # Fall back to single head if dim is not divisible (e.g. 260).
            n_heads = 1
        self.dim = dim
        self.concept_dim = concept_dim or dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, self.concept_dim, bias=False)

        self._init_near_mean()

    def _init_near_mean(self):
        """Initialize so the untrained pooler is a sensible masked pool."""
        nn.init.normal_(self.q_proj.weight, std=1e-3)
        nn.init.normal_(self.k_proj.weight, std=1e-3)
        nn.init.eye_(self.v_proj.weight)
        if self.concept_dim == self.dim:
            nn.init.eye_(self.out_proj.weight)
        else:
            nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, patch_hidden: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        P, L, _ = patch_hidden.shape
        h = self.norm(patch_hidden)  # [P, L, dim]

        mask_f = key_mask.float().unsqueeze(-1)  # [P, L, 1]
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # [P, 1]
        q_init = (h * mask_f).sum(dim=1) / denom  # [P, dim] masked-mean query

        q = self.q_proj(q_init).view(P, self.n_heads, self.head_dim)
        k = self.k_proj(h).view(P, L, self.n_heads, self.head_dim)
        v = self.v_proj(h).view(P, L, self.n_heads, self.head_dim)

        # scores: [P, n_heads, L]
        scores = torch.einsum("phd,plhd->phl", q, k) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~key_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # guard rows that are fully masked

        ctx = torch.einsum("phl,plhd->phd", attn, v).reshape(P, self.dim)
        return self.out_proj(ctx)


class PatchIndex(NamedTuple):
    """Flat gather plan for every patch in a batch of sentences.

    Patch boundaries are ragged, which is why the encoder used to run one
    sentence at a time. Precomputing the gather as flat index tensors lets a
    whole batch of sentences be pooled in a single cross-attention call:

        owner[p]      row of the byte batch patch ``p`` belongs to
        slot[p]       position of patch ``p`` within its own sentence
        byte_idx[p]   byte offsets to gather for patch ``p`` (0 where padded)
        key_mask[p]   True at the real bytes of patch ``p``
        max_patches   longest patch sequence in the batch

    ``owner``/``slot`` also scatter the pooled patches back into the padded
    ``[B, max_patches, concept_dim]`` sequence the latent transformer consumes.
    """

    owner: torch.Tensor
    slot: torch.Tensor
    byte_idx: torch.Tensor
    key_mask: torch.Tensor
    max_patches: int


def build_patch_index(patch_specs, device) -> PatchIndex:
    """Build a :class:`PatchIndex` from per-sentence ``(boundaries, lengths)``.

    Entirely vectorised: the per-patch work is numpy arithmetic and the result
    reaches the GPU in four transfers, regardless of how many patches there are.
    """
    counts = np.array([len(b) for b, _ in patch_specs], dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        empty_i = torch.zeros(0, dtype=torch.long, device=device)
        return PatchIndex(
            owner=empty_i,
            slot=empty_i,
            byte_idx=torch.zeros(0, 0, dtype=torch.long, device=device),
            key_mask=torch.zeros(0, 0, dtype=torch.bool, device=device),
            max_patches=0,
        )

    starts = np.concatenate(
        [np.asarray(b, dtype=np.int64) for b, _ in patch_specs if len(b)]
    )
    lens = np.concatenate(
        [np.asarray(l, dtype=np.int64) for _, l in patch_specs if len(l)]
    )
    owner = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    # Position of each patch inside its own sentence: a global arange minus the
    # start offset of the sentence that owns it.
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]])
    slot = np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)

    max_l = int(lens.max())
    ar = np.arange(max_l, dtype=np.int64)[None, :]
    key_mask = ar < lens[:, None]
    byte_idx = np.where(key_mask, starts[:, None] + ar, 0)

    return PatchIndex(
        owner=torch.from_numpy(owner).to(device),
        slot=torch.from_numpy(slot).to(device),
        byte_idx=torch.from_numpy(byte_idx).to(device),
        key_mask=torch.from_numpy(key_mask).to(device),
        max_patches=int(counts.max()),
    )


def local_block_causal_mask(seq_len: int, window: int, device) -> torch.Tensor:
    """Causal mask restricted to a fixed window of preceding bytes (BLT §3.2).

    "each byte attends to a fixed window of w_E preceding bytes that in general
    can cross the dynamic patch boundaries but can not cross document
    boundaries". Returns [seq_len, seq_len], True where attention is allowed.
    """
    idx = torch.arange(seq_len, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)
    return (delta >= 0) & (delta < window)


class _TransformerLayer(nn.Module):
    """Pre-norm transformer layer with masked self-attention and SwiGLU FFN."""

    def __init__(self, dim: int, n_heads: int, ffn_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * ffn_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        h = self.attn_norm(x)
        # nn.MultiheadAttention takes True = "not allowed to attend".
        mask = ~attn_mask if attn_mask is not None else None
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.dropout(a)
        h = self.ffn_norm(x)
        return x + self.dropout(self.w2(F.silu(self.w1(h)) * self.w3(h)))


class BLTLocalEncoder(nn.Module):
    """Learnable byte-level local encoder E (BLT §3.2).

    Byte embeddings are augmented with hash n-gram embeddings, contextualised by
    ``n_layers`` transformer layers under a local block-causal mask, then pooled
    per patch by cross-attention.

    This is a separate, trainable network. The entropy model is *not* reused as
    the backbone: BLT uses it only to decide where patches begin.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        dim: int = 256,
        n_layers: int = 1,
        n_heads: int = 8,
        concept_dim: int = 1024,
        window: int = 512,
        ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
        hash_vocab_size: int = DEFAULT_HASH_VOCAB,
        use_hash_ngrams: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.concept_dim = concept_dim
        self.window = window
        self.byte_emb = nn.Embedding(vocab_size, dim)
        self.hash_ngrams = (
            HashNGramEmbedder(dim, ngram_sizes, hash_vocab_size)
            if use_hash_ngrams
            else None
        )
        self.layers = nn.ModuleList(
            [_TransformerLayer(dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.pooler = PatchCrossAttentionPooler(dim, concept_dim=concept_dim, n_heads=n_heads)

    def byte_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens``: [B, T] -> contextual byte hidden states [B, T, dim]."""
        x = self.byte_emb(tokens)
        if self.hash_ngrams is not None:
            x = self.hash_ngrams(tokens, x)
        mask = local_block_causal_mask(tokens.shape[1], self.window, tokens.device)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return self.norm(x)

    def forward(
        self, tokens: torch.Tensor, boundaries: List[int], patch_lengths: List[int]
    ) -> torch.Tensor:
        """Encode one sentence into its sequence of patch representations.

        ``tokens``: [1, T]. Returns [P, concept_dim] — the patch *sequence*,
        not a pooled sentence vector.
        """
        reps, _ = self.forward_batch(tokens, build_patch_index(
            [(list(boundaries), list(patch_lengths))], tokens.device
        ))
        return reps[0]

    def forward_batch(self, tokens: torch.Tensor, index: PatchIndex):
        """Encode a whole batch of sentences into padded patch sequences.

        ``tokens``: [B, T] right-padded byte ids. Right-padding is safe because
        the local attention mask is causal, so a real byte never attends to the
        padding that follows it.

        Returns ``(patch_reps, patch_mask)`` with shapes [B, P, concept_dim] and
        [B, P], where P is the longest patch sequence in the batch.

        Every patch cross-attends only over its own bytes, so pooling the whole
        batch in one call is numerically identical to pooling each sentence
        separately — it just stops leaving the GPU idle between sentences.
        """
        B = tokens.shape[0]
        device = tokens.device
        P = index.max_patches
        if P == 0:
            return (
                torch.zeros(B, 0, self.concept_dim, device=device),
                torch.zeros(B, 0, dtype=torch.bool, device=device),
            )

        hidden = self.byte_hidden(tokens)  # [B, T, dim]
        # [total_P, max_L, dim]: one row per patch, gathered across the batch.
        patch_hidden = hidden[index.owner.unsqueeze(1), index.byte_idx]
        patch_reps = self.pooler(patch_hidden, index.key_mask)  # [total_P, concept_dim]

        out = patch_reps.new_zeros(B, P, self.concept_dim)
        out[index.owner, index.slot] = patch_reps
        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[index.owner, index.slot] = True
        return out, mask


class BLTLatentTransformer(nn.Module):
    """Global latent transformer over patch representations (BLT §3.1).

    "an autoregressive transformer model G with l_G layers, which maps a
    sequence of latent input patch representations p_j into a sequence of
    output patch representations o_j", using a block-causal attention mask.

    Without this stage the patch sequence collapses straight to a mean and the
    architecture's central mechanism — spending a transformer step per patch —
    never runs.
    """

    def __init__(
        self,
        concept_dim: int = 1024,
        model_dim: int = 1024,
        n_layers: int = 4,
        n_heads: int = 8,
        max_patches: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = (
            nn.Linear(concept_dim, model_dim)
            if concept_dim != model_dim
            else nn.Identity()
        )
        self.pos_emb = nn.Embedding(max_patches, model_dim)
        self.max_patches = max_patches
        self.layers = nn.ModuleList(
            [_TransformerLayer(model_dim, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = (
            nn.Linear(model_dim, concept_dim)
            if concept_dim != model_dim
            else nn.Identity()
        )

    def forward(self, patch_reps: torch.Tensor) -> torch.Tensor:
        """``patch_reps``: [P, concept_dim] -> [P, concept_dim]."""
        if patch_reps.shape[0] == 0:
            return patch_reps
        return self.forward_batch(patch_reps.unsqueeze(0))[0]

    def forward_batch(self, patch_reps: torch.Tensor) -> torch.Tensor:
        """``patch_reps``: [B, P, concept_dim] -> [B, P, concept_dim].

        Shorter sentences are right-padded to P. The attention mask is causal,
        so a real patch never attends to the padding after it and the batched
        result matches the per-sentence one; the rows sitting on padded
        positions are garbage and the caller masks them out when pooling.
        """
        if patch_reps.shape[1] == 0:
            return patch_reps
        x = self.in_proj(patch_reps)  # [B, P, model_dim]
        P = x.shape[1]
        pos = torch.arange(min(P, self.max_patches), device=x.device)
        if P > self.max_patches:
            # Clamp rather than crash on unusually long sentences.
            pos = torch.cat([pos, pos[-1].repeat(P - self.max_patches)])
        x = x + self.pos_emb(pos).unsqueeze(0)
        causal = torch.tril(torch.ones(P, P, dtype=torch.bool, device=x.device))
        for layer in self.layers:
            x = layer(x, attn_mask=causal)
        return self.out_proj(self.norm(x))


class BLTSentenceEncoder(nn.Module):
    """Full BLT byte-side stack: bytes -> patches -> latent transformer -> concept.

    The entropy model supplies boundaries only; this module owns every trainable
    parameter on the byte side.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        dim: int = 256,
        concept_dim: int = 1024,
        encoder_layers: int = 1,
        latent_layers: int = 4,
        n_heads: int = 8,
        window: int = 512,
        use_hash_ngrams: bool = True,
        hash_vocab_size: int = DEFAULT_HASH_VOCAB,
        ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.encoder = BLTLocalEncoder(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=encoder_layers,
            n_heads=n_heads,
            concept_dim=concept_dim,
            window=window,
            ngram_sizes=ngram_sizes,
            hash_vocab_size=hash_vocab_size,
            use_hash_ngrams=use_hash_ngrams,
        )
        self.latent = BLTLatentTransformer(
            concept_dim=concept_dim,
            model_dim=concept_dim,
            n_layers=latent_layers,
            n_heads=n_heads,
        )

    def forward(
        self, tokens: torch.Tensor, boundaries: List[int], patch_lengths: List[int]
    ) -> torch.Tensor:
        """Return the sentence concept vector [concept_dim]."""
        index = build_patch_index(
            [(list(boundaries), list(patch_lengths))], tokens.device
        )
        return self.forward_batch(tokens, index)[0]

    def forward_batch(self, tokens: torch.Tensor, index: PatchIndex) -> torch.Tensor:
        """Encode a batch of right-padded sentences into concepts [B, concept_dim].

        This is the path training should use: one local-encoder forward, one
        pooling call and one latent-transformer forward for the entire batch,
        instead of a Python loop that runs all three per sentence.
        """
        B = tokens.shape[0]
        device = tokens.device
        patch_reps, mask = self.encoder.forward_batch(tokens, index)
        if patch_reps.shape[1] == 0:
            return torch.zeros(B, self.concept_dim, device=device)

        out = self.latent.forward_batch(patch_reps)  # [B, P, concept_dim]

        # Pool the contextualised patch sequence into one concept. The last
        # *real* position summarises the whole sentence under the causal mask;
        # mean pooling is used alongside it for a more stable signal early in
        # training. Both ignore the padded patch slots.
        m = mask.unsqueeze(-1).to(out.dtype)
        counts = mask.sum(dim=1)  # [B]
        mean = (out * m).sum(dim=1) / counts.clamp_min(1).unsqueeze(-1).to(out.dtype)
        last = out[torch.arange(B, device=device), (counts - 1).clamp_min(0)]
        concept = 0.5 * (last + mean)
        # Sentences with no patches at all contribute a zero concept.
        return concept * (counts > 0).unsqueeze(-1).to(out.dtype)
