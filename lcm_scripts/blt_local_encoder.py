"""
BLT-faithful patch pooling (Byte Latent Transformer, Meta 2024, §3.2.2).

In BLT, patch representations are NOT read off the entropy model. The entropy
model is a separate small byte-LM used *only* to decide where patch boundaries
fall. Patch representations are formed by a local encoder: byte hidden states
are pooled into one vector per patch by a Perceiver-style multi-headed
cross-attention, where the query for patch j attends only to the byte hidden
states that belong to patch j (patch masking).

This module implements that pooling step. It consumes byte hidden states (from
the local-encoder backbone) and returns one representation per patch. It is a
small, learnable ``nn.Module`` so it can later be trained end-to-end; at
initialization it is deliberately configured to approximate a masked mean of
the byte hidden states, so even an untrained pooler produces sensible
embeddings (a strict improvement over averaging next-byte *logits*).
"""

import torch
import torch.nn as nn


class PatchCrossAttentionPooler(nn.Module):
    """Perceiver-style cross-attention pooling of bytes into a patch vector.

    Forward inputs:
        patch_hidden: [P, L, dim] byte hidden states for P padded patches.
        key_mask:     [P, L] bool, True for real bytes, False for padding.

    Returns:
        patch_reps:   [P, dim] one representation per patch.

    The single query per patch is initialized from the masked mean of that
    patch's byte hidden states (BLT initializes the patch query by pooling the
    patch's byte embeddings), then cross-attends over the patch's bytes only.
    Padding is masked out of the attention, so patch length never leaks into
    the pooled vector.
    """

    def __init__(self, dim: int, concept_dim: int = None, n_heads: int = 4):
        super().__init__()
        if dim % n_heads != 0:
            # Fall back to single head if dim is not divisible (e.g. 260).
            n_heads = 1
        self.dim = dim
        # Output concept dimension. Defaults to the byte-hidden width; set it to
        # 1024 to match SONAR / the LCM convention so the BLT and SONAR concept
        # sources are directly comparable.
        self.concept_dim = concept_dim or dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Projects the pooled patch vector to the concept dimension.
        self.out_proj = nn.Linear(dim, self.concept_dim, bias=False)

        self._init_near_mean()

    def _init_near_mean(self):
        """Initialize so the untrained pooler is a sensible masked pool.

        Small q/k weights make attention scores ~0 -> softmax is ~uniform over
        the patch's bytes, and an identity v projection makes the pooled vector
        ~= the mean of the (layer-normed) byte hidden states. When the output
        concept dimension equals the hidden width the out projection is identity
        (exact masked mean); otherwise it is a learned lift trained jointly with
        the decoder.
        """
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
