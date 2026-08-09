"""Quantized Large Concept Model (Barrault et al., 2024, §2.3.5).

Discretizes the concept space with Residual Vector Quantization (RVQ) and models
the next concept coarse-to-fine over the resulting units. Unlike diffusion, this
gives back the usual sampling controls -- temperature, top-k, top-p -- because
the model emits a categorical distribution again.

Two heads are provided, matching the paper:

``Quant-LCM-d``  discrete targets, cross-entropy over the next codebook's units.
``Quant-LCM-c``  continuous targets, MSE against the residual, with optional
                 softmax-over-distance sampling (Eq. 23).

Rather than materialising ``n_codebooks * units_per_codebook`` output logits,
the codebook index is fed to the model as an embedding and the head predicts
only ``units_per_codebook`` units, as the paper does for parameter efficiency.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_lcm import _Normalizer


class ResidualVectorQuantizer(nn.Module):
    """Coarse-to-fine RVQ over concept vectors (Zeghidour et al., 2021).

    Each codebook quantizes the residual left by the previous ones, so the
    cumulative sum of the selected centroids is a progressively finer
    approximation of the input.
    """

    def __init__(
        self,
        dim: int = 1024,
        n_codebooks: int = 64,
        units_per_codebook: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.n_codebooks = n_codebooks
        self.units_per_codebook = units_per_codebook
        self.register_buffer(
            "codebooks", torch.zeros(n_codebooks, units_per_codebook, dim)
        )
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    @torch.no_grad()
    def fit(self, samples: torch.Tensor, iters: int = 10, verbose: bool = True):
        """Fit the codebooks by iterative k-means on successive residuals.

        ``samples``: [N, dim]. The paper trains on 15M English sentences with 64
        codebooks of 8192 units; scale ``samples`` to what you can hold.
        """
        residual = samples.detach().to(torch.float32).clone()
        for c in range(self.n_codebooks):
            centroids = self._kmeans(residual, self.units_per_codebook, iters)
            self.codebooks[c].copy_(centroids)
            idx = torch.cdist(residual, centroids).argmin(dim=1)
            residual = residual - centroids[idx]
            if verbose:
                err = residual.pow(2).mean().item()
                print(f"  [RVQ] codebook {c + 1}/{self.n_codebooks} residual MSE {err:.6f}")
        self.fitted.fill_(True)
        return self

    @staticmethod
    def _kmeans(x: torch.Tensor, k: int, iters: int) -> torch.Tensor:
        n = x.shape[0]
        k = min(k, n)
        perm = torch.randperm(n, device=x.device)[:k]
        centroids = x[perm].clone()
        for _ in range(iters):
            idx = torch.cdist(x, centroids).argmin(dim=1)
            for j in range(k):
                sel = x[idx == j]
                if sel.numel():
                    centroids[j] = sel.mean(dim=0)
        if centroids.shape[0] < x.shape[0]:
            pad = torch.zeros(
                0, centroids.shape[1], device=x.device, dtype=centroids.dtype
            )
            centroids = torch.cat([centroids, pad], dim=0)
        return centroids

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return unit indices ``[N, n_codebooks]``."""
        residual = x.detach().to(torch.float32).clone()
        out = []
        for c in range(self.n_codebooks):
            idx = torch.cdist(residual, self.codebooks[c]).argmin(dim=1)
            out.append(idx)
            residual = residual - self.codebooks[c][idx]
        return torch.stack(out, dim=1)

    def decode(self, indices: torch.Tensor, n_codebooks: Optional[int] = None):
        """Cumulative centroid sum for the first ``n_codebooks`` levels."""
        n = n_codebooks or indices.shape[1]
        out = torch.zeros(indices.shape[0], self.dim, device=indices.device)
        for c in range(n):
            out = out + self.codebooks[c][indices[:, c]]
        return out


class QuantLCM(nn.Module):
    """Coarse-to-fine autoregressive model over RVQ units.

    At each refinement step the model sees the clean left context plus the
    partial reconstruction of the current concept (the cumulative sum of the
    centroids chosen so far) and predicts the next residual.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        model_dim: int = 2048,
        n_layers: int = 32,
        n_heads: int = 16,
        max_seq_len: int = 2048,
        n_codebooks: int = 64,
        units_per_codebook: int = 8192,
        target: str = "discrete",  # "discrete" (Quant-LCM-d) | "continuous" (-c)
        dropout: float = 0.1,
        cfg_dropout: float = 0.15,
    ):
        super().__init__()
        if target not in ("discrete", "continuous"):
            raise ValueError("target must be 'discrete' or 'continuous'")
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.target = target
        self.max_seq_len = max_seq_len
        self.cfg_dropout = cfg_dropout

        # Concepts are quantized in *normalized* space, matching the other LCM
        # variants, so the codebooks are not dominated by whichever dimensions
        # happen to have the largest raw scale.
        self.scaler = _Normalizer(embed_dim)
        self.quantizer = ResidualVectorQuantizer(
            embed_dim, n_codebooks, units_per_codebook
        )

        self.in_proj = nn.Linear(embed_dim, model_dim)
        self.partial_proj = nn.Linear(embed_dim, model_dim)
        # Replaces the diffusion timestep embedding: which refinement level we
        # are on (§2.3.5, "diffusion timestep embeddings as input are replaced
        # by codebook index embeddings").
        self.codebook_emb = nn.Embedding(n_codebooks, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    model_dim, n_heads, 4 * model_dim, dropout,
                    activation=F.gelu, batch_first=True, norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(model_dim)
        out_dim = units_per_codebook if target == "discrete" else embed_dim
        self.head = nn.Linear(model_dim, out_dim)

    # -- setup ------------------------------------------------------------- #

    def fit_normalizer(self, samples: torch.Tensor):
        """Fit the median/IQR scaler. Call before ``fit_quantizer``."""
        self.scaler.fit(samples)
        return self

    def fit_quantizer(self, samples: torch.Tensor, **kw):
        """Fit the RVQ codebooks on *normalized* concepts.

        Normalization is applied here rather than by the caller, so the
        codebooks always live in the same space the model predicts in.
        """
        self.quantizer.fit(self.scaler.normalize(samples), **kw)
        return self

    # -- core -------------------------------------------------------------- #

    def _backbone(self, context: torch.Tensor, partial: torch.Tensor, level: torch.Tensor):
        """``context``: [B, L, D] clean prefix. ``partial``: [B, D] partial
        reconstruction of the concept being predicted. ``level``: [B] codebook
        index. Returns [B, model_dim]."""
        B, L, _ = context.shape
        h = self.in_proj(context)
        pos = torch.arange(L, device=h.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        query = (
            self.partial_proj(partial) + self.codebook_emb(level)
        ).unsqueeze(1)
        x = torch.cat([h, query], dim=1)
        n = L + 1
        causal = torch.triu(
            torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, src_mask=causal)
        return self.out_norm(x)[:, -1]

    def loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Next-concept loss over documents ``[B, L, D]``.

        One random refinement level per example, matching the paper's
        "randomly sample codebook index k between 1 and n_codebooks".
        """
        if not self.quantizer.is_fitted:
            raise RuntimeError("call fit_quantizer(...) before training QuantLCM")
        B, L, D = embeddings.shape
        if L < 2:
            raise ValueError("need at least 2 concepts per document")
        embeddings = self.scaler.normalize(embeddings)
        context = embeddings[:, :-1]
        target = embeddings[:, -1]

        codes = self.quantizer.encode(target)  # [B, n_codebooks]
        k = torch.randint(0, self.quantizer.n_codebooks, (B,), device=embeddings.device)
        partial = torch.zeros(B, D, device=embeddings.device)
        for c in range(self.quantizer.n_codebooks):
            sel = k > c
            if sel.any():
                partial[sel] += self.quantizer.codebooks[c][codes[sel, c]]

        ctx = context
        if self.cfg_dropout > 0 and self.training:
            drop = torch.rand(B, 1, 1, device=ctx.device) < self.cfg_dropout
            ctx = ctx.masked_fill(drop, 0.0)

        h = self._backbone(ctx, partial, k)
        out = self.head(h)
        if self.target == "discrete":
            tgt_units = codes.gather(1, k.unsqueeze(1)).squeeze(1)
            return F.cross_entropy(out, tgt_units)
        residual = target - partial
        return F.mse_loss(out, residual)

    @torch.no_grad()
    def sample_next(
        self,
        prefix: torch.Tensor,
        n_codebooks: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 1,
        guidance_scale: float = 1.0,
        beta: float = 200.0,
    ) -> torch.Tensor:
        """Generate the next concept by successive refinement."""
        B, _, D = prefix.shape
        n = n_codebooks or self.quantizer.n_codebooks
        prefix = self.scaler.normalize(prefix)
        partial = torch.zeros(B, D, device=prefix.device)
        null_ctx = torch.zeros_like(prefix)
        for c in range(n):
            level = torch.full((B,), c, device=prefix.device, dtype=torch.long)
            h = self._backbone(prefix, partial, level)
            out = self.head(h)
            if guidance_scale != 1.0:
                out_u = self.head(self._backbone(null_ctx, partial, level))
                out = out_u + guidance_scale * (out - out_u)

            if self.target == "discrete":
                logits = out / max(temperature, 1e-6)
                if top_k and top_k < logits.shape[-1]:
                    v, _ = logits.topk(top_k, dim=-1)
                    logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
                idx = torch.multinomial(logits.softmax(dim=-1), 1).squeeze(1)
            else:
                # Eq. (23): sample a centroid by softmax over negative distance.
                dist = torch.cdist(out, self.quantizer.codebooks[c])
                probs = torch.softmax(-beta * dist, dim=-1)
                if top_k and top_k < probs.shape[-1]:
                    v, _ = probs.topk(top_k, dim=-1)
                    probs = probs.masked_fill(probs < v[:, -1:], 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                idx = torch.multinomial(probs, 1).squeeze(1)
            partial = partial + self.quantizer.codebooks[c][idx]
        # Back to raw concept coordinates, matching the other LCM variants.
        return self.scaler.denormalize(partial)

    def forward(self, src_embs, tgt_embs=None):
        return self.sample_next(src_embs)
