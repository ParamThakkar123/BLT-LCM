"""Base-LCM architecture (Barrault et al., 2024, §2.3.1).

A decoder-only Transformer that transduces a sequence of preceding concepts
into a sequence of future ones, wrapped by a PreNet and a PostNet::

    x -> PreNet -> decoder-only Transformer (causal) -> PostNet -> x_hat

PreNet and PostNet implement Eqs. (1)-(4): a *fixed* robust scaler (median /
IQR, fit once on sampled embeddings) moves SONAR vectors to a well-conditioned
scale on the way in, and its exact inverse is applied on the way out so
predictions land back in raw SONAR coordinates -- which is what the frozen
SONAR decoder expects.

Note the paper's own finding (Tables 3-4): with an MSE objective this model
regresses towards the *mean* of the plausible next sentences, which need not be
a valid point in SONAR space. ``diffusion_lcm.py`` and ``quant_lcm.py``
implement the stronger variants; this module is the baseline they are measured
against.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class RobustScaler(nn.Module):
    """Median / IQR scaler with frozen statistics (LCM Eq. 4).

    "we fit a robust scaler to a set of randomly sampled SONAR vectors from
    different corpora and domains of text data. This scaler removes the median
    statistics and scales the data according to the interquartile range."

    The statistics are buffers, so they travel with the checkpoint and are
    identical at train and inference time. Until ``fit`` is called this is the
    identity, and ``is_fitted`` is False so callers can warn.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("median", torch.zeros(dim))
        self.register_buffer("iqr", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    @torch.no_grad()
    def fit(self, samples: torch.Tensor, eps: float = 1e-6):
        """Fit on ``[N, dim]`` sampled embeddings. N should be large (>= 10k)."""
        if samples.dim() != 2:
            raise ValueError(f"expected [N, dim] samples, got {tuple(samples.shape)}")
        if samples.shape[0] < 2:
            raise ValueError("need at least 2 samples to fit a robust scaler")
        samples = samples.to(torch.float32)
        q = torch.quantile(
            samples, torch.tensor([0.25, 0.5, 0.75], device=samples.device), dim=0
        )
        self.median.copy_(q[1])
        self.iqr.copy_((q[2] - q[0]).clamp_min(eps))
        self.fitted.fill_(True)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.median) / self.iqr

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.median + self.iqr * x


class PreNet(nn.Module):
    """LCM Eq. (1): PreNet(x) = normalize(x) W_pre^T + b_pre."""

    def __init__(self, embed_dim, model_dim, scaler: RobustScaler):
        super().__init__()
        self.scaler = scaler
        self.linear = nn.Linear(embed_dim, model_dim)

    def forward(self, x):
        return self.linear(self.scaler.normalize(x))


class PostNet(nn.Module):
    """LCM Eq. (2): PostNet(x) = denormalize(x W_post^T + b_post).

    The denormalization is applied *after* the projection, so the output is in
    raw SONAR coordinates rather than in normalized space.
    """

    def __init__(self, model_dim, embed_dim, scaler: RobustScaler):
        super().__init__()
        self.scaler = scaler
        self.linear = nn.Linear(model_dim, embed_dim)

    def forward(self, x):
        return self.scaler.denormalize(self.linear(x))


class BaseLCM(nn.Module):
    """Decoder-only Transformer over concept embeddings.

    ``forward(src_embs)`` returns the next-concept prediction after the whole
    prefix, ``[batch, embed_dim]``.

    ``forward(src_embs, tgt_embs)`` keeps the historical calling convention:
    with ``tgt_embs`` of shape ``[batch, 1, embed_dim]`` it returns
    ``[batch, embed_dim]``, otherwise ``[batch, L, embed_dim]``. ``tgt_embs`` is
    used for its shape only -- under a causal mask, teacher forcing is already
    implicit in ``src_embs``, so feeding the targets in would leak them.

    ``forward_all(src_embs)`` returns a prediction at *every* position,
    ``[batch, seq_len, embed_dim]``, where position t predicts concept t+1.
    That is the efficient training path: one pass covers the whole document
    instead of one pass per position.
    """

    def __init__(
        self,
        embed_dim=1024,
        model_dim=2048,
        n_layers=24,
        n_heads=16,
        max_seq_len=1024,
        dropout=0.1,
        checkpointing=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.checkpointing = checkpointing

        self.scaler = RobustScaler(embed_dim)
        self.prenet = PreNet(embed_dim, model_dim, self.scaler)
        self.postnet = PostNet(model_dim, embed_dim, self.scaler)

        self.pos_emb = nn.Embedding(max_seq_len, model_dim)

        # Decoder-only stack: self-attention over the concept sequence with a
        # causal mask. TransformerEncoderLayer is the decoder-only block here
        # (it has no cross-attention); the causal mask makes it autoregressive.
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=model_dim,
                    nhead=n_heads,
                    dim_feedforward=4 * model_dim,
                    dropout=dropout,
                    activation=F.gelu,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

        # End-of-text concept. The paper suffixes every training document with
        # SONAR-encoded "End of text." and stops generation when the prediction
        # is close to it. It is a buffer, not a Parameter: it must equal the
        # encoder's embedding of that sentence, not something learned. Call
        # ``set_eot_embedding(encoder.encode(["End of text."])[0])`` after
        # constructing the model.
        self.register_buffer("eot_emb", torch.zeros(embed_dim))
        self.register_buffer("eot_set", torch.zeros(1, dtype=torch.bool))

    # -- setup ------------------------------------------------------------- #

    @torch.no_grad()
    def set_eot_embedding(self, emb: torch.Tensor):
        """Install the encoded "End of text." concept used as the stop signal."""
        emb = emb.detach().to(self.eot_emb.device, self.eot_emb.dtype).reshape(-1)
        if emb.numel() != self.embed_dim:
            raise ValueError(
                f"eot embedding has {emb.numel()} dims, expected {self.embed_dim}"
            )
        self.eot_emb.copy_(emb)
        self.eot_set.fill_(True)
        return self

    def fit_normalizer(self, samples: torch.Tensor):
        """Fit the PreNet/PostNet robust scaler on sampled concept vectors."""
        self.scaler.fit(samples)
        return self

    # -- core -------------------------------------------------------------- #

    def _backbone(self, src_embs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = src_embs.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence of {seq_len} concepts exceeds max_seq_len="
                f"{self.max_seq_len}; rebuild BaseLCM with a larger max_seq_len"
            )
        h = self.prenet(src_embs)
        positions = torch.arange(seq_len, device=h.device).unsqueeze(0)
        h = h + self.pos_emb(positions)

        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=h.device),
            diagonal=1,
        )
        from torch.utils.checkpoint import checkpoint as _cp

        for layer in self.layers:
            if self.checkpointing and self.training:
                h = _cp(
                    lambda inp, m, layer=layer: layer(inp, src_mask=m),
                    h,
                    causal,
                    use_reentrant=False,
                )
            else:
                h = layer(h, src_mask=causal)
        return self.final_norm(h)

    def forward_all(self, src_embs: torch.Tensor) -> torch.Tensor:
        """Next-concept prediction at every position: [batch, seq_len, embed_dim]."""
        return self.postnet(self._backbone(src_embs))

    def forward(self, src_embs, tgt_embs=None):
        preds = self.forward_all(src_embs)
        if tgt_embs is None:
            return preds[:, -1]
        single_step = tgt_embs.dim() == 3 and tgt_embs.shape[1] == 1
        if single_step:
            return preds[:, -1]
        # Predict the L targets that follow the last L source positions.
        L = tgt_embs.shape[1]
        return preds[:, -L:]

    # -- generation -------------------------------------------------------- #

    @torch.no_grad()
    def generate_sequence(
        self,
        initial_embs: torch.Tensor,
        max_len: int = 50,
        s_eot: float = 0.9,
        s_prev: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressively continue a sequence of concepts (LCM §2.3.1).

        Stops when the new concept is within ``s_eot`` cosine similarity of the
        encoded "End of text." concept, or within ``s_prev`` of the previous
        generation (the paper's second early-stopping mechanism, which catches
        the model looping).
        """
        if not bool(self.eot_set.item()):
            raise RuntimeError(
                "EOT concept is unset, so the stop criterion cannot fire. Call "
                'set_eot_embedding(encoder.encode(["End of text."])[0]) first.'
            )
        current = initial_embs.clone()
        generated = []
        prev = None
        for _ in range(max_len):
            pred = self.forward(current.unsqueeze(0)).squeeze(0)
            if torch.cosine_similarity(pred, self.eot_emb, dim=0) > s_eot:
                break
            if prev is not None and torch.cosine_similarity(pred, prev, dim=0) > s_prev:
                break
            generated.append(pred.unsqueeze(0))
            prev = pred
            current = torch.cat([current, pred.unsqueeze(0)], dim=0)
        if not generated:
            return torch.empty(0, self.embed_dim, device=initial_embs.device)
        return torch.cat(generated, dim=0)
