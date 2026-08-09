"""Diffusion-based Large Concept Models (Barrault et al., 2024, §2.3.2-2.3.4).

Implements the two diffusion architectures the LCM paper found to outperform
the MSE Base-LCM:

``OneTowerDiffusionLCM``  §2.3.3. A single transformer backbone that denoises
                          x_n conditioned on the clean prefix x_<n, trained by
                          interleaving noisy and clean embeddings so every
                          position in a document is supervised in one pass.
``TwoTowerDiffusionLCM``  §2.3.4. A causal *contextualizer* over the prefix and
                          a separate *denoiser* that cross-attends to it, with
                          AdaLN modulation from the diffusion timestep.

Both use x_0-prediction with the simple reconstruction loss (Eq. 16), a
variance-preserving forward process (Eq. 7-9), zero-terminal-SNR rescaling of
the variance schedule (Lin et al., 2024), classifier-free guidance, guidance
rescaling and epsilon-scaling at inference.

Noise schedules: cosine (default), quadratic, and the sigmoid schedule the
paper introduces in Eq. (14) to study the effect of the log-SNR distribution.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Noise schedules (LCM §2.3.2, Figure 5)
# --------------------------------------------------------------------------- #


def _enforce_zero_terminal_snr(alphas_bar: torch.Tensor) -> torch.Tensor:
    """Rescale alpha_bar so the terminal SNR is exactly zero (Lin et al., 2024).

    "we follow Lin et al. (2024) and rescale the variance schedule to enforce
    zero terminal SNR". Without this, the last training step still contains a
    trace of the signal while inference starts from pure noise.
    """
    alphas_bar_sqrt = alphas_bar.sqrt()
    first = alphas_bar_sqrt[0].clone()
    last = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt = alphas_bar_sqrt - last
    alphas_bar_sqrt = alphas_bar_sqrt * (first / (first - last))
    return alphas_bar_sqrt.pow(2)


def make_noise_schedule(
    name: str = "cosine",
    timesteps: int = 100,
    sigmoid_gamma: float = 1.5,
    sigmoid_delta: float = -1.0,
    quad_beta_start: float = 0.001,
    quad_beta_end: float = 0.0012,
) -> torch.Tensor:
    """Return alpha_bar over ``timesteps``, with zero terminal SNR enforced.

    The sigmoid schedule is parameterized as the paper writes it: gamma
    controls the *scale* of the log-SNR distribution and delta its centre, and
    the paper's ``Sigmoid(1.5, -1)`` means ``gamma=1.5, delta=-1``. gamma must
    be positive, otherwise f(t) increases with t and the "schedule" adds signal
    instead of noise.
    """
    t = torch.linspace(0, 1, timesteps + 1)[1:]
    if name == "cosine":
        # Eq. (12), Nichol & Dhariwal with s = 0.008
        s = 0.008
        f = torch.cos((t + s) / (1 + s) * math.pi / 2).pow(2)
        f0 = math.cos(s / (1 + s) * math.pi / 2) ** 2
        alphas_bar = f / f0
    elif name == "quadratic":
        # Eq. (13): betas increase quadratically from beta_0 to beta_1
        betas = (
            math.sqrt(quad_beta_start)
            + t * (math.sqrt(quad_beta_end) - math.sqrt(quad_beta_start))
        ) ** 2
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
    elif name == "sigmoid":
        # Eq. (14): f(t) = sigmoid(delta - gamma * logit(t))
        if sigmoid_gamma <= 0:
            raise ValueError(
                f"sigmoid_gamma must be > 0 (got {sigmoid_gamma}); a non-positive "
                "gamma makes alpha_bar increase with t, i.e. removes noise"
            )
        eps = 1e-6
        tc = t.clamp(eps, 1 - eps)
        logit = torch.log(tc / (1 - tc))
        f = torch.sigmoid(sigmoid_delta - sigmoid_gamma * logit)
        # f(0) evaluated at the same clamp, so alpha_bar[0] == 1 exactly.
        f0 = torch.sigmoid(
            torch.tensor(sigmoid_delta - sigmoid_gamma * math.log(eps / (1 - eps)))
        )
        alphas_bar = f / f0
    else:
        raise ValueError(f"unknown noise schedule {name!r}")
    alphas_bar = alphas_bar.clamp(1e-8, 1.0)
    return _enforce_zero_terminal_snr(alphas_bar).clamp(0.0, 1.0)


class GaussianDiffusion(nn.Module):
    """Variance-preserving forward process and x_0-prediction sampler."""

    def __init__(self, timesteps: int = 100, schedule: str = "cosine", **kw):
        super().__init__()
        self.timesteps = timesteps
        alphas_bar = make_noise_schedule(schedule, timesteps, **kw)
        self.register_buffer("alphas_bar", alphas_bar)
        # Eq. (9): alpha_t^2 = sigmoid(lambda_t), sigma_t^2 = 1 - alpha_t^2
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Eq. (8): x_t = alpha_t x_0 + sigma_t eps."""
        a = self.sqrt_alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        s = self.sqrt_one_minus_alphas_bar[t].view(-1, *([1] * (x0.dim() - 1)))
        return a * x0 + s * noise

    def log_snr(self, t: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_bar[t]
        return torch.log(ab / (1.0 - ab).clamp_min(1e-8))

    def loss_weight(
        self, t: torch.Tensor, strategy: str = "simple", lmin: float = 0.0, lmax: float = 10.0
    ) -> torch.Tensor:
        """Eq. (17): simple (all ones) or clamped-SNR weighting."""
        if strategy == "simple":
            return torch.ones_like(t, dtype=torch.float32)
        if strategy == "clamped_snr":
            return self.log_snr(t).exp().clamp(lmin, lmax)
        raise ValueError(f"unknown loss weighting {strategy!r}")


def timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0):
    """Sinusoidal frequency embedding of the diffusion timestep."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #


class _Normalizer(nn.Module):
    """Robust scaler shared with base_lcm.RobustScaler semantics."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("median", torch.zeros(dim))
        self.register_buffer("iqr", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, samples: torch.Tensor, eps: float = 1e-6):
        q = torch.quantile(
            samples.float(),
            torch.tensor([0.25, 0.5, 0.75], device=samples.device),
            dim=0,
        )
        self.median.copy_(q[1])
        self.iqr.copy_((q[2] - q[0]).clamp_min(eps))
        self.fitted.fill_(True)
        return self

    def normalize(self, x):
        return (x - self.median) / self.iqr

    def denormalize(self, x):
        return self.median + self.iqr * x


class AdaLNBlock(nn.Module):
    """Transformer block with adaptive layer-norm modulation (LCM Eq. 21-22).

    Regresses channel-wise scale, shift and residual gate from the timestep
    embedding. The output projection of each residual branch is zero-initialized
    so the block starts as the identity (Peebles & Xie, 2023).
    """

    def __init__(self, dim: int, n_heads: int, cross_attn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = (
            nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            if cross_attn
            else None
        )
        if cross_attn:
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )
        n_mod = 9 if cross_attn else 6
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, n_mod * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x, t_emb, memory=None, memory_mask=None):
        mods = self.modulation(t_emb).chunk(
            9 if self.cross_attn is not None else 6, dim=-1
        )
        if self.cross_attn is not None:
            s1, g1, a1, s2, g2, a2, s3, g3, a3 = mods
        else:
            s1, g1, a1, s3, g3, a3 = mods
            s2 = g2 = a2 = None

        h = self.norm1(x) * (1 + g1.unsqueeze(1)) + s1.unsqueeze(1)
        attn, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + a1.unsqueeze(1) * attn

        if self.cross_attn is not None and memory is not None:
            h = self.norm2(x) * (1 + g2.unsqueeze(1)) + s2.unsqueeze(1)
            attn, _ = self.cross_attn(
                h, memory, memory, need_weights=False, attn_mask=memory_mask
            )
            x = x + a2.unsqueeze(1) * attn

        h = self.norm3(x) * (1 + g3.unsqueeze(1)) + s3.unsqueeze(1)
        return x + a3.unsqueeze(1) * self.ff(h)


# --------------------------------------------------------------------------- #
# Two-Tower diffusion LCM (§2.3.4)
# --------------------------------------------------------------------------- #


class TwoTowerDiffusionLCM(nn.Module):
    """Contextualizer over the prefix + denoiser cross-attending to it.

    This is the variant the paper scales to 7B, chosen for its smaller memory
    footprint on long contexts.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        model_dim: int = 2048,
        context_layers: int = 5,
        denoiser_layers: int = 13,
        n_heads: int = 16,
        max_seq_len: int = 2048,
        timesteps: int = 100,
        schedule: str = "cosine",
        dropout: float = 0.1,
        cfg_dropout: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.cfg_dropout = cfg_dropout

        self.scaler = _Normalizer(embed_dim)
        self.diffusion = GaussianDiffusion(timesteps, schedule)

        self.in_proj = nn.Linear(embed_dim, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)

        self.context_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    model_dim, n_heads, 4 * model_dim, dropout,
                    activation=F.gelu, batch_first=True, norm_first=True,
                )
                for _ in range(context_layers)
            ]
        )
        self.context_norm = nn.LayerNorm(model_dim)

        self.noisy_proj = nn.Linear(embed_dim, model_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(256, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
        self.denoiser = nn.ModuleList(
            [
                AdaLNBlock(model_dim, n_heads, cross_attn=True, dropout=dropout)
                for _ in range(denoiser_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, embed_dim)

    def fit_normalizer(self, samples):
        self.scaler.fit(samples)
        return self

    def contextualize(self, prefix: torch.Tensor) -> torch.Tensor:
        """Causally encode the clean prefix. ``prefix``: [B, L, embed_dim]."""
        B, L, _ = prefix.shape
        h = self.in_proj(self.scaler.normalize(prefix))
        pos = torch.arange(L, device=h.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1
        )
        for layer in self.context_layers:
            h = layer(h, src_mask=causal)
        return self.context_norm(h)

    def denoise(self, x_t, t, memory, memory_mask=None):
        """Predict x_0 from the noised concept. ``x_t``: [B, 1, embed_dim]."""
        h = self.noisy_proj(x_t)
        t_emb = self.t_embed(timestep_embedding(t, 256))
        for block in self.denoiser:
            h = block(h, t_emb, memory=memory, memory_mask=memory_mask)
        return self.out_proj(self.out_norm(h))

    def loss(self, embeddings: torch.Tensor, weighting: str = "simple") -> torch.Tensor:
        """Next-concept diffusion loss over a batch of documents [B, L, D].

        Every position is denoised in the same pass: position n is denoised
        conditioned on the contextualizer output at n-1, with a zero vector
        prepended so position 0 is predictable (§2.3.4).
        """
        B, L, D = embeddings.shape
        x0 = self.scaler.normalize(embeddings)
        memory = self.contextualize(embeddings)
        # Shift by one: predicting concept n uses context up to n-1.
        zero = torch.zeros(B, 1, self.model_dim, device=memory.device, dtype=memory.dtype)
        memory = torch.cat([zero, memory[:, :-1]], dim=1)

        t = torch.randint(0, self.diffusion.timesteps, (B * L,), device=x0.device)
        noise = torch.randn_like(x0).reshape(B * L, 1, D)
        flat_x0 = x0.reshape(B * L, 1, D)
        x_t = self.diffusion.q_sample(flat_x0, t, noise)

        mem = memory.reshape(B * L, 1, self.model_dim)
        # Classifier-free training: drop the conditioning on a random subset.
        if self.cfg_dropout > 0 and self.training:
            drop = torch.rand(B * L, device=mem.device) < self.cfg_dropout
            mem = mem.masked_fill(drop.view(-1, 1, 1), 0.0)

        pred = self.denoise(x_t, t, mem)
        w = self.diffusion.loss_weight(t, weighting).view(-1, 1, 1)
        return (w * (pred - flat_x0).pow(2)).mean()

    @torch.no_grad()
    def sample_next(
        self,
        prefix: torch.Tensor,
        steps: int = 40,
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.7,
        initial_noise_scale: float = 0.6,
        epsilon_scale: float = 1.00045,
    ) -> torch.Tensor:
        """Sample the next concept given a clean prefix ``[B, L, embed_dim]``."""
        B = prefix.shape[0]
        device = prefix.device
        memory = self.contextualize(prefix)[:, -1:]
        null_memory = torch.zeros_like(memory)

        T = self.diffusion.timesteps
        # "trailing" step selection (Lu et al., 2022), better for small S.
        taus = torch.linspace(T, 0, steps + 1)[:-1].round().long().clamp(0, T - 1)

        x = torch.randn(B, 1, self.embed_dim, device=device) * initial_noise_scale
        for i, tau in enumerate(taus):
            t = torch.full((B,), int(tau), device=device, dtype=torch.long)
            cond = self.denoise(x, t, memory)
            if guidance_scale != 1.0:
                uncond = self.denoise(x, t, null_memory)
                guided = uncond + guidance_scale * (cond - uncond)
                if guidance_rescale > 0:
                    # Rescale to keep the std of the guided prediction (Lin 2024)
                    std_cond = cond.std(dim=-1, keepdim=True)
                    std_guided = guided.std(dim=-1, keepdim=True).clamp_min(1e-6)
                    guided = guidance_rescale * (guided * std_cond / std_guided) + (
                        1 - guidance_rescale
                    ) * guided
                x0_pred = guided
            else:
                x0_pred = cond

            if i == len(taus) - 1:
                x = x0_pred
                break
            # DDIM-style step towards the next (lower) noise level.
            a_t = self.diffusion.sqrt_alphas_bar[tau]
            s_t = self.diffusion.sqrt_one_minus_alphas_bar[tau]
            eps = (x - a_t * x0_pred) / s_t.clamp_min(1e-6)
            # Epsilon-scaling shrinks the over-predicted error magnitude and
            # mitigates exposure bias (Ning et al., 2023).
            eps = eps / epsilon_scale
            tau_next = int(taus[i + 1])
            a_n = self.diffusion.sqrt_alphas_bar[tau_next]
            s_n = self.diffusion.sqrt_one_minus_alphas_bar[tau_next]
            x = a_n * x0_pred + s_n * eps

        return self.scaler.denormalize(x.squeeze(1))

    def forward(self, src_embs, tgt_embs=None):
        """Inference-compatible shim matching BaseLCM's calling convention."""
        return self.sample_next(src_embs)


# --------------------------------------------------------------------------- #
# One-Tower diffusion LCM (§2.3.3)
# --------------------------------------------------------------------------- #


class OneTowerDiffusionLCM(nn.Module):
    """Single backbone denoising x_n while attending to the clean prefix.

    Training interleaves noisy and clean embeddings so that every position is
    supervised in one pass, with an attention mask that lets a noisy position
    see only the clean context (§2.3.3, Figure 7).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        model_dim: int = 2048,
        n_layers: int = 32,
        n_heads: int = 16,
        max_seq_len: int = 2048,
        timesteps: int = 100,
        schedule: str = "cosine",
        dropout: float = 0.1,
        cfg_dropout: float = 0.15,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.cfg_dropout = cfg_dropout

        self.scaler = _Normalizer(embed_dim)
        self.diffusion = GaussianDiffusion(timesteps, schedule)

        self.in_proj = nn.Linear(embed_dim, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.t_embed = nn.Sequential(
            nn.Linear(256, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim)
        )
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
        self.out_proj = nn.Linear(model_dim, embed_dim)

    def fit_normalizer(self, samples):
        self.scaler.fit(samples)
        return self

    def _run(self, clean, noisy, t):
        """Interleave [clean_0, noisy_0, clean_1, noisy_1, ...] (Figure 7).

        ``clean``/``noisy``: [B, L, embed_dim]; ``t``: [B, L] timesteps.
        Returns the x_0 prediction at each noisy slot: [B, L, embed_dim].
        """
        B, L, _ = clean.shape
        c = self.in_proj(clean)
        n = self.in_proj(noisy) + self.t_embed(timestep_embedding(t.reshape(-1), 256)).view(
            B, L, self.model_dim
        )
        # interleave along the sequence: length 2L
        x = torch.stack([c, n], dim=2).reshape(B, 2 * L, self.model_dim)
        pos = torch.arange(L, device=x.device).repeat_interleave(2).unsqueeze(0)
        x = x + self.pos_emb(pos)

        # Mask: a noisy slot at position 2i+1 may attend to clean slots 2j
        # (j <= i) only; clean slots attend causally to earlier clean slots.
        idx = torch.arange(2 * L, device=x.device)
        is_clean = (idx % 2) == 0
        step = idx // 2
        allowed = is_clean.unsqueeze(0) & (step.unsqueeze(1) >= step.unsqueeze(0))
        # never let a position attend to a *later* clean slot
        allowed = allowed & (step.unsqueeze(0) <= step.unsqueeze(1))
        allowed = allowed | torch.eye(2 * L, dtype=torch.bool, device=x.device)
        attn_mask = ~allowed

        for layer in self.layers:
            x = layer(x, src_mask=attn_mask)
        x = self.out_proj(self.out_norm(x))
        return x[:, 1::2]  # the noisy slots

    def loss(self, embeddings: torch.Tensor, weighting: str = "simple") -> torch.Tensor:
        B, L, D = embeddings.shape
        x0 = self.scaler.normalize(embeddings)
        t = torch.randint(0, self.diffusion.timesteps, (B * L,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.diffusion.q_sample(x0.reshape(B * L, D), t, noise.reshape(B * L, D))
        x_t = x_t.view(B, L, D)

        clean = x0
        if self.cfg_dropout > 0 and self.training:
            drop = torch.rand(B, L, 1, device=x0.device) < self.cfg_dropout
            clean = clean.masked_fill(drop, 0.0)

        pred = self._run(clean, x_t, t.view(B, L))
        w = self.diffusion.loss_weight(t, weighting).view(B, L, 1)
        return (w * (pred - x0).pow(2)).mean()

    @torch.no_grad()
    def sample_next(
        self,
        prefix: torch.Tensor,
        steps: int = 40,
        guidance_scale: float = 3.0,
        initial_noise_scale: float = 0.6,
        epsilon_scale: float = 1.00045,
    ) -> torch.Tensor:
        B, L, D = prefix.shape
        device = prefix.device
        clean = self.scaler.normalize(prefix)
        # Append a slot for the concept being generated.
        clean = torch.cat([clean, torch.zeros(B, 1, D, device=device)], dim=1)
        T = self.diffusion.timesteps
        taus = torch.linspace(T, 0, steps + 1)[:-1].round().long().clamp(0, T - 1)

        x = torch.randn(B, 1, D, device=device) * initial_noise_scale
        noisy = torch.zeros(B, L + 1, D, device=device)
        for i, tau in enumerate(taus):
            noisy[:, -1:] = x
            t = torch.full((B, L + 1), int(tau), device=device, dtype=torch.long)
            cond = self._run(clean, noisy, t)[:, -1:]
            if guidance_scale != 1.0:
                uncond = self._run(torch.zeros_like(clean), noisy, t)[:, -1:]
                x0_pred = uncond + guidance_scale * (cond - uncond)
            else:
                x0_pred = cond
            if i == len(taus) - 1:
                x = x0_pred
                break
            a_t = self.diffusion.sqrt_alphas_bar[tau]
            s_t = self.diffusion.sqrt_one_minus_alphas_bar[tau]
            eps = ((x - a_t * x0_pred) / s_t.clamp_min(1e-6)) / epsilon_scale
            tau_n = int(taus[i + 1])
            x = (
                self.diffusion.sqrt_alphas_bar[tau_n] * x0_pred
                + self.diffusion.sqrt_one_minus_alphas_bar[tau_n] * eps
            )
        return self.scaler.denormalize(x.squeeze(1))

    def forward(self, src_embs, tgt_embs=None):
        return self.sample_next(src_embs)
