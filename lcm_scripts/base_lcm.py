"""
Base-LCM Architecture
MSE-based regressor for next sentence prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class PreNet(nn.Module):
    def __init__(self, embed_dim, model_dim):
        super().__init__()
        self.normalize = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, model_dim)

    def forward(self, x):
        x = self.normalize(x)
        return self.linear(x)


class PostNet(nn.Module):
    def __init__(self, model_dim, embed_dim):
        super().__init__()
        self.denormalize = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, embed_dim)

    def forward(self, x):
        x = self.denormalize(x)
        return self.linear(x)


class BaseLCM(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        model_dim=2048,
        n_layers=24,
        n_heads=16,
        max_seq_len=128,
        checkpointing=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.checkpointing = checkpointing

        self.prenet = PreNet(embed_dim, model_dim)
        self.postnet = PostNet(model_dim, embed_dim)

        # Position embeddings
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)

        # Beginning-of-sequence token (in model_dim space, used as decoder start)
        self.bos_emb = nn.Parameter(torch.randn(model_dim))

        # Transformer decoder layers (keep as ModuleList so we can optionally
        # apply activation checkpointing per-layer to reduce memory) and end of text token
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerDecoderLayer(
                d_model=model_dim,
                nhead=n_heads,
                dim_feedforward=4 * model_dim,
                dropout=0.1,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
            )
            self.layers.append(layer)

        # End of text token
        self.eot_emb = nn.Parameter(torch.randn(embed_dim))

    def forward(self, src_embs, tgt_embs=None):
        """
        src_embs: [batch, seq_len, embed_dim] - previous sentences
        tgt_embs: [batch, 1, embed_dim] - target next sentence (for training)
        """
        batch_size, seq_len, _ = src_embs.shape

        # PreNet
        src = self.prenet(src_embs)  # [batch, seq_len, model_dim]

        # Add position embeddings
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0)
        src = src + self.pos_emb(positions)

        if tgt_embs is not None:
            # Training: teacher forcing with BOS prefix.
            # Decoder input = [BOS, prenet(tgt_embs[:, :-1, :])]
            # Decoder target = tgt_embs  (position k predicts k-th target embedding)
            single_step = tgt_embs.dim() == 3 and tgt_embs.shape[1] == 1
            bos = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            tgt_input = self.prenet(tgt_embs[:, :-1, :])  # [batch, L-1, model_dim]
            tgt_input = torch.cat([bos, tgt_input], dim=1)  # [batch, L, model_dim]
            L = tgt_input.shape[1]
            tgt_pos = torch.arange(seq_len, seq_len + L, device=src.device).unsqueeze(0)
            tgt_input = tgt_input + self.pos_emb(tgt_pos)

            # Causal mask: position k attends to positions ≤ k (safe because
            # input at position k is the k-1-th target, not the k-th, so no
            # autoencoder shortcut)
            tgt_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=src.device), diagonal=1
            )

            output = tgt_input
            from torch.utils.checkpoint import checkpoint as _cp

            for layer in self.layers:
                if self.checkpointing and self.training:
                    output = _cp(
                        lambda o, m, tmask, layer=layer: layer(o, m, tgt_mask=tmask),
                        output,
                        src,
                        tgt_mask,
                    )
                else:
                    output = layer(output, src, tgt_mask=tgt_mask)

            pred = self.postnet(output)  # [batch, L, embed_dim]
            if single_step:
                return pred.squeeze(1)  # [batch, embed_dim]
            return pred
        else:
            # Inference: start with BOS, predict the next embedding
            bos = self.bos_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
            tgt_pos = torch.arange(seq_len, seq_len + 1, device=src.device).unsqueeze(0)
            output = bos + self.pos_emb(tgt_pos)

            # Single-token mask: BOS attends to itself (diagonal=1 → [[False]])
            tgt_mask = torch.triu(
                torch.ones(1, 1, dtype=torch.bool, device=src.device), diagonal=1
            )
            from torch.utils.checkpoint import checkpoint as _cp

            for layer in self.layers:
                if self.checkpointing and self.training:
                    output = _cp(
                        lambda o, m, mask, layer=layer: layer(o, m, tgt_mask=mask),
                        output,
                        src,
                        tgt_mask,
                    )
                else:
                    output = layer(output, src, tgt_mask=tgt_mask)

            return self.postnet(output.squeeze(1))  # [batch, embed_dim]

    def generate_sequence(self, initial_embs, max_len=50):
        """Generate a sequence of embeddings autoregressively"""
        current_seq = initial_embs.clone()
        generated = [initial_embs]

        for _ in range(max_len):
            pred = self.forward(current_seq.unsqueeze(0)).squeeze(0)
            generated.append(pred.unsqueeze(0))
            current_seq = torch.cat([current_seq, pred.unsqueeze(0)], dim=0)

            # Check similarity to EOT
            if torch.cosine_similarity(pred, self.eot_emb, dim=0) > 0.9:
                break

        return torch.cat(generated[1:], dim=0)
