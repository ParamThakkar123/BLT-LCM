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

        # Transformer decoder layers (keep as ModuleList so we can optionally
        # apply activation checkpointing per-layer to reduce memory)
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

        # For autoregressive prediction, target is the next embedding
        if tgt_embs is not None:
            # Support both single-step targets of shape [B,1,E] and
            # multi-step targets [B, L, E]. Running the decoder once for the
            # full target sequence is much more efficient than looping per
            # position in the training loop.
            single_step = tgt_embs.dim() == 3 and tgt_embs.shape[1] == 1
            tgt = self.prenet(tgt_embs)  # [batch, L, model_dim] or [batch,1,model_dim]
            L = tgt.shape[1]
            tgt_pos = torch.arange(seq_len, seq_len + L, device=src.device).unsqueeze(0)
            tgt = tgt + self.pos_emb(tgt_pos)

            # Causal mask for decoder: shape [L, L]
            tgt_mask = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=src.device), diagonal=1
            )

            # Decode in a single call for all target positions
            # Run decoder layers sequentially; apply gradient checkpointing per
            # layer if enabled to reduce activation memory.
            output = tgt
            from torch.utils.checkpoint import checkpoint as _cp

            for layer in self.layers:
                if self.checkpointing and self.training:
                    # checkpoint requires positional args only
                    output = _cp(
                        lambda o, m, mask: layer(o, m, tgt_mask=mask),
                        output,
                        src,
                        tgt_mask,
                    )
                else:
                    output = layer(output, src, tgt_mask=tgt_mask)

            # PostNet
            pred = self.postnet(output)  # [batch, L, embed_dim]
            if single_step:
                return pred.squeeze(1)
            return pred
        else:
            # Inference: predict next (autoregressively) without artificial dummies.
            # Use the last source sentence representation as the initial real
            # conditioning token rather than creating a synthetic zero tensor.
            # src is already prenet-applied and position-embedded: [batch, seq_len, model_dim]
            batch_size = src.shape[0]
            # take most recent source token as starting target (real data)
            output = src[:, -1:, :]
            tgt_mask = torch.triu(
                torch.ones(1, 1, dtype=torch.bool, device=src.device), diagonal=1
            )
            from torch.utils.checkpoint import checkpoint as _cp

            for layer in self.layers:
                if self.checkpointing and self.training:
                    # checkpoint requires positional args only
                    output = _cp(
                        lambda o, m, mask: layer(o, m, tgt_mask=mask),
                        output,
                        src,
                        tgt_mask,
                    )
                else:
                    output = layer(output, src, tgt_mask=tgt_mask)

            pred_emb = self.postnet(output.squeeze(1))
            return pred_emb

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
