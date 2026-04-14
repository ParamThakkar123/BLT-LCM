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
        self, embed_dim=1024, model_dim=2048, n_layers=24, n_heads=16, max_seq_len=128
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.prenet = PreNet(embed_dim, model_dim)
        self.postnet = PostNet(model_dim, embed_dim)

        # Position embeddings
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = TransformerDecoder(decoder_layer, num_layers=n_layers)

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
            tgt = self.prenet(tgt_embs)  # [batch, 1, model_dim]
            tgt_pos = torch.arange(seq_len, seq_len + 1, device=src.device).unsqueeze(0)
            tgt = tgt + self.pos_emb(tgt_pos)

            # Causal mask for decoder
            tgt_mask = torch.triu(
                torch.ones(1, 1, dtype=torch.bool, device=src.device), diagonal=1
            )

            # Decode
            output = self.transformer(
                tgt, src, tgt_mask=tgt_mask
            )  # [batch, 1, model_dim]

            # PostNet
            pred_emb = self.postnet(output.squeeze(1))  # [batch, embed_dim]
            return pred_emb
        else:
            # Inference: predict next
            # For simplicity, assume single step
            batch_size = src.shape[0]
            dummy_tgt = torch.zeros(batch_size, 1, self.model_dim, device=src.device)
            tgt_mask = torch.triu(
                torch.ones(1, 1, dtype=torch.bool, device=src.device), diagonal=1
            )
            output = self.transformer(dummy_tgt, src, tgt_mask=tgt_mask)
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
