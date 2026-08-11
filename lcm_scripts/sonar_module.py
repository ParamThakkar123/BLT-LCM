"""
Lightweight SONAR-like encoder/decoder implemented in PyTorch.

This is a compact, self-contained implementation inspired by the SONAR repo but
designed to run without the original dependencies. It provides a fixed-size
sentence embedding (embed_dim=1024) and a decoder that can reconstruct UTF-8
bytes from the embedding. The implementation operates on UTF-8 bytes with a
small special-token offset to reserve ids for special tokens.

This is intended as a research-grade, fully working replacement when the
original SONAR package cannot be imported. It is not a drop-in replica of
Meta's SONAR but follows the same encoder/decoder bottleneck idea.
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# Vocabulary: reserve 0..3 for special tokens, bytes map to 4..259
OFFSET = 4
VOCAB_SIZE = 256 + OFFSET
PAD = 0
BOS = 1
EOT = 2


def text_to_byte_tokens(text: str) -> List[int]:
    return [b + OFFSET for b in text.encode("utf-8", errors="replace")]


def byte_tokens_to_text(tokens: List[int]) -> str:
    byte_vals = [t - OFFSET for t in tokens if OFFSET <= t < OFFSET + 256]
    return bytes(byte_vals).decode("utf-8", errors="replace")


def causal_mask(seq_len: int, device) -> torch.Tensor:
    """Boolean [T, T] causal mask (True = disallowed) for a Transformer decoder.

    Position k may attend only to positions <= k. Without this, teacher-forced
    training lets each target position see future tokens and the decoder learns
    to copy, which fails at autoregressive inference.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=512, n_layers=6, n_heads=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, tokens, mask=None):
        # tokens: [batch, seq]
        x = self.tok_emb(tokens)
        # transformer expects src_key_padding_mask: True for padded positions
        if mask is not None:
            # mask: [batch, seq] where 1 means real token
            key_padding = ~mask.bool()
        else:
            key_padding = None
        out = self.encoder(x, src_key_padding_mask=key_padding)
        return out


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, emb_dim=512, n_layers=6, n_heads=8):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output = nn.Linear(emb_dim, vocab_size, bias=True)

    def forward(self, tgt_tokens, memory, tgt_mask=None, memory_key_padding_mask=None):
        # tgt_tokens: [batch, tgt_len]
        t = self.tok_emb(tgt_tokens)
        out = self.decoder(
            t,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output(out)
        return logits


class SonarLite(nn.Module):
    """Encoder-decoder with fixed-size bottleneck embedding.

    Interface:
      - encode_sentences(list[str]) -> Tensor[batch, embed_dim]
      - decode_embeddings(Tensor[batch, embed_dim]) -> list[str]
    """

    def __init__(
        self,
        embed_dim=1024,
        token_emb_dim=512,
        enc_layers=6,
        dec_layers=6,
        n_heads=8,
        max_decode_len=256,
        device=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb_dim = token_emb_dim
        self.max_decode_len = max_decode_len
        from device_utils import report_device

        self.device = str(
            report_device(device, label="SonarLite", warn_cpu=False)
        )

        self.encoder = SimpleTransformerEncoder(
            vocab_size=VOCAB_SIZE,
            emb_dim=token_emb_dim,
            n_layers=enc_layers,
            n_heads=n_heads,
        )
        self.pool = nn.Linear(token_emb_dim, embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)

        # Decoder: maps embedding (as memory) to byte sequence
        self.decoder_proj = nn.Linear(embed_dim, token_emb_dim)
        self.decoder = SimpleTransformerDecoder(
            vocab_size=VOCAB_SIZE,
            emb_dim=token_emb_dim,
            n_layers=dec_layers,
            n_heads=n_heads,
        )

        # running stats for robust scaler (updated during training if enabled)
        self.register_buffer("robust_median", torch.zeros(embed_dim))
        self.register_buffer("robust_iqr", torch.ones(embed_dim))
        self.robust_enabled = False

    def encode_tokens(self, token_tensor, mask):
        # token_tensor: [batch, seq]
        enc_out = self.encoder(token_tensor, mask)
        # mean pool over valid positions
        mask_f = mask.float()
        lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        summed = (enc_out * mask_f.unsqueeze(-1)).sum(dim=1)
        pooled = summed / lengths
        emb = self.pool(pooled)
        emb = self.post_norm(emb)
        return emb

    MIN_ROBUST_BATCH = 8

    def normalize_bottleneck(self, emb: torch.Tensor, eps: float = 1e-6):
        """Robust scaling (median / IQR) estimated across the batch dimension.

        Returns normalized embeddings and the (median, iqr) used.

        Quartiles come from ``torch.quantile`` rather than ``kthvalue(n//4)``:
        the latter is a crude order statistic that, for small batches, can
        return the same element for Q1 and Q3, collapsing the IQR to ``eps`` and
        blowing up the division. Batches below ``MIN_ROBUST_BATCH`` cannot
        support a meaningful quartile estimate at all, so they leave the
        embeddings untouched instead of scaling by noise.
        """
        # emb: [batch, embed_dim]
        if emb.size(0) < self.MIN_ROBUST_BATCH:
            median = torch.zeros(emb.size(1), device=emb.device, dtype=emb.dtype)
            iqr = torch.ones(emb.size(1), device=emb.device, dtype=emb.dtype)
            return emb, median, iqr
        q = torch.quantile(
            emb.detach().float(),
            torch.tensor([0.25, 0.5, 0.75], device=emb.device),
            dim=0,
        )
        median = q[1].to(emb.dtype)
        iqr = (q[2] - q[0]).clamp_min(eps).to(emb.dtype)
        normalized = (emb - median.unsqueeze(0)) / iqr.unsqueeze(0)
        return normalized, median, iqr

    def apply_running_robust(self, emb: torch.Tensor):
        """Apply stored median/IQR (from buffers) to emb."""
        return (emb - self.robust_median.unsqueeze(0)) / (
            self.robust_iqr.unsqueeze(0).clamp_min(1e-6)
        )

    @torch.no_grad()
    def fit_robust(self, samples: torch.Tensor, eps: float = 1e-6):
        """Fit the robust scaler once, offline, on a large sample of embeddings.

        This is what the LCM paper does: the scaler is estimated from randomly
        sampled vectors spanning corpora and domains, then frozen. Prefer this
        over per-batch estimation.
        """
        if samples.dim() != 2 or samples.shape[0] < self.MIN_ROBUST_BATCH:
            raise ValueError(
                f"need [N, dim] with N >= {self.MIN_ROBUST_BATCH} to fit a robust scaler"
            )
        q = torch.quantile(
            samples.float(),
            torch.tensor([0.25, 0.5, 0.75], device=samples.device),
            dim=0,
        )
        self.robust_median.copy_(q[1])
        self.robust_iqr.copy_((q[2] - q[0]).clamp_min(eps))
        self.robust_enabled = True
        return self

    def update_running_robust(
        self, median: torch.Tensor, iqr: torch.Tensor, momentum: float = 0.9
    ):
        """Blend batch statistics into the stored median/IQR.

        ``momentum`` is the weight kept on the existing statistics, so the
        default 0.9 is a genuine running estimate. Passing 0.0 replaces them
        with the current batch's, which makes the buffers track only the most
        recent batch — that was the previous behaviour and it meant the stored
        "running" statistics were never actually accumulated.
        """
        if momentum <= 0.0:
            self.robust_median.copy_(median)
            self.robust_iqr.copy_(iqr)
        else:
            self.robust_median.mul_(momentum).add_(median * (1 - momentum))
            self.robust_iqr.mul_(momentum).add_(iqr * (1 - momentum))

    def encode_sentences(self, sentences: List[str], max_len=256) -> torch.Tensor:
        toks = [text_to_byte_tokens(s)[:max_len] for s in sentences]
        lens = [len(t) for t in toks]
        maxl = max(lens) if lens else 0
        batch = len(toks)
        token_tensor = torch.full(
            (batch, maxl), PAD, dtype=torch.long, device=self.device
        )
        mask = torch.zeros((batch, maxl), dtype=torch.bool, device=self.device)
        for i, t in enumerate(toks):
            if len(t) > 0:
                token_tensor[i, : len(t)] = torch.tensor(
                    t, dtype=torch.long, device=self.device
                )
                mask[i, : len(t)] = 1
        with torch.no_grad():
            emb = self.encode_tokens(token_tensor, mask)
        return emb

    def forward(self, token_tensor, mask, tgt_in=None, apply_robust=False):
        """
        Full forward: encode tokens to bottleneck and optionally decode with tgt_in.

        Returns: logits (if tgt_in provided) and embedding tensor [batch, embed_dim]
        """
        emb = self.encode_tokens(token_tensor, mask)

        # optionally apply robust scaler
        if apply_robust:
            emb_normed, median, iqr = self.normalize_bottleneck(emb)
            # update running stats (replace)
            with torch.no_grad():
                self.update_running_robust(median, iqr, momentum=0.0)
            emb_for_dec = emb_normed
        elif self.robust_enabled:
            emb_for_dec = self.apply_running_robust(emb)
        else:
            emb_for_dec = emb

        logits = None
        if tgt_in is not None:
            # prepare memory for decoder
            memory = self.decoder_proj(emb_for_dec).unsqueeze(1)
            # Causal mask so teacher forcing cannot attend to future tokens.
            tgt_mask = causal_mask(tgt_in.size(1), tgt_in.device)
            logits = self.decoder(tgt_in, memory, tgt_mask=tgt_mask)

        return logits, emb

    def decode_embeddings(
        self, embeddings: torch.Tensor, max_len: int = None
    ) -> List[str]:
        max_len = max_len or self.max_decode_len
        batch = embeddings.shape[0]
        device = embeddings.device

        # Prepare memory for decoder: project embedding to token_emb_dim and unsqueeze as memory length 1
        memory = self.decoder_proj(embeddings).unsqueeze(1)  # [batch, 1, token_emb_dim]

        # Greedy decode
        cur_tokens = torch.full((batch, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        outputs = [[] for _ in range(batch)]

        for step in range(max_len):
            # Causal mask keeps decoding consistent with masked training.
            tgt_mask = causal_mask(cur_tokens.size(1), device)
            logits = self.decoder(cur_tokens, memory, tgt_mask=tgt_mask)  # [b, seq, vocab]
            next_logits = logits[:, -1, :]
            next_tok = next_logits.argmax(dim=-1)
            cur_tokens = torch.cat([cur_tokens, next_tok.unsqueeze(1)], dim=1)

            for i in range(batch):
                tok = int(next_tok[i].item())
                if not finished[i]:
                    if tok == EOT:
                        finished[i] = True
                    else:
                        outputs[i].append(tok)

            if finished.all():
                break

        texts = [byte_tokens_to_text(seq) for seq in outputs]
        return texts


if __name__ == "__main__":
    # quick smoke test -- pinned to CPU, so CPU is not a fallback here.
    # SonarLite reports the device itself on construction.
    model = SonarLite(device="cpu")
    sents = ["हॅलो, तुम्हाला कसे आहात?", "ही एक चाचणी वाक्य आहे."]
    emb = model.encode_sentences(sents)
    print("Embeddings:", emb.shape)
    dec = model.decode_embeddings(emb, max_len=64)
    # Printing Unicode to Windows consoles can raise a UnicodeEncodeError
    # if the terminal encoding cannot represent the characters. Use
    # sys.stdout.buffer to write UTF-8 bytes as a robust fallback.
    import sys

    try:
        print("Decoded:", dec)
    except UnicodeEncodeError:
        for i, d in enumerate(dec):
            try:
                sys.stdout.buffer.write(f"Decoded[{i}]: ".encode("utf-8"))
                sys.stdout.buffer.write((d + "\n").encode("utf-8"))
            except Exception:
                # Last resort: print a safe repr
                print(f"Decoded[{i}]: {repr(d)}")
