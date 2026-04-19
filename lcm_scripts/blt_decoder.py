"""
BLT Decoder: reconstructs UTF-8 byte sequences from BLT sentence embeddings.

The BLT encoder (ByteEntropyModel + entropy-based patching) compresses a
sentence into a single 256-dim vector via mean-pooled patch hidden states.
This decoder reverses that process using a Transformer with cross-attention
to the projected embedding, autoregressively generating byte tokens.

Training uses teacher-forced cross-entropy against the original byte sequence.
"""

import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from run_blt_patching import VOCAB_SIZE, OFFSET, text_to_byte_tokens, byte_tokens_to_text

PAD = 0
BOS = 1
EOT = 2


class BLTDecoder(nn.Module):
    """Autoregressive byte-level decoder from BLT sentence embeddings.

    Architecture:
        embedding [embed_dim]
            -> Linear(embed_dim, dec_dim)   # project to decoder width
            -> unsqueeze as memory [B, 1, dec_dim]
            -> Transformer Decoder (cross-attention to memory)
            -> Linear(dec_dim, vocab_size)  # byte logits

    During inference, greedy decoding generates one byte token at a time
    until EOT or max_decode_len is reached.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        dec_dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        vocab_size: int = VOCAB_SIZE,
        max_decode_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dec_dim = dec_dim
        self.vocab_size = vocab_size
        self.max_decode_len = max_decode_len

        self.embedding_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, dec_dim),
            nn.GELU(),
            nn.Linear(dec_dim, dec_dim),
        )

        self.tok_emb = nn.Embedding(vocab_size, dec_dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_decode_len, dec_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dec_dim,
            nhead=n_heads,
            dim_feedforward=dec_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(dec_dim)
        self.output_head = nn.Linear(dec_dim, vocab_size)

    def forward(
        self,
        embeddings: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            embeddings: [B, embed_dim] sentence embeddings from BLT encoder.
            tgt_tokens: [B, T] target byte token ids (BOS-prefixed).
            tgt_padding_mask: [B, T] True for padded positions.

        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = tgt_tokens.shape
        device = tgt_tokens.device

        memory = self.embedding_proj(embeddings).unsqueeze(1)  # [B, 1, dec_dim]

        positions = torch.arange(T, device=device).unsqueeze(0)
        tgt = self.tok_emb(tgt_tokens) + self.pos_emb(positions)

        tgt_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        )

        out = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )

        logits = self.output_head(self.output_norm(out))
        return logits

    @torch.no_grad()
    def decode(
        self,
        embeddings: torch.Tensor,
        max_len: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[str]:
        """Greedy autoregressive decoding from embeddings to text.

        Args:
            embeddings: [B, embed_dim] sentence embeddings.
            max_len: maximum number of byte tokens to generate.
            temperature: softmax temperature (1.0 = greedy argmax).

        Returns:
            list of decoded strings.
        """
        max_len = max_len or self.max_decode_len
        B = embeddings.shape[0]
        device = embeddings.device

        memory = self.embedding_proj(embeddings).unsqueeze(1)

        cur_tokens = torch.full((B, 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        outputs: list[list[int]] = [[] for _ in range(B)]

        for step in range(max_len):
            T = cur_tokens.shape[1]
            positions = torch.arange(T, device=device).unsqueeze(0)
            tgt = self.tok_emb(cur_tokens) + self.pos_emb(positions)

            tgt_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
            )

            out = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            last_logits = self.output_head(self.output_norm(out[:, -1, :]))

            if temperature <= 0 or temperature == 1.0:
                next_tok = last_logits.argmax(dim=-1)
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)

            cur_tokens = torch.cat([cur_tokens, next_tok.unsqueeze(1)], dim=1)

            for i in range(B):
                tok = int(next_tok[i].item())
                if not finished[i]:
                    if tok == EOT:
                        finished[i] = True
                    else:
                        outputs[i].append(tok)

            if finished.all():
                break

        return [byte_tokens_to_text(seq) for seq in outputs]


def prepare_decoder_data(sentences: list[str], max_byte_len: int = 512):
    """Convert sentences to (byte_input, byte_target) pairs for decoder training.

    For each sentence:
      input  = [BOS] + byte_tokens            (teacher-forced input)
      target = byte_tokens + [EOT]             (what to predict)
    Both are padded to the same length within a batch.
    """
    inputs, targets = [], []
    for sent in sentences:
        tokens = text_to_byte_tokens(sent)[:max_byte_len]
        if len(tokens) == 0:
            continue
        inputs.append([BOS] + tokens)
        targets.append(tokens + [EOT])
    return inputs, targets


def collate_decoder_batch(batch):
    """Collate (embedding, input_tokens, target_tokens) tuples."""
    embs, inputs, targets = zip(*batch)
    B = len(embs)
    max_len = max(len(t) for t in inputs)
    emb_tensor = torch.stack(embs)
    inp_tensor = torch.full((B, max_len), PAD, dtype=torch.long)
    tgt_tensor = torch.full((B, max_len), -100, dtype=torch.long)  # -100 = ignore
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)
    for i in range(B):
        L = len(inputs[i])
        inp_tensor[i, :L] = torch.tensor(inputs[i], dtype=torch.long)
        tgt_tensor[i, :L] = torch.tensor(targets[i], dtype=torch.long)
        pad_mask[i, :L] = False
    return emb_tensor, inp_tensor, tgt_tensor, pad_mask


class DecoderDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings: list[torch.Tensor], input_seqs, target_seqs):
        assert len(embeddings) == len(input_seqs) == len(target_seqs)
        self.embeddings = embeddings
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.input_seqs[idx], self.target_seqs[idx]


def train_decoder(
    decoder: BLTDecoder,
    blt_loader,
    sentences: list[str],
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    max_byte_len: int = 512,
    device: str = "cuda",
    log_every: int = 50,
    save_path: str = "lcm_models/blt_decoder.pth",
):
    """Train the BLT decoder on (embedding, sentence) pairs.

    1. Encode all sentences with BLTLoader to get embeddings.
    2. Build teacher-forced (input, target) byte-token pairs.
    3. Train with cross-entropy loss.
    """
    import time

    print(f"Encoding {len(sentences)} sentences with BLT encoder...")
    enc_batch = 64
    embeddings = []
    for i in range(0, len(sentences), enc_batch):
        batch = sentences[i : i + enc_batch]
        emb_batch = blt_loader.encode_sentences_batch(batch)
        embeddings.extend([e.cpu() for e in emb_batch])

    inputs_all, targets_all = prepare_decoder_data(sentences, max_byte_len)

    valid_embs, valid_inputs, valid_targets = [], [], []
    emb_idx = 0
    for sent in sentences:
        tokens = text_to_byte_tokens(sent)[:max_byte_len]
        if len(tokens) == 0:
            emb_idx += 1
            continue
        valid_embs.append(embeddings[emb_idx])
        emb_idx += 1
    valid_inputs = inputs_all
    valid_targets = targets_all
    valid_embs = valid_embs[: len(valid_inputs)]

    dataset = DecoderDataset(valid_embs, valid_inputs, valid_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_decoder_batch
    )

    decoder = decoder.to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    best_loss = float("inf")
    for epoch in range(epochs):
        decoder.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        for step, (emb, inp, tgt, pad_mask) in enumerate(loader):
            emb = emb.to(device)
            inp = inp.to(device)
            tgt = tgt.to(device)
            pad_mask = pad_mask.to(device)

            logits = decoder(emb, inp, tgt_padding_mask=pad_mask)
            loss = F.cross_entropy(
                logits.view(-1, decoder.vocab_size), tgt.view(-1), ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % log_every == 0:
                avg = total_loss / n_batches
                print(
                    f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(loader)} | "
                    f"Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - start
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch+1}/{epochs} DONE | "
            f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": decoder.state_dict(),
                        "config": {
                            "embed_dim": decoder.embed_dim,
                            "dec_dim": decoder.dec_dim,
                            "n_layers": len(decoder.decoder.layers),
                            "n_heads": decoder.decoder.layers[0].self_attn.num_heads,
                            "vocab_size": decoder.vocab_size,
                            "max_decode_len": decoder.max_decode_len,
                        },
                    },
                    save_path,
                )
                print(f"  Saved best decoder to {save_path}")

    return decoder


def load_decoder(
    checkpoint_path: str, device: str = "cpu"
) -> BLTDecoder:
    """Load a trained BLTDecoder from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    decoder = BLTDecoder(
        embed_dim=cfg.get("embed_dim", 256),
        dec_dim=cfg.get("dec_dim", 512),
        n_layers=cfg.get("n_layers", 6),
        n_heads=cfg.get("n_heads", 8),
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        max_decode_len=cfg.get("max_decode_len", 512),
    ).to(device)
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.eval()
    return decoder


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train BLT byte-level decoder")
    parser.add_argument(
        "--entropy_model",
        type=str,
        default="patching_scratch/entropy_model_marathi.pt",
    )
    parser.add_argument("--num_sentences", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dec_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_byte_len", type=int, default=512)
    parser.add_argument(
        "--save_path", type=str, default="lcm_models/blt_decoder.pth"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fraction", type=float, default=0.25, help="Fraction of BhashaSetu to use"
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm

    print("Loading BhashaSetu dataset...")
    ds = load_dataset("ParamTh/BhashaSetu", split="train", streaming=True)
    sentences = []
    for row in tqdm(ds, desc="Loading", total=args.num_sentences):
        text = row.get("marathi", "")
        if text and len(text.strip()) > 5:
            sentences.append(text.strip())
        if len(sentences) >= args.num_sentences:
            break
    print(f"Loaded {len(sentences)} sentences")

    from blt_loader import BLTLoader

    blt = BLTLoader(entropy_model_path=args.entropy_model, device=args.device)

    decoder = BLTDecoder(
        embed_dim=blt.model.tok_embeddings.weight.shape[1],  # 256
        dec_dim=args.dec_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_decode_len=args.max_byte_len,
    )
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {total_params:,}")

    trained = train_decoder(
        decoder,
        blt,
        sentences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_byte_len=args.max_byte_len,
        device=args.device,
        save_path=args.save_path,
    )

    print("\nTesting reconstruction on sample sentences...")
    trained.eval()
    test_sents = sentences[:5]
    test_embs = blt.encode_sentences_batch(test_sents)
    test_embs_tensor = torch.stack(test_embs).to(args.device)
    decoded = trained.decode(test_embs_tensor)
    for orig, dec in zip(test_sents, decoded):
        try:
            print(f"  Original: {orig[:80]}")
            print(f"  Decoded:  {dec[:80]}")
            print()
        except UnicodeEncodeError:
            import sys
            sys.stdout.buffer.write(f"  Original: {orig[:80]}\n".encode("utf-8"))
            sys.stdout.buffer.write(f"  Decoded:  {dec[:80]}\n\n".encode("utf-8"))
