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

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from device_utils import report_device
from checkpoint_utils import ResumableLoader, TrainingCheckpointer

from run_blt_patching import (
    VOCAB_SIZE,
    OFFSET,
    text_to_byte_tokens,
    byte_tokens_to_text,
)

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
            tgt_padding_mask: [B, T] True for padded positions. Only needed for
                LEFT-padded targets. With right padding it is redundant — the
                causal mask already stops a real position from attending to the
                padding that follows it — and passing it anyway is expensive:
                ``nn.MultiheadAttention`` then merges the two masks into a dense
                ``[B * n_heads, T, T]`` float tensor per layer and loses the
                fused attention kernel. Leave it None for right-padded batches.

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
        # Never request positions beyond the position-embedding table.
        max_len = min(max_len, self.max_decode_len)
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


class SentenceReconDataset(torch.utils.data.Dataset):
    """Holds byte tokens + teacher-forced byte (input, target) pairs.

    Concepts are (re)encoded on the fly during training so the pooler stays in
    the autograd graph; we therefore keep the sentence's byte tokens, not a
    frozen embedding, alongside the byte-token supervision.

    Patch boundaries depend only on the (fixed) text and the frozen entropy
    model, so they are computed once by :meth:`attach_patch_specs` and carried
    through the loader instead of being recomputed every epoch.
    """

    def __init__(self, sentences: list[str], max_byte_len: int = 512):
        self.items = []
        for sent in sentences:
            tokens = text_to_byte_tokens(sent)[:max_byte_len]
            if len(tokens) == 0:
                continue
            self.items.append((tokens, [BOS] + tokens, tokens + [EOT]))
        self.patch_specs: Optional[list] = None

    @property
    def token_lists(self) -> list[list[int]]:
        return [tokens for tokens, _, _ in self.items]

    def attach_patch_specs(self, specs: list):
        """Cache one ``(boundaries, patch_lengths)`` pair per retained sentence."""
        if len(specs) != len(self.items):
            raise ValueError(
                f"expected {len(self.items)} patch specs, got {len(specs)}"
            )
        self.patch_specs = specs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tokens, inp, tgt = self.items[idx]
        spec = self.patch_specs[idx] if self.patch_specs is not None else None
        return tokens, inp, tgt, spec


def collate_sentences(batch):
    """Collate (byte_tokens, input_tokens, target_tokens, patch_spec) tuples.

    Returns the byte-token lists and their cached patch specs (for batched
    on-the-fly encoding) plus padded input / target tensors and a padding mask.
    """
    token_lists, inputs, targets, specs = zip(*batch)
    B = len(token_lists)
    max_len = max(len(t) for t in inputs)
    inp_tensor = torch.full((B, max_len), PAD, dtype=torch.long)
    tgt_tensor = torch.full((B, max_len), -100, dtype=torch.long)  # -100 = ignore
    pad_mask = torch.ones((B, max_len), dtype=torch.bool)
    for i in range(B):
        L = len(inputs[i])
        inp_tensor[i, :L] = torch.tensor(inputs[i], dtype=torch.long)
        tgt_tensor[i, :L] = torch.tensor(targets[i], dtype=torch.long)
        pad_mask[i, :L] = False
    return list(token_lists), inp_tensor, tgt_tensor, pad_mask, list(specs)


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
    train_pooler: bool = True,
    pooler_save_path: str = "lcm_models/blt_pooler.pth",
    threshold: float = 1.335,
    resume: str = "auto",
    fingerprint: Optional[str] = None,
    save_interval_steps: int = 200,
    save_interval_seconds: float = 0.0,
    max_checkpoints: int = 5,
    seed: int = 42,
    amp: bool = False,
):
    """Jointly train the cross-attention pooler and the byte decoder.

    This is a BLT-style local autoencoder objective: the concept vector must be
    sufficient to reconstruct the original bytes. The byte backbone (and the
    entropy boundaries) stay FROZEN; gradients flow only into the pooler and the
    decoder, so patch boundaries never drift during training.

    Concepts are re-encoded every step via ``encode_sentences_differentiable``
    so the pooler remains in the autograd graph. When ``train_pooler`` is False
    the pooler is frozen and only the decoder learns (original behavior).
    """
    import time

    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    if amp and amp_device != "cuda":
        print("  [amp] no CUDA device; running in fp32")
        amp = False

    # A teacher-forced sequence is [BOS] + up to max_byte_len bytes, so it must
    # fit within the decoder's position-embedding table.
    max_byte_len = min(max_byte_len, decoder.max_decode_len - 1)
    dataset = SentenceReconDataset(sentences, max_byte_len=max_byte_len)

    # Patch boundaries come from the FROZEN entropy model and fixed text, so
    # they are identical on every epoch and every step. Computing them once here
    # takes the entropy model out of the training loop entirely.
    t0 = time.time()
    print(f"Precomputing patch boundaries for {len(dataset)} sentences...")
    dataset.attach_patch_specs(
        blt_loader.compute_patch_specs(dataset.token_lists, threshold=threshold)
    )
    print(f"  done in {time.time() - t0:.1f}s")

    loader = ResumableLoader(
        dataset,
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
        collate_fn=collate_sentences,
    )
    print(f"Training on {len(dataset)} sentences (pooler {'ON' if train_pooler else 'frozen'})")

    decoder = decoder.to(device)

    # Parameter groups: decoder always; pooler only if train_pooler.
    params = list(decoder.parameters())
    if train_pooler:
        blt_loader.pooler.to(device)
        blt_loader.pooler.train()
        for p in blt_loader.pooler.parameters():
            p.requires_grad_(True)
        params += list(blt_loader.pooler.parameters())
    else:
        blt_loader.pooler.eval()
        for p in blt_loader.pooler.parameters():
            p.requires_grad_(False)

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = epochs * max(len(loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1)
    )

    # The pooler is trained jointly with the decoder, so both — plus the
    # scheduler, whose cosine position depends on total steps taken — have to be
    # checkpointed together for a resumed run to continue the same trajectory.
    trainable = nn.ModuleDict({"decoder": decoder, "pooler": blt_loader.pooler})
    ckpt_dir = os.path.dirname(save_path) or "lcm_models"
    ckpt = TrainingCheckpointer(
        ckpt_dir,
        prefix=os.path.splitext(os.path.basename(save_path))[0],
        fingerprint=fingerprint,
        max_keep=max_checkpoints,
        save_interval_steps=save_interval_steps,
        save_interval_seconds=save_interval_seconds,
    )
    rp = ckpt.restore(
        ckpt.load(resume, map_location=device), trainable, optimizer, scheduler
    )
    best_loss = rp.best_score if rp.best_score is not None else float("inf")
    global_step = rp.global_step
    if rp.resumed:
        print(
            f"  [resume] continuing at epoch {rp.start_epoch + 1}/{epochs}, "
            f"batch {rp.start_batch} (best loss so far {best_loss:.4f})"
        )

    for epoch in range(rp.start_epoch, epochs):
        decoder.train()
        if train_pooler:
            blt_loader.pooler.train()
        total_loss = 0.0
        n_batches = 0
        start = time.time()

        skip = rp.batches_to_skip(epoch)
        for step, (batch_tokens, inp, tgt, _pad_mask, specs) in loader.epoch(
            epoch, skip=skip
        ):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            # Re-encode concepts in-graph so the pooler receives gradients, as a
            # single batched forward. Boundaries were precomputed above, so the
            # frozen entropy model never runs here.
            # bf16 needs no GradScaler, and cross_entropy is autocast-promoted
            # back to fp32, so the loss stays numerically well behaved.
            with torch.autocast(
                device_type=amp_device, dtype=torch.bfloat16, enabled=amp
            ):
                emb = blt_loader.encode_tokens_differentiable(
                    batch_tokens, threshold=threshold, patch_specs=specs
                )  # [B, dim]

                # No tgt_padding_mask: `inp` is right-padded and the decoder's
                # mask is causal, so real positions never see the padding, and
                # padded positions are excluded from the loss by ignore_index.
                # Passing it would force a dense [B * heads, T, T] merged mask
                # per layer and disable the fused attention kernel.
                logits = decoder(emb, inp)
                loss = F.cross_entropy(
                    logits.view(-1, decoder.vocab_size), tgt.view(-1), ignore_index=-100
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            ckpt.maybe_save(
                trainable,
                optimizer,
                scheduler,
                epoch=epoch,
                batch_in_epoch=step,
                global_step=global_step,
                best_score=best_loss,
            )

            if (step + 1) % log_every == 0:
                avg = total_loss / n_batches
                print(
                    f"  Epoch {epoch + 1}/{epochs} | Step {step + 1}/{len(loader)} | "
                    f"Loss: {avg:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        elapsed = time.time() - start
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"  Epoch {epoch + 1}/{epochs} DONE | "
            f"Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
        )
        # End-of-epoch snapshot goes into the rolling `_last` checkpoint only.
        # No per-epoch `_epoch{N}` files: the run keeps exactly two checkpoints,
        # `_last` (the resume target) and `_best`.
        ckpt.save(
            trainable,
            optimizer,
            scheduler,
            epoch=epoch,
            batch_in_epoch=0,
            epoch_completed=True,
            global_step=global_step,
            best_score=best_loss,
        )
        print(f"  [checkpoint] epoch {epoch + 1} -> {ckpt.last_path}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_path:
                # `save_path` keeps the standalone decoder format that
                # `load_decoder` and every eval script expect; the resumable
                # state lives alongside it under the same prefix.
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": decoder.state_dict(),
                        "pooler_state_dict": blt_loader.pooler.state_dict(),
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
            # Persist the learned pooler as a sidecar so BLTLoader (used by
            # train_lcm_blt / eval) can pick up the improved concept space.
            if train_pooler and pooler_save_path:
                blt_loader.save_pooler(pooler_save_path)
                print(f"  Saved learned pooler to {pooler_save_path}")

    blt_loader.pooler.eval()
    return decoder


def load_decoder(checkpoint_path: str, device: str = "cpu") -> BLTDecoder:
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
    parser.add_argument("--save_path", type=str, default="lcm_models/blt_decoder.pth")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--fraction", type=float, default=0.25, help="Fraction of BhashaSetu to use"
    )
    parser.add_argument(
        "--no_train_pooler",
        action="store_true",
        help="Freeze the cross-attention pooler and train only the decoder",
    )
    parser.add_argument(
        "--pooler_save_path",
        type=str,
        default="lcm_models/blt_pooler.pth",
        help="Where to save the learned pooler sidecar (loaded by BLTLoader)",
    )
    parser.add_argument("--threshold", type=float, default=1.335)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run the forward/backward in bfloat16 autocast on CUDA. Roughly "
        "halves step time on Ampere and newer at a small numerical cost; the "
        "loss itself is still accumulated in fp32.",
    )
    parser.add_argument(
        "--encoder_dim", type=int, default=256, help="Local encoder byte width"
    )
    parser.add_argument("--encoder_layers", type=int, default=1)
    parser.add_argument("--latent_layers", type=int, default=4)
    parser.add_argument("--encoder_heads", type=int, default=8)
    parser.add_argument(
        "--encoder_window",
        type=int,
        default=512,
        help="Local block-causal attention window in bytes (BLT §3.2)",
    )
    parser.add_argument(
        "--hash_vocab_size",
        type=int,
        default=None,
        help="Hash n-gram table size per n. Memory is "
        "len(ngram_sizes) * hash_vocab_size * encoder_dim * 4 bytes, and AdamW "
        "quadruples it. The paper uses 500000 over n=3..8, which is ~11.4 GiB "
        "of optimizer-committed memory at encoder_dim 256 and will OOM a 16 GiB "
        "GPU. Default is 100000.",
    )
    parser.add_argument(
        "--ngram_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Byte n-gram sizes for hash embeddings (default 3 4 5; the paper "
        "uses 3 4 5 6 7 8).",
    )
    parser.add_argument(
        "--no_hash_ngrams",
        action="store_true",
        help="Disable hash n-gram embeddings entirely (BLT Table 8 ablation). "
        "Removes their memory cost at a real quality cost.",
    )
    parser.add_argument(
        "--concept_dim",
        type=int,
        default=1024,
        help="Concept dimension (default 1024, matching SONAR / the LCM). The "
        "pooler learns to project pooled byte features to this dimension.",
    )
    from checkpoint_utils import add_resume_args, config_fingerprint, seed_everything

    add_resume_args(parser, default_interval_steps=200)
    args = parser.parse_args()

    report_device(args.device)
    seed_everything(args.ckpt_seed)
    fingerprint = config_fingerprint(args)

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

    from blt_local_encoder import DEFAULT_HASH_VOCAB, DEFAULT_NGRAM_SIZES

    blt = BLTLoader(
        entropy_model_path=args.entropy_model,
        device=args.device,
        concept_dim=args.concept_dim,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        latent_layers=args.latent_layers,
        encoder_heads=args.encoder_heads,
        encoder_window=args.encoder_window,
        use_hash_ngrams=not args.no_hash_ngrams,
        ngram_sizes=tuple(args.ngram_sizes) if args.ngram_sizes else DEFAULT_NGRAM_SIZES,
        hash_vocab_size=(
            args.hash_vocab_size
            if args.hash_vocab_size is not None
            else DEFAULT_HASH_VOCAB
        ),
    )

    decoder = BLTDecoder(
        embed_dim=blt.dim,  # concept dimension (default 1024)
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
        train_pooler=not args.no_train_pooler,
        pooler_save_path=args.pooler_save_path,
        threshold=args.threshold,
        resume=args.resume,
        fingerprint=fingerprint,
        save_interval_steps=args.save_interval_steps,
        save_interval_seconds=args.save_interval_seconds,
        max_checkpoints=args.max_checkpoints,
        seed=args.ckpt_seed,
        amp=args.amp,
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
