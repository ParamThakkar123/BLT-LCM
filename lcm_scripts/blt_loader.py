"""
BLT embedding loader for LCM.

Produces sentence-level concept embeddings the BLT way (Byte Latent Transformer,
Meta 2024, §3.2.2):

  1. The entropy model is used ONLY to place patch boundaries (via per-byte
     entropy of its next-byte distribution).
  2. The whole sentence is run through the local-encoder backbone to obtain
     byte *hidden states* (full-sentence context, not per-patch in isolation).
  3. Each patch's byte hidden states are pooled into one patch representation by
     a Perceiver-style cross-attention pooler (patch-masked; padding excluded).
  4. Patch representations are aggregated (mean) into a sentence embedding.

The entropy model's output *logits* are never used as patch/concept embeddings.
"""

import torch
import sys
import os
from typing import Optional

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from run_blt_patching import (
    text_to_byte_tokens,
    byte_tokens_to_text,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
)
from run_blt_patching import ByteEntropyModel
from blt_local_encoder import PatchCrossAttentionPooler


class BLTLoader:
    def __init__(
        self,
        entropy_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pooler_path: Optional[str] = None,
        concept_dim: int = 1024,
    ):
        self.device = device
        # Resolve path relative to the repo root when not absolute
        if not os.path.isabs(entropy_model_path) and not os.path.exists(
            entropy_model_path
        ):
            repo_root = os.path.join(os.path.dirname(__file__), "..")
            candidate = os.path.join(
                repo_root, "patching_scratch", os.path.basename(entropy_model_path)
            )
            if os.path.exists(candidate):
                entropy_model_path = candidate
        # Load the entropy model
        checkpoint = torch.load(
            entropy_model_path, map_location=device, weights_only=False
        )

        cfg = checkpoint.get("config", {})
        self.model = ByteEntropyModel(
            vocab_size=cfg.get("vocab_size", 260),
            dim=cfg.get("dim", 256),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            max_seqlen=cfg.get("max_seqlen", 512),
            ffn_dim_multiplier=cfg.get("ffn_dim_multiplier", 1.3),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.token_cache = {}
        self._decoder = None

        # Byte-encoder hidden width (256), distinct from the output concept dim.
        self.hidden_dim = self.model.dim
        # Output concept dimension. Default 1024 matches SONAR / the LCM
        # convention, so BLT and SONAR concepts live in the same space and are
        # directly comparable. A learned projection lifts pooled 256-d byte
        # features to this dimension (identity when concept_dim == hidden_dim).
        self.dim = concept_dim
        self.pooler_heads = int(cfg.get("pooler_heads", 4))

        # BLT-faithful patch pooling. The entropy model above decides *where*
        # patches begin; this module turns each patch's byte hidden states into
        # a patch representation via cross-attention (BLT §3.2.2) and projects
        # it to the concept dimension.
        self.pooler = PatchCrossAttentionPooler(
            self.hidden_dim, concept_dim=self.dim, n_heads=self.pooler_heads
        ).to(device)
        self.pooler.eval()
        # Load trained pooler weights if they were bundled with the checkpoint.
        if isinstance(checkpoint, dict) and checkpoint.get("pooler_state_dict"):
            try:
                self.pooler.load_state_dict(checkpoint["pooler_state_dict"])
            except Exception as e:  # pragma: no cover - defensive
                print(f"[BLTLoader] Could not load pooler weights: {e}")
        # A sidecar pooler file (from joint pooler+decoder training) takes
        # precedence, so the learned concept space is used by training/eval.
        if pooler_path and os.path.exists(pooler_path):
            try:
                self.load_pooler(pooler_path)
                print(f"[BLTLoader] Loaded learned pooler from {pooler_path}")
            except Exception as e:  # pragma: no cover - defensive
                print(f"[BLTLoader] Could not load pooler from {pooler_path}: {e}")

    @torch.no_grad()
    def _extract_patch_hidden(self, tokens_batch, threshold):
        """Frozen feature extraction (no gradient through the byte backbone).

        Returns a list (one entry per sentence) of lists of ``[Lp, dim]`` byte
        hidden-state tensors, one per patch. The entropy model is used only to
        place the boundaries; the hidden states come from ``return_hidden=True``.
        """
        max_seqlen = self.model.max_length
        per_sentence = []
        for tokens in tokens_batch:
            if len(tokens) == 0:
                per_sentence.append([])
                continue
            # RoPE cache is sized to max_seqlen; cap the byte sequence to fit.
            tokens = tokens[:max_seqlen]
            tokens_tensor = torch.tensor(
                [tokens], dtype=torch.long, device=self.device
            )
            # (1) Boundaries only — entropy of the next-byte distribution.
            entropies = compute_entropies_for_tokens(
                tokens_tensor, self.model, device=self.device
            )
            boundaries, patch_lengths = entropy_patch_sentence(
                entropies[0].tolist(), threshold
            )
            # (2) Byte hidden states for the whole sentence (full context).
            byte_hidden = self.model(tokens_tensor, return_hidden=True)[0]  # [L, dim]
            per_sentence.append(
                [byte_hidden[s : s + l] for s, l in zip(boundaries, patch_lengths)]
            )
        return per_sentence

    def _pool_per_sentence(self, per_sentence):
        """Pool each sentence's patch hidden states into one concept vector.

        Runs the (possibly trainable) pooler in the CURRENT autograd context, so
        gradients reach the pooler when it is in train mode and grad is enabled.
        ``per_sentence`` is the structure returned by ``_extract_patch_hidden``.
        """
        flat = [p for patches in per_sentence for p in patches]
        counts = [len(patches) for patches in per_sentence]

        patch_reps = None
        if flat:
            max_lp = max(p.shape[0] for p in flat)
            P = len(flat)
            # Byte hidden states are hidden_dim wide; the pooler projects the
            # pooled result to the concept dimension (self.dim).
            padded = torch.zeros(P, max_lp, self.hidden_dim, device=self.device)
            key_mask = torch.zeros(P, max_lp, dtype=torch.bool, device=self.device)
            for i, p in enumerate(flat):
                lp = p.shape[0]
                padded[i, :lp] = p
                key_mask[i, :lp] = True
            patch_reps = self.pooler(padded.float(), key_mask)  # [P, dim]

        embeddings = []
        index = 0
        for c in counts:
            if c > 0 and patch_reps is not None:
                embeddings.append(patch_reps[index : index + c].mean(dim=0))
            else:
                embeddings.append(torch.zeros(self.dim, device=self.device))
            index += c
        return embeddings

    @torch.no_grad()
    def encode_tokens_batch(self, tokens_batch, threshold=1.335):
        """Encode tokenized sentences to BLT concept embeddings (inference).

        Faithful to BLT §3.2.2: entropy decides boundaries; byte *hidden states*
        (with full-sentence context) are pooled per patch via cross-attention;
        patch reps are averaged into a sentence embedding. No gradients.
        """
        per_sentence = self._extract_patch_hidden(tokens_batch, threshold)
        return self._pool_per_sentence(per_sentence)

    def encode_sentences_batch(self, sentences, threshold=1.335):
        """Encode raw text sentences to BLT concept embeddings (inference)."""
        tokens_batch = [text_to_byte_tokens(s) for s in sentences]
        return self.encode_tokens_batch(tokens_batch, threshold)

    def encode_sentences_differentiable(self, sentences, threshold=1.335):
        """Encode sentences with gradients flowing to the pooler.

        The byte backbone stays frozen (features are extracted under
        ``no_grad``), but the cross-attention pooler runs in-graph, so its
        parameters receive gradients when it is in ``train()`` mode. Use this in
        a training loop that optimizes the pooler (e.g. joint pooler+decoder
        reconstruction). Returns a list of ``[dim]`` concept tensors.
        """
        tokens_batch = [text_to_byte_tokens(s) for s in sentences]
        per_sentence = self._extract_patch_hidden(tokens_batch, threshold)
        return self._pool_per_sentence(per_sentence)

    def save_pooler(self, path: str):
        """Persist the learned cross-attention pooler weights (self-describing)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "pooler_state_dict": self.pooler.state_dict(),
                "concept_dim": self.dim,
                "hidden_dim": self.hidden_dim,
                "pooler_heads": self.pooler_heads,
            },
            path,
        )

    def load_pooler(self, path: str):
        """Load learned pooler weights from a sidecar file.

        The sidecar records the concept dimension it was trained with; if it
        differs from the current pooler, the pooler is rebuilt to match and
        ``self.dim`` is updated so downstream modules size correctly.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("pooler_state_dict", ckpt)
        saved_cd = ckpt.get("concept_dim") if isinstance(ckpt, dict) else None
        if saved_cd is not None and saved_cd != self.dim:
            self.dim = int(saved_cd)
            self.pooler = PatchCrossAttentionPooler(
                self.hidden_dim,
                concept_dim=self.dim,
                n_heads=int(ckpt.get("pooler_heads", self.pooler_heads)),
            ).to(self.device)
        self.pooler.load_state_dict(state)
        self.pooler.to(self.device)
        self.pooler.eval()

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Decode BLT embeddings back to text using a trained BLTDecoder.

        Requires a trained decoder checkpoint at lcm_models/blt_decoder.pth.
        Falls back to placeholder if no decoder is available.
        """
        if self._decoder is None:
            return ["<decoded: no decoder loaded>" for _ in embeddings]
        if isinstance(embeddings, list):
            embeddings = torch.stack(embeddings)
        embeddings = embeddings.to(self.device)
        return self._decoder.decode(embeddings)

    def load_decoder(self, checkpoint_path: str = "lcm_models/blt_decoder.pth"):
        """Load a trained BLTDecoder checkpoint."""
        from blt_decoder import load_decoder

        self._decoder = load_decoder(checkpoint_path, device=self.device)
