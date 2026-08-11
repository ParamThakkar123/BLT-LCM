"""
BLT embedding loader for LCM.

Produces sentence-level concept embeddings the BLT way (Byte Latent Transformer,
Meta 2024):

  1. The entropy model is used ONLY to place patch boundaries, via the per-byte
     entropy of its next-byte distribution (§2.3). It contributes no
     representations and stays frozen.
  2. A separate, trainable local encoder E embeds the bytes, augments them with
     hash n-gram embeddings (§3.2.1) and contextualises them with transformer
     layers under a local block-causal mask (§3.2).
  3. Each patch's byte hidden states are pooled into a patch representation by a
     Perceiver-style cross-attention pooler (§3.2.2, patch-masked).
  4. The resulting *patch sequence* is processed by the global latent
     transformer (§3.1) and pooled into the sentence concept.

Both the entropy model's logits and its hidden states are unused downstream.
"""

import torch
import sys
import os
from typing import Optional, Sequence

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from run_blt_patching import (
    DEFAULT_THRESHOLD,
    DEFAULT_THRESHOLD_ADD,
    text_to_byte_tokens,
    byte_tokens_to_text,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
)
from run_blt_patching import ByteEntropyModel
from device_utils import report_device
from blt_local_encoder import (
    DEFAULT_HASH_VOCAB,
    DEFAULT_NGRAM_SIZES,
    BLTSentenceEncoder,
)


class BLTLoader:
    def __init__(
        self,
        entropy_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pooler_path: Optional[str] = None,
        concept_dim: int = 1024,
        encoder_dim: int = 256,
        encoder_layers: int = 1,
        latent_layers: int = 4,
        encoder_heads: int = 8,
        encoder_window: int = 512,
        use_hash_ngrams: bool = True,
        ngram_sizes: Sequence[int] = DEFAULT_NGRAM_SIZES,
        hash_vocab_size: int = DEFAULT_HASH_VOCAB,
        patching_mode: str = "global",
        threshold_add: float = DEFAULT_THRESHOLD_ADD,
        verbose: bool = True,
    ):
        self.device = device
        self.patching_mode = patching_mode
        self.threshold_add = threshold_add
        if verbose:
            # The caller usually reports its own device already; this confirms the
            # loader was handed the same one rather than quietly defaulting.
            report_device(device, label="BLTLoader", warn_cpu=False)
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
        self.pooler_heads = int(cfg.get("pooler_heads", encoder_heads))

        # The trainable byte side of BLT: local encoder (byte embeddings + hash
        # n-grams + block-causal transformer layers) -> per-patch cross-attention
        # pooling -> global latent transformer over the patch sequence. The
        # entropy model above only supplies boundaries and stays frozen.
        self.encoder = BLTSentenceEncoder(
            vocab_size=self.model.vocab_size,
            dim=encoder_dim,
            concept_dim=self.dim,
            encoder_layers=encoder_layers,
            latent_layers=latent_layers,
            n_heads=encoder_heads,
            window=encoder_window,
            use_hash_ngrams=use_hash_ngrams,
            ngram_sizes=ngram_sizes,
            hash_vocab_size=hash_vocab_size,
        ).to(device)
        self.encoder.eval()
        if verbose:
            total = sum(p.numel() for p in self.encoder.parameters())
            print(f"[BLTLoader] trainable byte-side parameters: {total:,}")
            ngrams = self.encoder.encoder.hash_ngrams
            if ngrams is not None:
                print(f"[BLTLoader] {ngrams.memory_summary()}")
        # Backwards-compatible alias: joint pooler+decoder training and the
        # sidecar files address this stack as ".pooler".
        self.pooler = self.encoder
        if isinstance(checkpoint, dict) and checkpoint.get("pooler_state_dict"):
            try:
                self.encoder.load_state_dict(checkpoint["pooler_state_dict"])
            except Exception as e:  # pragma: no cover - defensive
                print(f"[BLTLoader] Could not load encoder weights: {e}")
        # A sidecar pooler file (from joint pooler+decoder training) takes
        # precedence, so the learned concept space is used by training/eval.
        if pooler_path and os.path.exists(pooler_path):
            try:
                self.load_pooler(pooler_path)
                print(f"[BLTLoader] Loaded learned pooler from {pooler_path}")
            except Exception as e:  # pragma: no cover - defensive
                print(f"[BLTLoader] Could not load pooler from {pooler_path}: {e}")

    @torch.no_grad()
    def _boundaries(self, tokens, threshold):
        """Patch boundaries for one byte sequence. Entropy model only, frozen."""
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        entropies = compute_entropies_for_tokens(
            tokens_tensor, self.model, device=self.device
        )
        return entropy_patch_sentence(
            entropies[0].tolist(),
            threshold,
            mode=self.patching_mode,
            threshold_add=self.threshold_add,
        )

    def _encode(self, tokens_batch, threshold):
        """bytes -> boundaries (frozen) -> local encoder + latent transformer.

        Runs the trainable stack in the CURRENT autograd context, so gradients
        reach it whenever it is in ``train()`` mode and grad is enabled.
        """
        embeddings = []
        for tokens in tokens_batch:
            if len(tokens) == 0:
                embeddings.append(torch.zeros(self.dim, device=self.device))
                continue
            boundaries, patch_lengths = self._boundaries(tokens, threshold)
            tokens_tensor = torch.tensor(
                [tokens], dtype=torch.long, device=self.device
            )
            embeddings.append(
                self.encoder(tokens_tensor, boundaries, patch_lengths)
            )
        return embeddings

    @torch.no_grad()
    def encode_tokens_batch(self, tokens_batch, threshold=DEFAULT_THRESHOLD):
        """Encode tokenized sentences to BLT concept embeddings (inference).

        Returns a list of ``[dim]`` tensors. No gradients.
        """
        return self._encode(tokens_batch, threshold)

    def encode_sentences_batch(self, sentences, threshold=DEFAULT_THRESHOLD):
        """Encode raw text sentences to BLT concept embeddings (inference).

        Returns a list of ``[dim]`` tensors.
        """
        tokens_batch = [text_to_byte_tokens(s) for s in sentences]
        return self.encode_tokens_batch(tokens_batch, threshold)

    def encode_sentences(self, sentences, threshold=DEFAULT_THRESHOLD):
        """Encode sentences and stack into a single ``[N, dim]`` tensor.

        Convenience wrapper matching ``SonarLoader.encode_sentences``, which
        returns a tensor rather than a list.
        """
        embs = self.encode_sentences_batch(sentences, threshold)
        if not embs:
            return torch.empty(0, self.dim, device=self.device)
        return torch.stack(embs)

    def encode_sentences_differentiable(self, sentences, threshold=DEFAULT_THRESHOLD):
        """Encode sentences with gradients flowing to the trainable encoder.

        The entropy model stays frozen (boundaries are computed under
        ``no_grad``), but the local encoder, pooler and latent transformer run
        in-graph, so their parameters receive gradients in ``train()`` mode.
        Returns a list of ``[dim]`` concept tensors.
        """
        tokens_batch = [text_to_byte_tokens(s) for s in sentences]
        return self._encode(tokens_batch, threshold)

    def save_pooler(self, path: str):
        """Persist the trainable byte-side encoder stack (self-describing).

        The filename is kept for compatibility with existing runbooks; the
        payload now covers the whole local encoder + latent transformer, not
        just the cross-attention pooler.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "pooler_state_dict": self.encoder.state_dict(),
                "concept_dim": self.dim,
                "hidden_dim": self.hidden_dim,
                "encoder_dim": self.encoder.encoder.dim,
                "pooler_heads": self.pooler_heads,
                "format": "blt_sentence_encoder_v2",
            },
            path,
        )

    def load_pooler(self, path: str):
        """Load byte-side encoder weights from a sidecar file.

        Sidecars written before the local encoder existed only contain the
        cross-attention pooler; those are loaded into the pooler submodule and
        the rest of the stack keeps its initialization, with a warning, rather
        than failing outright.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt.get("pooler_state_dict", ckpt)
        saved_cd = ckpt.get("concept_dim") if isinstance(ckpt, dict) else None
        if saved_cd is not None and saved_cd != self.dim:
            raise ValueError(
                f"Pooler sidecar {path} was trained with concept_dim={saved_cd}, "
                f"but this loader was built with concept_dim={self.dim}. "
                f"Construct BLTLoader(concept_dim={saved_cd}) to match."
            )
        if isinstance(ckpt, dict) and ckpt.get("format") == "blt_sentence_encoder_v2":
            self.encoder.load_state_dict(state)
        else:
            print(
                f"[BLTLoader] {path} is a legacy pooler-only sidecar; loading it "
                "into the cross-attention pooler and leaving the local encoder "
                "and latent transformer at their initialization. Retrain with "
                "blt_decoder.py to produce a full encoder checkpoint."
            )
            self.encoder.encoder.pooler.load_state_dict(state)
        self.encoder.to(self.device)
        self.encoder.eval()
        self.pooler = self.encoder

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
