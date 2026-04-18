"""
BLT embedding loader for LCM
Uses BLT byte patching and entropy model for concept embeddings
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "patching_scratch"))

from run_blt_patching import (
    text_to_byte_tokens,
    byte_tokens_to_text,
    compute_entropies_for_tokens,
    entropy_patch_sentence,
)
from run_blt_patching import ByteEntropyModel


class BLTLoader:
    def __init__(
        self,
        entropy_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
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
        # Compile model for faster inference
        self.model = torch.compile(self.model)
        # Cache for tokenized sentences
        self.token_cache = {}

    def encode_tokens_batch(self, tokens_batch, threshold=1.335):
        """Encode a batch of tokenized sentences to BLT patch embeddings"""
        all_patch_tokens = []
        sentence_patch_counts = []
        for tokens in tokens_batch:
            if len(tokens) == 0:
                sentence_patch_counts.append(0)
                continue
            tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            entropies = compute_entropies_for_tokens(
                tokens_tensor, self.model, device=self.device
            )
            entropies_list = entropies[0].tolist()
            boundaries, patch_lengths = entropy_patch_sentence(
                entropies_list, threshold
            )

            patch_tokens_list = []
            for start, length in zip(boundaries, patch_lengths):
                end = start + length
                patch_tokens = tokens[start:end]
                patch_tokens_list.append(patch_tokens)
            all_patch_tokens.extend(patch_tokens_list)
            sentence_patch_counts.append(len(patch_tokens_list))

        all_embeddings = []
        if all_patch_tokens:
            # Pad all patches to max length
            max_len = max(len(p) for p in all_patch_tokens)
            padded_patches = []
            for p in all_patch_tokens:
                pad_len = max_len - len(p)
                padded_patches.append(
                    p + [0] * pad_len
                )  # Pad with 0 (assuming 0 is padding token)
            batch_tensor = torch.tensor(padded_patches, dtype=torch.long).to(
                self.device
            )  # [num_patches, max_len]
            with torch.no_grad():
                if "cuda" in str(self.device):
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self.model(
                            batch_tensor
                        )  # [num_patches, max_len, dim]
                        patch_embs = outputs.float().mean(dim=1)  # [num_patches, dim]
                else:
                    outputs = self.model(batch_tensor)  # [num_patches, max_len, dim]
                    patch_embs = outputs.mean(dim=1)  # [num_patches, dim]
            index = 0
            for num_patches in sentence_patch_counts:
                if num_patches > 0:
                    sent_patches = patch_embs[index : index + num_patches]
                    sent_emb = sent_patches.mean(dim=0)
                else:
                    sent_emb = torch.zeros(self.model.dim, device=self.device)
                all_embeddings.append(sent_emb)
                index += num_patches
        else:
            all_embeddings = [
                torch.zeros(self.model.dim, device=self.device) for _ in tokens_batch
            ]

        return all_embeddings

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Not implemented"""
        return ["<decoded>" for _ in embeddings]
