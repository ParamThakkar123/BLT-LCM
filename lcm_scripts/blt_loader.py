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


class BLTLoader:
    def __init__(
        self,
        entropy_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        # Load the entropy model
        checkpoint = torch.load(entropy_model_path, map_location=device)
        from run_blt_patching import ByteEntropyModel

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

    def encode_sentences_batch(self, sentences_batch, threshold=1.335):
        """Encode a batch of sentences to BLT patch embeddings"""
        all_embeddings = []
        for sent in sentences_batch:
            tokens = text_to_byte_tokens(sent)
            if len(tokens) == 0:
                all_embeddings.append(torch.zeros(self.model.dim, device=self.device))
                continue
            tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            entropies = compute_entropies_for_tokens(
                tokens_tensor, self.model, device=self.device
            )
            entropies_list = entropies[0].tolist()
            boundaries, patch_lengths = entropy_patch_sentence(
                entropies_list, threshold
            )

            patch_embeddings = []
            for start, length in zip(boundaries, patch_lengths):
                end = start + length
                patch_tokens = tokens[start:end]
                # Get embedding as average hidden state
                with torch.no_grad():
                    inputs = torch.tensor([patch_tokens], dtype=torch.long).to(
                        self.device
                    )
                    outputs = self.model(inputs)
                    # Average over sequence
                    emb = outputs.mean(dim=1).squeeze(0)  # [dim]
                patch_embeddings.append(emb)

            # For sentence embedding, average patch embeddings
            if patch_embeddings:
                sent_emb = torch.stack(patch_embeddings).mean(dim=0)
            else:
                sent_emb = torch.zeros(self.model.dim, device=self.device)
            all_embeddings.append(sent_emb)

        return all_embeddings

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Not implemented"""
        return ["<decoded>" for _ in embeddings]
