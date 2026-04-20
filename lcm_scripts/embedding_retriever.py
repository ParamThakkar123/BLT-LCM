"""
Nearest-neighbor retrieval decoder for BLT-LCM evaluation.

Converts predicted embeddings back to text by finding the closest sentence
in a precomputed index. No training required — works immediately with the
existing BLT embedding cache.

Usage:
    retriever = EmbeddingRetriever.from_corpus(sentences, blt_loader)
    texts = retriever.retrieve(predicted_embeddings)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


class EmbeddingRetriever:
    """Look up the nearest real sentence for a given embedding vector."""

    def __init__(
        self,
        sentences: List[str],
        embeddings: torch.Tensor,
    ):
        """
        Args:
            sentences: list of N source sentences.
            embeddings: [N, embed_dim] normalized or raw embeddings.
        """
        self.sentences = sentences
        self.embeddings = F.normalize(embeddings.float(), dim=1)

    @classmethod
    def from_corpus(
        cls,
        sentences: List[str],
        blt_loader,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> "EmbeddingRetriever":
        """Build an index by encoding sentences with BLTLoader."""
        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            emb_batch = blt_loader.encode_sentences_batch(batch)
            all_embs.extend([e.detach().cpu() for e in emb_batch])
        embeddings = torch.stack(all_embs)
        return cls(sentences, embeddings)

    @classmethod
    def from_cache(
        cls,
        sentences: List[str],
        embed_cache_path: str,
    ) -> "EmbeddingRetriever":
        """Build index from a precomputed embedding cache (list of [S, D] tensors).

        The cache format matches train_lcm_blt.py's blt_embeddings_cache.pth:
        a list of tensors, one per document, each shaped [num_sentences, embed_dim].
        ``sentences`` should be the flat list of sentences in the same order.
        """
        raw = torch.load(embed_cache_path, map_location="cpu", weights_only=False)
        flat = []
        for doc_emb in raw:
            if doc_emb.dim() == 2 and doc_emb.shape[0] > 0:
                flat.append(doc_emb)
        embeddings = torch.cat(flat, dim=0)
        n = min(len(sentences), embeddings.shape[0])
        return cls(sentences[:n], embeddings[:n])

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 1,
    ) -> List[str]:
        """Find nearest sentences for each query embedding.

        Args:
            query_embeddings: [B, embed_dim] predicted embeddings.
            top_k: return the top-k nearest sentence (uses top-1 for text output).

        Returns:
            list of B sentences (top-1 matches).
        """
        query = F.normalize(query_embeddings.float().detach().cpu(), dim=1)
        sims = query @ self.embeddings.T  # [B, N]
        indices = sims.argmax(dim=1).tolist()
        return [self.sentences[i] for i in indices]

    def retrieve_with_scores(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> List[List[tuple]]:
        """Return top-k (sentence, score) pairs per query."""
        query = F.normalize(query_embeddings.float().detach().cpu(), dim=1)
        sims = query @ self.embeddings.T
        topk = sims.topk(top_k, dim=1)
        results = []
        for i in range(query.shape[0]):
            pairs = []
            for j in range(top_k):
                idx = int(topk.indices[i, j].item())
                score = float(topk.values[i, j].item())
                pairs.append((self.sentences[idx], score))
            results.append(pairs)
        return results
