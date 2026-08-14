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
        self.embeddings = F.normalize(
            embeddings.float(), dim=1
        )  # embeddings stay on original device

    @classmethod
    def from_corpus(
        cls,
        sentences: List[str],
        blt_loader,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> "EmbeddingRetriever":
        """Build an index by encoding sentences with BLTLoader.

        ``batch_size`` is accepted for call compatibility but no longer slices
        the corpus: the loader length-sorts across everything it is given and
        sizes each forward by padded byte count, so pre-slicing into fixed
        batches could only sort within a slice and still paid for that slice's
        longest sentence.
        """
        if not sentences:
            return cls(sentences, torch.empty(0, blt_loader.dim, device=device))
        embeddings = blt_loader.encode_sentences(sentences).detach().to(device)
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
        if len(sentences) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(sentences)} sentences vs {embeddings.shape[0]} embeddings. "
                "Sentences and embeddings must have the same length."
            )
        return cls(sentences[:n], embeddings[:n])

    def _similarities(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of every query against the index, as one matmul.

        The queries are moved to wherever the index lives rather than the index
        being dragged to the CPU: the index is the large operand (N x dim), and
        pulling it back for every batch turned a GPU matmul into a CPU one.
        """
        query = F.normalize(
            query_embeddings.detach().to(self.embeddings.device).float(), dim=1
        )
        return query @ self.embeddings.T  # [B, N]

    def retrieve(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 1,
    ) -> List[str]:
        """Find nearest sentences for each query embedding.

        Args:
            query_embeddings: [B, embed_dim] predicted embeddings.
            top_k: accepted for call compatibility; the text output is top-1.

        Returns:
            list of B sentences (top-1 matches).
        """
        if query_embeddings.numel() == 0:
            return []
        indices = self._similarities(query_embeddings).argmax(dim=1).cpu().tolist()
        return [self.sentences[i] for i in indices]

    def retrieve_with_scores(
        self,
        query_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> List[List[tuple]]:
        """Return top-k (sentence, score) pairs per query."""
        if query_embeddings.numel() == 0:
            return []
        top_k = min(top_k, len(self.sentences))
        topk = self._similarities(query_embeddings).topk(top_k, dim=1)
        # One transfer for the whole batch instead of two scalar syncs per pair.
        idx_rows = topk.indices.cpu().tolist()
        score_rows = topk.values.cpu().tolist()
        return [
            [(self.sentences[i], float(s)) for i, s in zip(idxs, scores)]
            for idxs, scores in zip(idx_rows, score_rows)
        ]
