"""
SONAR embedding loader for LCM.

Uses the multilingual SONAR text encoder via the sonar-space card system,
which handles downloading and loading the correct checkpoint automatically.
"""

import torch


SONAR_ENCODER_CARD = "text_sonar_basic_encoder"
SONAR_EMBEDDING_DIM = 1024


class SonarLoader:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        encoder_card: str = SONAR_ENCODER_CARD,
        batch_size: int = 8,
    ):
        self.device = torch.device(device)
        self.encoder_card = encoder_card
        self.batch_size = batch_size
        self.model = self._load_pipeline()

    def _load_pipeline(self):
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        return TextToEmbeddingModelPipeline(
            encoder=self.encoder_card,
            tokenizer=self.encoder_card,
            device=self.device,
        )

    def encode_sentences(self, sentences, lang="mar_Deva"):
        """Encode a list of sentences to 1024-dimensional SONAR embeddings."""
        if not sentences:
            return torch.empty(0, SONAR_EMBEDDING_DIM, device=self.device)
        return self.model.predict(
            sentences,
            source_lang=lang,
            batch_size=self.batch_size,
        )

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Decode embeddings to text - not implemented for this encoder-only loader."""
        return ["<decoded>" for _ in embeddings]
