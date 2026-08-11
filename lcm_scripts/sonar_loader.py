"""
SONAR embedding loader for LCM.

Uses the multilingual SONAR text encoder via the sonar-space card system,
which handles downloading and loading the correct checkpoint automatically.

``sonar-space`` and ``fairseq2`` are deliberately NOT project dependencies (see
the note in pyproject.toml -- fairseq2 pins torch==2.2.2 and cannot be resolved
against this project's torch==2.5.1). They install only inside apptainer.def's
isolated environment, so this module is import-safe but will raise on
construction outside the container.
"""

import torch


SONAR_ENCODER_CARD = "text_sonar_basic_encoder"
SONAR_DECODER_CARD = "text_sonar_basic_decoder"
SONAR_EMBEDDING_DIM = 1024


class SonarLoader:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        encoder_card: str = SONAR_ENCODER_CARD,
        batch_size: int = 8,
        decoder_card: str = SONAR_DECODER_CARD,
    ):
        from device_utils import report_device

        self.device = report_device(device, label="SonarLoader", warn_cpu=False)
        self.encoder_card = encoder_card
        self.decoder_card = decoder_card
        self.batch_size = batch_size
        self._decoder = None
        self.model = self._load_pipeline()

    def _load_pipeline(self):
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

        return TextToEmbeddingModelPipeline(
            encoder=self.encoder_card,
            tokenizer=self.encoder_card,
            device=self.device,
        )

    # SONAR text encoder max positional length is 514 tokens.
    # Observed ratio ~2.75 chars/token for Marathi Devanagari SentencePiece,
    # so 1000 chars ≈ 364 tokens — well within the limit.
    _MAX_CHARS = 1000

    def encode_sentences(self, sentences, lang="mar_Deva"):
        """Encode a list of sentences to 1024-dimensional SONAR embeddings."""
        if not sentences:
            return torch.empty(0, SONAR_EMBEDDING_DIM, device=self.device)
        sentences = [s[: self._MAX_CHARS] for s in sentences]
        return self.model.predict(
            sentences,
            source_lang=lang,
            batch_size=self.batch_size,
        )

    def _load_decoder(self):
        """Lazily build the SONAR embedding-to-text decoder pipeline."""
        if self._decoder is None:
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

            self._decoder = EmbeddingToTextModelPipeline(
                decoder=self.decoder_card,
                tokenizer=self.decoder_card,
                device=self.device,
            )
        return self._decoder

    def decode_embeddings(self, embeddings, target_lang="mar_Deva", max_seq_len=256):
        """Decode SONAR embeddings back to text.

        This previously returned the literal string ``"<decoded>"`` for every
        input, which silently turned any evaluation routed through it into
        nonsense. It now runs the real SONAR text decoder; if the decoder assets
        are unavailable the underlying error is raised rather than masked.
        """
        if embeddings is None or len(embeddings) == 0:
            return []
        if isinstance(embeddings, list):
            embeddings = torch.stack(list(embeddings))
        embeddings = embeddings.to(self.device)
        return self._load_decoder().predict(
            embeddings,
            target_lang=target_lang,
            max_seq_len=max_seq_len,
            batch_size=self.batch_size,
        )
