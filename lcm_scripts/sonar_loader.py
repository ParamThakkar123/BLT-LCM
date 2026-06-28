"""
SONAR embedding loader for LCM.

This loader uses Meta's Marathi SONAR text encoder checkpoint from Hugging Face
instead of the previous XLM-RoBERTa SONAR-like stand-in.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download


SONAR_REPO_ID = "facebook/SONAR"
MARATHI_ENCODER_FILENAME = "spenc.v5ap.mar.pt"
SONAR_TOKENIZER_CARD = "text_sonar_basic_encoder"
SONAR_EMBEDDING_DIM = 1024


class SonarLoader:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        encoder_repo_id: str = SONAR_REPO_ID,
        encoder_filename: str = MARATHI_ENCODER_FILENAME,
        tokenizer: str = SONAR_TOKENIZER_CARD,
        batch_size: int = 8,
    ):
        self.device = torch.device(device)
        self.encoder_repo_id = encoder_repo_id
        self.encoder_filename = encoder_filename
        self.tokenizer_name = tokenizer
        self.batch_size = batch_size
        self.model = self._load_pipeline()

    def _load_pipeline(self):
        """Load the real SONAR Marathi encoder pipeline from Hugging Face."""
        # fairseq2 0.2.x exposes 'asset_store'; sonar-space 0.3.x expects 'default_asset_store'
        import fairseq2.assets as _fa
        if not hasattr(_fa, "default_asset_store") and hasattr(_fa, "asset_store"):
            _fa.default_asset_store = _fa.asset_store

        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.sonar_text.builder import (
            create_sonar_text_encoder_model,
            sonar_text_encoder_archs,
        )
        from sonar.models.sonar_text.loader import (
            convert_sonar_text_encoder_checkpoint,
            load_sonar_tokenizer,
        )

        checkpoint_path = hf_hub_download(
            repo_id=self.encoder_repo_id,
            filename=self.encoder_filename,
        )
        config = sonar_text_encoder_archs.get("basic")
        encoder = create_sonar_text_encoder_model(config, device=self.device)
        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        converted = convert_sonar_text_encoder_checkpoint(checkpoint, config)
        state_dict = converted.get("model", converted)
        encoder.load_state_dict(state_dict)
        encoder.eval()

        tokenizer = load_sonar_tokenizer(self.tokenizer_name, progress=False)
        return TextToEmbeddingModelPipeline(
            encoder=encoder,
            tokenizer=tokenizer,
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
            target_device=self.device,
        )

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Decode embeddings to text - not implemented for this encoder-only loader."""
        return ["<decoded>" for _ in embeddings]
