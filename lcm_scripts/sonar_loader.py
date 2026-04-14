"""
SONAR-like embedding loader for LCM
Uses XLM-RoBERTa for multilingual sentence embeddings
"""

import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer


class SonarLoader:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base").to(device)

    def encode_sentences(self, sentences, lang="mar_Deva"):
        """Encode list of sentences to embeddings"""
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output  # [batch, 768]
            # Project to 1024 dim to match SONAR
            embeddings = torch.nn.functional.pad(
                embeddings, (0, 1024 - 768)
            )  # Simple pad
        return embeddings

    def decode_embeddings(self, embeddings, target_lang="mar_Deva"):
        """Decode embeddings to text - not implemented"""
        return ["<decoded>" for _ in embeddings]
