"""ChemBERTa embedder (Step 2).

Uses seyonec/ChemBERTa-zinc-base-v1 (the DeepChem-trained checkpoint, published under the
`seyonec` Hugging Face namespace rather than `DeepChem`), a RoBERTa encoder pretrained on ~77M
SMILES from ZINC.
The embedding for a SMILES string is the mean-pooled last hidden state over its real (non-pad)
tokens — standard sentence-embedding pooling for an encoder-only model; CLS-token pooling is
available in prior ChemBERTa work but mean pooling is more robust when fine-tuning wasn't done
on a sentence-embedding objective, which is the case for the base pretrained checkpoint used here.
"""

from __future__ import annotations

import numpy as np
import torch

from src.common.config import LANGUAGE_MODEL_HF_IDS, RuntimeConfig
from src.common.exceptions import ModelDownloadError
from src.common.logging_setup import get_logger
from src.common.runtime import get_device
from src.language.base_embedder import BaseSMILESEmbedder

logger = get_logger(__name__)

CHEMBERTA_EMBEDDING_DIM = 768


class ChemBERTaEmbedder(BaseSMILESEmbedder):
    model_name = "chemberta"
    embedding_dim = CHEMBERTA_EMBEDDING_DIM

    def __init__(self, config: RuntimeConfig | None = None, hf_model_id: str | None = None):
        config = config or RuntimeConfig(language_model="chemberta")
        self.device = get_device(config)
        self.hf_model_id = hf_model_id or LANGUAGE_MODEL_HF_IDS["chemberta"]

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self.model = AutoModel.from_pretrained(self.hf_model_id).to(self.device)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed boundary error below
            raise ModelDownloadError(
                f"Failed to load ChemBERTa ({self.hf_model_id}) from Hugging Face Hub or local "
                f"cache. Check network access, or pre-download the model with "
                f"`huggingface-cli download {self.hf_model_id}`."
            ) from exc

        self.model.eval()
        logger.info(f"Loaded ChemBERTa ({self.hf_model_id}) on device={self.device}")

    @torch.no_grad()
    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        encoded = self.tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden_states * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.cpu().numpy()
