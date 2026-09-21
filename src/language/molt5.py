"""MolT5 embedder (Step 2).

Uses laituan245/molt5-base, a T5 encoder-decoder pretrained/fine-tuned on molecule<->text
translation. Only the encoder is used here (no generation) — the embedding for a SMILES string
is the mean-pooled encoder last-hidden-state over its real (non-pad) tokens, matching the
pooling strategy used for ChemBERTa (src/language/chemberta.py) so the two are directly
comparable as graph-construction inputs (Step 3) without a pooling-method confound.
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

MOLT5_EMBEDDING_DIM = 768


class MolT5Embedder(BaseSMILESEmbedder):
    model_name = "molt5"
    embedding_dim = MOLT5_EMBEDDING_DIM

    def __init__(self, config: RuntimeConfig | None = None, hf_model_id: str | None = None):
        config = config or RuntimeConfig(language_model="molt5")
        self.device = get_device(config)
        self.hf_model_id = hf_model_id or LANGUAGE_MODEL_HF_IDS["molt5"]

        try:
            from transformers import AutoTokenizer, T5EncoderModel

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
            self.model = T5EncoderModel.from_pretrained(self.hf_model_id).to(self.device)
        except Exception as exc:  # noqa: BLE001 - re-raised as a typed boundary error below
            raise ModelDownloadError(
                f"Failed to load MolT5 ({self.hf_model_id}) from Hugging Face Hub or local "
                f"cache. Check network access, or pre-download the model with "
                f"`huggingface-cli download {self.hf_model_id}`."
            ) from exc

        self.model.eval()
        logger.info(f"Loaded MolT5 ({self.hf_model_id}) on device={self.device}")

    @torch.no_grad()
    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        encoded = self.tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**encoded)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (hidden_states * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.cpu().numpy()
