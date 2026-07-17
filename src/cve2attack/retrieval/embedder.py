"""Embedding backend abstraction."""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers backend so schema/evaluation needs no Torch import."""

    def __init__(self, model_name: str, *, local_files_only: bool = True):
        if local_files_only:
            # transformers otherwise starts a background safetensors-conversion
            # request even when SentenceTransformer itself is offline.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.local_files_only = local_files_only
        try:
            self._model = SentenceTransformer(
                model_name,
                local_files_only=local_files_only,
                model_kwargs={"use_safetensors": False},
            )
        except OSError as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Model is not available in the local cache: {model_name}. "
                    "Set retrieval.local_files_only=false only when an intentional "
                    "model download is possible."
                ) from exc
            raise

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
