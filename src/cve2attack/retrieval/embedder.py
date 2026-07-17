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
    """Lazy local embedding backend shared by all candidate-retrieval experiments.

    Construction is delayed until a retrieval run so validation and data
    inspection remain lightweight. Offline flags are set before importing
    transformers because it can otherwise start background network requests
    even when the selected model already exists in the local cache.
    """

    def __init__(self, model_name: str, *, local_files_only: bool = True):
        print(
            f"[retrieval] loading embedding model={model_name}; "
            f"local_files_only={local_files_only}",
            flush=True,
        )
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
            print(f"[retrieval] embedding model ready={model_name}", flush=True)
        except OSError as exc:
            if local_files_only:
                raise RuntimeError(
                    f"Model is not available in the local cache: {model_name}. "
                    "Set retrieval.local_files_only=false only when an intentional "
                    "model download is possible."
                ) from exc
            raise

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """Encode one caller-defined batch and return float32 vectors for ranking."""
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
