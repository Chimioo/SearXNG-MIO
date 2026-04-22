# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embedding model wrapper for Hoshimi."""

from __future__ import annotations

import os
import typing as t

from searx import logger

logger = logger.getChild("hoshimi.embedding")

_DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class EmbeddingModel:
    """Wrapper around sentence-transformers for text embedding."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _DEFAULT_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Enable offline mode to avoid network timeouts when HuggingFace
            # is unreachable. Model must be cached locally (e.g. via prior download).
            if not os.environ.get("HF_HUB_OFFLINE"):
                os.environ["HF_HUB_OFFLINE"] = "1"

            from sentence_transformers import SentenceTransformer
            logger.info("Hoshimi: Loading embedding model '%s' (offline mode)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def ensure_loaded(self):
        """Preload the model (called during startup)."""
        _ = self.model

    def encode(self, text: str) -> list[float] | None:
        """Encode a single text to vector."""
        if not text.strip():
            return None
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error("Hoshimi: Failed to encode text: %s", e)
            return None

    def encode_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Encode multiple texts to vectors."""
        valid_texts = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not valid_texts:
            return [None] * len(texts)

        try:
            indices = [i for i, _ in valid_texts]
            content = [t for _, t in valid_texts]

            embeddings = self.model.encode(content, convert_to_numpy=True)
            vectors = [v.tolist() for v in embeddings]

            result: list[list[float] | None] = [None] * len(texts)
            for idx, vec in zip(indices, vectors):
                result[idx] = vec
            return result
        except Exception as e:
            logger.error("Hoshimi: Failed to batch encode: %s", e)
            return [None] * len(texts)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
