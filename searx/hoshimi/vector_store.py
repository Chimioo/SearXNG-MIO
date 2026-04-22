# SPDX-License-Identifier: AGPL-3.0-or-later
"""Abstract vector store interface for Hoshimi."""

from __future__ import annotations

import abc
import typing as t

if t.TYPE_CHECKING:
    from searx.hoshimi import HoshimiResult


class VectorStore(abc.ABC):
    """Abstract base class for vector storage backends."""

    @abc.abstractmethod
    def ensure_initialized(self) -> bool:
        """Initialize the storage (create collections, indexes, etc.).
        Returns True if initialization was successful.
        """

    @abc.abstractmethod
    def add(self, results: list[HoshimiResult]) -> bool:
        """Add results to the vector store."""

    @abc.abstractmethod
    def search(self, query_vector: list[float], top_k: int = 20, category: str = "") -> list[dict[str, t.Any]]:
        """Search for similar vectors.
        Returns list of result dicts with metadata.
        """

    @abc.abstractmethod
    def keyword_search(self, query: str, top_k: int = 20, category: str = "") -> list[dict[str, t.Any]]:
        """Search using keyword matching.
        Returns list of result dicts with metadata.
        """

    @abc.abstractmethod
    def close(self):
        """Clean up resources."""
