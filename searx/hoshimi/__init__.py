# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoshimi - Vector-based search engine for SearXNG.

Hoshimi caches search results from other engines, embeds them into vectors,
and provides fast semantic search through Milvus Lite.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field


@dataclass
class HoshimiResult:
    """A single cached result from an external search engine."""

    url: str
    title: str = ""
    content: str = ""
    thumbnail: str = ""
    engine: str = ""
    category: str = ""
    engine_weight: float = 1.0
    vector: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "thumbnail": self.thumbnail,
            "engine": self.engine,
            "category": self.category,
            "engine_weight": self.engine_weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "HoshimiResult":
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            thumbnail=data.get("thumbnail", ""),
            engine=data.get("engine", ""),
            category=data.get("category", ""),
            engine_weight=data.get("engine_weight", 1.0),
        )


class HoshimiEngine:
    """Main Hoshimi engine class.

    Manages embedding, vector storage, and search operations.
    """

    def __init__(self, config: dict[str, t.Any]):
        self.config = config
        self._embedding = None
        self._vector_store = None
        self._collector = None

    @property
    def embedding(self):
        if self._embedding is None:
            from searx.hoshimi.embedding import EmbeddingModel
            self._embedding = EmbeddingModel(self.config.get("embedding_model"))
        return self._embedding

    @property
    def vector_store(self):
        if self._vector_store is None:
            from searx.hoshimi.milvus_store import MilvusStore
            self._vector_store = MilvusStore(self.config)
        return self._vector_store

    @property
    def collector(self):
        if self._collector is None:
            from searx.hoshimi.collector import ResultCollector
            self._collector = ResultCollector(self)
        return self._collector

    def _apply_engine_weight(self, results: list[HoshimiResult], score_attr: str = "_score") -> list[HoshimiResult]:
        """Apply engine weight to result scores and re-sort."""
        max_weight = max((r.engine_weight for r in results), default=1.0) or 1.0
        for r in results:
            base_score = getattr(r, score_attr, 0) or 0
            # Normalize engine weight to 0.5-1.5 range to avoid overwhelming the score
            weight_factor = 0.5 + (r.engine_weight / max_weight)
            r._score = base_score * weight_factor  # type: ignore

        results.sort(key=lambda x: getattr(x, "_score", 0), reverse=True)
        return results

    def search(self, query: str, top_k: int = 20, category: str = "") -> list[HoshimiResult]:
        """Search cached results by query using vector similarity."""
        query_vector = self.embedding.encode(query)
        if query_vector is None:
            return []

        raw_results = self.vector_store.search(query_vector, top_k=top_k, category=category)
        results = [HoshimiResult.from_dict(r) for r in raw_results]
        # Attach vector score
        for r, raw in zip(results, raw_results):
            r._score = raw.get("score", 0)  # type: ignore
        return self._apply_engine_weight(results)

    def search_keyword(self, query: str, top_k: int = 20, category: str = "") -> list[HoshimiResult]:
        """Search cached results using keyword matching."""
        raw_results = self.vector_store.keyword_search(query, top_k=top_k, category=category)
        results = [HoshimiResult.from_dict(r) for r in raw_results]
        # Attach keyword score
        for r, raw in zip(results, raw_results):
            r._score = raw.get("score", 0)  # type: ignore
        return self._apply_engine_weight(results)

    def search_hybrid(self, query: str, top_k: int = 20, vector_weight: float = 0.7, category: str = "") -> list[HoshimiResult]:
        """Search using both vector similarity and keyword matching.

        Combines vector search and keyword search scores, then applies engine weight.
        vector_weight controls the balance (0.7 = 70% vector, 30% keyword).
        """
        query_vector = self.embedding.encode(query)
        if query_vector is None:
            # Fall back to keyword only
            return self.search_keyword(query, top_k, category=category)

        # Run both searches with category filter
        vector_results = self.vector_store.search(query_vector, top_k=top_k * 2, category=category)
        keyword_results = self.vector_store.keyword_search(query, top_k=top_k * 2, category=category)

        # Build a combined score map by URL
        url_data: dict[str, dict] = {}

        # Normalize vector scores to 0-1
        max_vector_score = max((r.get("score", 0) for r in vector_results), default=1.0) or 1.0
        for r in vector_results:
            url = r.get("url", "")
            normalized = r.get("score", 0) / max_vector_score
            url_data[url] = {"vector_score": normalized, "keyword_score": 0.0, "entity": r}

        # Normalize keyword scores to 0-1
        max_kw_score = max((r.get("score", 0) for r in keyword_results), default=1.0) or 1.0
        for r in keyword_results:
            url = r.get("url", "")
            normalized = r.get("score", 0) / max_kw_score
            if url in url_data:
                url_data[url]["keyword_score"] = normalized
            else:
                url_data[url] = {"vector_score": 0.0, "keyword_score": normalized, "entity": r}

        # Calculate combined score with engine weight
        kw_weight = 1.0 - vector_weight
        results = []
        for url, data in url_data.items():
            entity = data["entity"]
            combined = data["vector_score"] * vector_weight + data["keyword_score"] * kw_weight

            # Apply engine weight (normalize to 0.5-1.5 multiplier)
            weight_factor = 0.5 + (entity.get("engine_weight", 1.0) / max(d["entity"].get("engine_weight", 1.0) for d in url_data.values()))
            final_score = combined * weight_factor

            result = HoshimiResult.from_dict(entity)
            result._score = final_score  # type: ignore
            results.append(result)

        results.sort(key=lambda x: getattr(x, "_score", 0), reverse=True)
        return results[:top_k]

    def add_results(self, results: list[HoshimiResult]):
        """Add results to the cache."""
        if not results:
            return

        # Batch encode
        texts = [f"{r.title} {r.content}" for r in results]
        vectors = self.embedding.encode_batch(texts)

        for i, result in enumerate(results):
            if i < len(vectors) and vectors[i] is not None:
                result.vector = vectors[i]

        self.vector_store.add(results)

    def initialize(self):
        """Initialize the engine (called during SearXNG startup)."""
        self.embedding.ensure_loaded()
        self.vector_store.ensure_initialized()
