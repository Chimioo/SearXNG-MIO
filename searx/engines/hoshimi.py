# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoshimi - Vector-based cached search engine for SearXNG.

Hoshimi caches search results from other engines using Milvus Lite
for semantic vector search.
"""

from __future__ import annotations

import typing as t

from searx import logger
from searx.result_types import EngineResults, MainResult

if t.TYPE_CHECKING:
    from searx.search.processors import OfflineParams

logger = logger.getChild("engines.hoshimi")

# Engine metadata
about = {
    "website": "",
    "wikidata_id": "",
    "official_api_documentation": "",
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

categories = ["general"]
engine_type = "offline"
paging = False
time_range_support = False
safesearch = False
shortcut = "hs"

# Global engine instance
_hoshimi_engine = None


def init(engine_settings: dict[str, t.Any]) -> bool:
    """Initialize Hoshimi engine."""
    global _hoshimi_engine  # pylint: disable=global-statement

    from searx.hoshimi import HoshimiEngine

    config = engine_settings.get("hoshimi", {})
    # Ensure embedding_dim is set based on model
    if "embedding_dim" not in config:
        config["embedding_dim"] = 384  # default for all-MiniLM-L6-v2

    _hoshimi_engine = HoshimiEngine(config)

    try:
        _hoshimi_engine.initialize()
        logger.info("Hoshimi engine initialized successfully")
        return True
    except Exception as e:
        logger.error("Hoshimi engine initialization failed: %s", e)
        return False


def search(query: str, params: "OfflineParams") -> EngineResults:
    """Search cached results."""
    global _hoshimi_engine  # pylint: disable=global-statement

    results = EngineResults()

    if _hoshimi_engine is None:
        logger.error("Hoshimi engine not initialized")
        return results

    try:
        top_k = _hoshimi_engine.config.get("top_k", 20)
        search_mode = _hoshimi_engine.config.get("search_mode", "hybrid")  # vector, keyword, hybrid
        # Get category from params to filter results
        category = params.get("category", "")

        if search_mode == "keyword":
            cached_results = _hoshimi_engine.search_keyword(query, top_k=top_k, category=category)
        elif search_mode == "hybrid":
            vector_weight = _hoshimi_engine.config.get("vector_weight", 0.7)
            cached_results = _hoshimi_engine.search_hybrid(query, top_k=top_k, vector_weight=vector_weight, category=category)
        else:
            cached_results = _hoshimi_engine.search(query, top_k=top_k, category=category)

        for item in cached_results:
            result = MainResult(
                url=item.url,
                title=item.title,
                content=item.content,
                thumbnail=item.thumbnail if item.thumbnail else None,
                engine="hoshimi",
            )
            results.add(result)

        logger.debug("Hoshimi: Found %d cached results for '%s' (mode: %s, category: %s)", len(cached_results), query, search_mode, category)
    except Exception as e:
        logger.error("Hoshimi: Search failed: %s", e)

    return results


def get_hoshimi_engine():
    """Get the global Hoshimi engine instance.

    Note: Due to how SearXNG loads engine modules with short names,
    the _hoshimi_engine variable is set on the module registered in
    searx.engines.engines['hoshimi'], not on this module directly.
    """
    # First try our local variable (for direct module usage)
    global _hoshimi_engine  # pylint: disable=global-variable-not-assigned
    if _hoshimi_engine is not None:
        return _hoshimi_engine

    # Fall back to the module in engines dict
    try:
        from searx.engines import engines
        hoshimi_mod = engines.get('hoshimi')
        if hoshimi_mod is not None:
            return getattr(hoshimi_mod, '_hoshimi_engine', None)
    except Exception:
        pass

    return None
