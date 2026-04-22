# SPDX-License-Identifier: AGPL-3.0-or-later
"""Result collector for Hoshimi.

Collects search results from other engines after each search request,
embeds them, and stores in the vector database.
"""

from __future__ import annotations

import typing as t
from threading import Thread

from searx import logger
from searx.hoshimi import HoshimiEngine, HoshimiResult
from searx.result_types import MainResult, LegacyResult

logger = logger.getChild("hoshimi.collector")


class ResultCollector:
    """Collects and caches search results from other engines."""

    def __init__(self, engine: HoshimiEngine):
        self.engine = engine
        self.source_weights = engine.config.get("source_weights", {})

    def get_engine_weight(self, engine_name: str) -> float:
        """Get the weight for a source engine.

        First checks the manual source_weights config, then falls back
        to the engine's weight defined in SearXNG's global settings.
        Default is 1.0 if neither is found.
        """
        # 1. Check manual config override
        if engine_name in self.source_weights:
            return self.source_weights[engine_name]

        # 2. Read from SearXNG engine settings
        try:
            import searx.engines
            eng = searx.engines.engines.get(engine_name)
            if eng is not None:
                return float(getattr(eng, 'weight', 1.0))
        except Exception:
            pass

        return 1.0

    def collect_from_results(self, result_container) -> list[HoshimiResult]:
        """Extract results from a SearXNG ResultContainer."""
        hoshimi_results = []
        seen_urls = set()

        # Get ordered results
        ordered = result_container.get_ordered_results()

        for result in ordered:
            url = result.url or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            engine_name = getattr(result, "engine", "") or ""
            # Skip hoshimi's own results
            if engine_name == "hoshimi":
                continue

            title = getattr(result, "title", "") or ""
            content = getattr(result, "content", "") or ""
            thumbnail = getattr(result, "thumbnail", "") or ""
            if not thumbnail:
                thumbnail = getattr(result, "img_src", "") or ""

            # Get category from engine
            category = ""
            if engine_name:
                import searx.engines
                eng = searx.engines.engines.get(engine_name)
                if eng and eng.categories:
                    category = eng.categories[0]

            engine_weight = self.get_engine_weight(engine_name)

            hoshimi_results.append(HoshimiResult(
                url=url,
                title=title,
                content=content,
                thumbnail=thumbnail,
                engine=engine_name,
                category=category,
                engine_weight=engine_weight,
            ))

        logger.debug("Hoshimi: Extracted %d results for caching", len(hoshimi_results))
        return hoshimi_results

    def collect_async(self, result_container):
        """Collect results asynchronously to not block the search response."""
        def _collect():
            try:
                results = self.collect_from_results(result_container)
                if results:
                    self.engine.add_results(results)
                    logger.info("Hoshimi: Collected and cached %d results", len(results))
            except Exception as e:
                logger.error("Hoshimi: Failed to collect results: %s", e)

        thread = Thread(target=_collect, daemon=True)
        thread.start()
