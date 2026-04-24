# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoshimi Cache Plugin - Caches search results from all engines.

This plugin intercepts results from all search engines and stores them in a
local SQLite database, which the Hoshimi engine can then serve as cached results.
"""

import typing as t
from collections import defaultdict

from flask_babel import gettext

from searx.plugins import Plugin, PluginInfo
from searx.result_types import LegacyResult

if t.TYPE_CHECKING:
    from searx.plugins import PluginCfg
    from searx.extended_types import SXNG_Request
    from searx.search import SearchWithPlugins


# Track result positions per engine per query
_result_position_counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))


def _get_position(query: str, engine_name: str) -> int:
    """Get and increment the position for a result from a specific engine."""
    key = f"{query}:{engine_name}"
    _result_position_counters[key][engine_name] += 1
    return _result_position_counters[key][engine_name]


@t.final
class SXNGPlugin(Plugin):
    """Plugin that caches results from all engines to Hoshimi's SQLite database."""

    id = "hoshimi_cache"

    def __init__(self, plg_cfg: "PluginCfg") -> None:
        super().__init__(plg_cfg)

        self.info = PluginInfo(
            id=self.id,
            name=gettext("Hoshimi Cache"),
            description=gettext("Caches search results from all engines for local searching"),
        )

    def init(self, app) -> bool:
        """Initialize the plugin."""
        try:
            from searx.engines.hoshimi import init_db, _db_paths
            # Initialize all category databases
            for category in _db_paths.keys():
                init_db(category)
            self.log.info("Hoshimi cache plugin initialized")
        except Exception as e:  # pylint: disable=broad-except
            self.log.warning("Hoshimi cache plugin init failed: %s", e)
        return True

    def pre_search(self, request: "SXNG_Request", search) -> bool:
        """Reset position counters before each search."""
        global _result_position_counters
        _result_position_counters = defaultdict(lambda: defaultdict(int))
        return True

    def on_result(self, request: "SXNG_Request", search: "SearchWithPlugins", result) -> bool:
        """Cache each result from all engines to Hoshimi's database."""
        try:
            from searx.engines.hoshimi import store_results
            from searx import engines as searx_engines

            # Get engine name from the result
            if isinstance(result, dict):
                engine_name = result.get("engine", "unknown")
                result_dict = dict(result)
            else:
                engine_name = getattr(result, "engine", "unknown")
                # Convert object to dict with all possible fields
                result_dict = {}
                # First, get all attributes that might exist
                for attr in dir(result):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(result, attr, None)
                            if val is not None and isinstance(val, (str, int, float, bool, type(None))):
                                result_dict[attr] = val
                        except (AttributeError, TypeError):
                            pass
                # Then get specific important fields
                for attr in ["title", "url", "content", "img_src", "thumbnail", "publishedDate",
                             "template", "iframe_src", "length", "duration", "resolution",
                             "author", "creator", "views", "metadata"]:
                    val = getattr(result, attr, None)
                    if val is not None:
                        result_dict[attr] = val
                # Handle parsed_url - convert to string for storage
                if hasattr(result, 'parsed_url') and result.parsed_url is not None:
                    try:
                        result_dict['parsed_url'] = result.parsed_url.geturl()
                    except (AttributeError, TypeError):
                        result_dict['parsed_url'] = str(result.parsed_url)

            # Get query from search
            query = search.search_query.query

            # Determine the category from the engine that produced this result
            category = "general"
            if engine_name != "unknown" and engine_name != "hoshimi":
                eng = getattr(searx_engines, 'engines', {}).get(engine_name)
                if eng and hasattr(eng, 'categories') and eng.categories:
                    category = eng.categories[0]

            # Get position using our own counter
            position = _get_position(query, engine_name)

            # Store the result with engine name and position
            if result_dict.get("url") and engine_name != "hoshimi":
                # Don't cache results from hoshimi itself to avoid loops
                store_results(query, category, [result_dict], engine_name, position)

        except Exception as e:  # pylint: disable=broad-except
            # Don't let caching errors break the search
            self.log.debug("Hoshimi cache error: %s", e)

        return True  # Always keep the result
