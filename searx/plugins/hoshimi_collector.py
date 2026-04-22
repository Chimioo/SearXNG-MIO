# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoshimi Result Collector Plugin.

This plugin collects search results from other engines after each search
and caches them in the Hoshimi vector database for future semantic searches.
"""

from __future__ import annotations

import typing as t

from searx.extended_types import SXNG_Request
from searx.plugins._core import Plugin, PluginCfg, PluginInfo
from searx.search import SearchWithPlugins

if t.TYPE_CHECKING:
    from searx.result_types import Result


class SXNGPlugin(Plugin):
    """Hoshimi result collector plugin."""

    id = "hoshimi_collector"
    active = True
    info = PluginInfo(
        id=id,
        name="Hoshimi Collector",
        description="Caches search results from all engines for semantic search",
        preference_section="general",
    )

    def __init__(self, plg_cfg: PluginCfg):
        super().__init__(plg_cfg)
        self._engine_cache = None

    def init(self, app) -> bool:  # pylint: disable=unused-argument
        """Initialize the plugin."""
        # Always return True - the engine may not be loaded yet during Flask init,
        # but it will be available when actual searches happen.
        self.log.info("Hoshimi collector plugin registered")
        return True

    def _get_engine(self):
        """Lazy-load the hoshimi engine."""
        if self._engine_cache is not None:
            return self._engine_cache
        try:
            # CRITICAL: Must use the module from searx.engines.engines dict,
            # NOT import searx.engines.hoshimi directly. SearXNG loads engine
            # modules with short names (e.g. 'hoshimi') which creates a separate
            # module object from 'searx.engines.hoshimi'. The init() function
            # sets _hoshimi_engine on the short-name module.
            from searx.engines import engines
            hoshimi_mod = engines.get('hoshimi')
            if hoshimi_mod is None:
                self.log.warning("Hoshimi collector: hoshimi engine not found in engines dict")
                return None

            # Get the engine instance from the module
            self._engine_cache = getattr(hoshimi_mod, '_hoshimi_engine', None)
            if self._engine_cache is not None:
                self.log.debug("Hoshimi collector: engine found and cached")
            else:
                self.log.debug("Hoshimi collector: _hoshimi_engine is None (init not completed?)")
            return self._engine_cache
        except Exception as e:
            self.log.warning("Hoshimi collector: engine not available: %s", e)
            return None

    def post_search(
        self, request: SXNG_Request, search: SearchWithPlugins
    ) -> t.List[Result] | None:  # pylint: disable=unused-argument
        """Collect results after search completes."""
        try:
            engine = self._get_engine()
            if engine is None:
                return None

            # Collect results asynchronously
            engine.collector.collect_async(search.result_container)
        except Exception as e:
            self.log.error("Hoshimi collector post_search error: %s", e)

        return None
