# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hoshimi - Local search engine that caches results from other engines.

Hoshimi intercepts search results from all other engines in SearXNG and stores
them in a local SQLite database. Subsequent searches return cached results from
this database.

Features:
- SQLite FTS5 contentless mode for efficient full-text search
- jieba Chinese word segmentation
- Ranking: rank × log(1+hits) × time_decay (30-day half-life)
- Configurable fallback threshold for network search
- WAL mode for concurrent read/write
"""

import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from datetime import datetime

from searx import logger, settings

logger = logger.getChild("hoshimi")

# about
about = {
    "website": "https://github.com/searxng/searxng",
    "wikidata_id": None,
    "official_api_documentation": None,
    "use_official_api": False,
    "require_api_key": False,
    "results": "HTML",
}

# Engine configuration
engine_type = "offline"
categories = ["general"]
paging = True
max_page = 10
time_range_support = True

# Data directory for the SQLite database
_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
# Each category has its own database file
_db_paths = {
    "general": os.path.join(_data_dir, "hoshimi_general.db"),
    "images": os.path.join(_data_dir, "hoshimi_images.db"),
    "videos": os.path.join(_data_dir, "hoshimi_videos.db"),
    "news": os.path.join(_data_dir, "hoshimi_news.db"),
    "map": os.path.join(_data_dir, "hoshimi_map.db"),
    "music": os.path.join(_data_dir, "hoshimi_music.db"),
    "it": os.path.join(_data_dir, "hoshimi_it.db"),
    "science": os.path.join(_data_dir, "hoshimi_science.db"),
    "files": os.path.join(_data_dir, "hoshimi_files.db"),
    "social media": os.path.join(_data_dir, "hoshimi_social_media.db"),
}
_lock = threading.Lock()

# Search syntax patterns
SEARCH_SYNTAX_PATTERNS = {
    "site:": re.compile(r'\bsite:(\S+)', re.IGNORECASE),
    "intitle:": re.compile(r'\bintitle:(\S+)', re.IGNORECASE),
    "inurl:": re.compile(r'\binurl:(\S+)', re.IGNORECASE),
    "intext:": re.compile(r'\bintext:(\S+)', re.IGNORECASE),
    "filetype:": re.compile(r'\b(?:filetype|ext):(\S+)', re.IGNORECASE),
}

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS cached_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    engine TEXT,
    category TEXT,
    template TEXT,
    thumbnail TEXT,
    img_src TEXT,
    iframe_src TEXT,
    length TEXT,
    published_date INTEGER,
    position INTEGER DEFAULT 0,
    result_data TEXT,
    hits INTEGER DEFAULT 1,
    cached_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    last_used INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
    token,
    tokenize='unicode61'
);

CREATE INDEX IF NOT EXISTS idx_url ON cached_results(url);
CREATE INDEX IF NOT EXISTS idx_cached_at ON cached_results(cached_at);
CREATE INDEX IF NOT EXISTS idx_last_used ON cached_results(last_used);
CREATE INDEX IF NOT EXISTS idx_category ON cached_results(category);
"""

DEFAULT_SETTINGS = {
    "fallback_threshold": 5,
    "expire_days": 30,
}


def _get_settings() -> dict:
    """Get hoshimi settings from searx settings."""
    hoshimi_settings = settings.get('hoshimi', {})
    result = DEFAULT_SETTINGS.copy()
    result.update(hoshimi_settings)
    return result


def _get_connection(category="general"):
    """Get a thread-safe SQLite connection for the specified category."""
    db_path = _db_paths.get(category, _db_paths["general"])
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db(category="general"):
    """Initialize the database schema for a specific category."""
    db_path = _db_paths.get(category, _db_paths["general"])
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = _get_connection(category)
    try:
        conn.executescript(DB_SCHEMA)
        logger.info("Hoshimi %s database initialized at %s", category, db_path)
    finally:
        conn.close()


def _tokenize_for_fts(text: str) -> str:
    """Tokenize text using jieba and return space-separated tokens for FTS5."""
    if not text:
        return ""
    try:
        import jieba
        tokens = list(jieba.cut(text))
        return " ".join(t.strip().lower() for t in tokens if t.strip() and len(t.strip()) >= 2)
    except ImportError:
        return text.lower()


def _tokenize(text: str) -> list:
    """Tokenize text using jieba and return list of tokens."""
    if not text:
        return []
    try:
        import jieba
        tokens = list(jieba.cut(text))
        return [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) >= 2]
    except ImportError:
        import re
        return [t.lower() for t in re.findall(r'\b[a-zA-Z0-9\u4e00-\u9fff]+\b', text) if len(t) >= 2]


def parse_search_syntax(query: str) -> dict:
    """Parse basic search engine syntax from the query."""
    filters = {}
    for pattern_name, pattern in SEARCH_SYNTAX_PATTERNS.items():
        matches = pattern.findall(query)
        if matches:
            key = pattern_name.replace(":", "")
            filters[key] = matches
    # Remove syntax patterns from query to get the base query
    base_query = query
    for pattern in SEARCH_SYNTAX_PATTERNS.values():
        base_query = pattern.sub("", base_query).strip()
    filters["base_query"] = base_query
    return filters


def store_results(query: str, category: str, results: list, engine: str = "mixed", position: int = 0):
    """Store search results in the database. Updates hits if URL already exists."""
    if not results:
        return

    db_path = _db_paths.get(category, _db_paths["general"])

    if not os.path.exists(db_path):
        init_db(category)

    now = int(time.time())

    with _lock:
        conn = _get_connection(category)
        try:
            for idx, result in enumerate(results):
                url = result.get("url", "")
                if not url:
                    continue

                title = result.get("title", "")
                content = result.get("content", "")
                template = result.get("template", "")
                thumbnail = result.get("thumbnail", "") or result.get("img_src", "")
                img_src = result.get("img_src", "")
                iframe_src = result.get("iframe_src", "")
                length = result.get("length", "")
                published_date = result.get("publishedDate")

                if published_date:
                    try:
                        if isinstance(published_date, datetime):
                            published_date = int(published_date.timestamp())
                        elif isinstance(published_date, str):
                            dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                            published_date = int(dt.timestamp())
                    except (ValueError, TypeError):
                        published_date = None

                result_position = position if position > 0 else idx + 1

                serializable_result = {}
                for k, v in result.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        serializable_result[k] = v
                    elif isinstance(v, urllib.parse.ParseResult):
                        serializable_result[k] = v.geturl()
                    elif isinstance(v, (list, dict)):
                        try:
                            json.dumps(v, default=str)
                            serializable_result[k] = v
                        except (TypeError, ValueError):
                            pass

                result_data = json.dumps(serializable_result, default=str)

                cursor = conn.execute(
                    "SELECT id FROM cached_results WHERE url = ?",
                    (url,)
                )
                existing = cursor.fetchone()

                if existing:
                    doc_id = existing[0]
                    conn.execute(
                        """UPDATE cached_results SET
                           title=?, snippet=?, hits=hits+1, cached_at=?, last_used=?,
                           result_data=?, position=?
                           WHERE url=?""",
                        (title, content, now, now, result_data, result_position, url)
                    )
                    conn.execute("DELETE FROM search_index WHERE rowid=?", (doc_id,))
                else:
                    cursor = conn.execute(
                        """INSERT INTO cached_results
                           (url, title, snippet, engine, category, template, thumbnail,
                            img_src, iframe_src, length, published_date, position,
                            result_data, hits, cached_at, last_used)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                        (url, title, content, engine, category, template, thumbnail,
                         img_src, iframe_src, length, published_date, result_position,
                         result_data, now, now)
                    )
                    doc_id = cursor.lastrowid

                tokens_title = _tokenize_for_fts(title)
                tokens_snippet = _tokenize_for_fts(content)
                tokens_url = _tokenize_for_fts(url)

                all_tokens = f"{tokens_title} {tokens_snippet} {tokens_url}".strip()
                if all_tokens:
                    conn.execute(
                        "INSERT INTO search_index(rowid, token) VALUES (?, ?)",
                        (doc_id, all_tokens)
                    )

            conn.commit()
            logger.debug("Hoshimi stored %d results for query: %s", len(results), query[:50])
        except Exception as e:
            logger.error("Hoshimi storage error: %s", e)
            conn.rollback()
        finally:
            conn.close()


def search_cached(query: str, category: str = "general", pageno: int = 1,
                  time_range: str = None, safesearch: int = 0) -> dict:
    """Search cached results.

    Returns dict with:
      - results: list of search results
      - need_fallback: bool, True if results < fallback_threshold
    """
    filters = parse_search_syntax(query)
    base_query = filters.get("base_query", query)

    db_path = _db_paths.get(category, _db_paths["general"])
    if not os.path.exists(db_path):
        return {"results": [], "need_fallback": True}

    hoshimi_settings = _get_settings()
    fallback_threshold = hoshimi_settings.get("fallback_threshold", 5)

    query_tokens = _tokenize(base_query)
    if not query_tokens:
        return {"results": [], "need_fallback": True}

    filter_conditions = []
    filter_params = []

    if "site" in filters:
        for site in filters["site"]:
            filter_conditions.append("url LIKE ?")
            filter_params.append(f"%{site}%")

    if "intitle" in filters:
        for term in filters["intitle"]:
            filter_conditions.append("title LIKE ?")
            filter_params.append(f"%{term}%")

    if "inurl" in filters:
        for term in filters["inurl"]:
            filter_conditions.append("url LIKE ?")
            filter_params.append(f"%{term}%")

    if "filetype" in filters:
        for ext in filters["filetype"]:
            filter_conditions.append("(url LIKE ? OR snippet LIKE ?)")
            filter_params.extend([f"%.{ext}%", f"%.{ext}%"])

    filter_conditions.append("category IN (?, 'general')")
    filter_params.append(category)

    if time_range:
        now = int(time.time())
        time_filters = {
            "day": now - 86400,
            "week": now - 604800,
            "month": now - 2592000,
            "year": now - 31536000,
        }
        if time_range in time_filters:
            filter_conditions.append("(published_date >= ? OR cached_at >= ?)")
            filter_params.extend([time_filters[time_range], time_filters[time_range]])

    filter_clause = " AND ".join(filter_conditions) if filter_conditions else "1=1"

    offset = (pageno - 1) * 10
    limit = max(fallback_threshold, 20)

    results = []
    need_fallback = True

    with _lock:
        conn = _get_connection(category)
        try:
            now = int(time.time())

            try:
                fts_query = " OR ".join([f'"{t}"' for t in query_tokens])

                sql = f"""
                    SELECT r.id, r.url, r.title, r.snippet, r.engine, r.category,
                           r.thumbnail, r.img_src, r.published_date, r.position,
                           r.result_data, r.hits, r.cached_at
                    FROM search_index
                    JOIN cached_results r ON search_index.rowid = r.id
                    WHERE search_index MATCH ?
                    AND {filter_clause}
                    ORDER BY bm25(search_index) ASC, r.last_used DESC
                    LIMIT ? OFFSET ?
                """
                params = [fts_query] + filter_params + [limit, offset]

                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            except Exception as fts_error:
                logger.debug("Hoshimi FTS5 error, falling back to LIKE: %s", fts_error)
                rows = []

            if not rows:
                word_conditions = []
                like_params = []
                for word in query_tokens:
                    word_conditions.append("(title LIKE ? OR snippet LIKE ? OR url LIKE ?)")
                    like_params.extend([f"%{word}%", f"%{word}%", f"%{word}%"])

                fallback_where = "(" + " OR ".join(word_conditions) + f") AND {filter_clause}"
                fallback_sql = f"""
                    SELECT id, url, title, snippet, engine, category,
                           thumbnail, img_src, published_date, position,
                           result_data, hits, cached_at
                    FROM cached_results
                    WHERE {fallback_where}
                    ORDER BY hits DESC, cached_at DESC
                    LIMIT ? OFFSET ?
                """
                fallback_params = like_params + filter_params + [limit, offset]
                cursor = conn.execute(fallback_sql, fallback_params)
                rows = cursor.fetchall()

            for row in rows:
                (doc_id, url, title, snippet, engine, cat, thumbnail,
                 img_src, published_date, position, result_data, hits,
                 cached_at) = row

                days_old = (now - cached_at) / 86400
                time_decay = math.exp(-days_old / 30)
                score_multiplier = math.log(1 + hits) * time_decay

                if result_data:
                    try:
                        result = json.loads(result_data)
                        result["engine"] = "hoshimi"
                        result["hoshimi_position"] = position
                        result["hoshimi_score"] = score_multiplier
                        result["hoshimi_hits"] = hits
                        if "parsed_url" in result and isinstance(result.get("parsed_url"), str):
                            result["parsed_url"] = urllib.parse.urlparse(result["parsed_url"])
                    except (json.JSONDecodeError, TypeError):
                        result = {
                            "title": title,
                            "url": url,
                            "content": snippet or "",
                            "engine": "hoshimi",
                            "category": cat,
                            "hoshimi_position": position,
                            "hoshimi_score": score_multiplier,
                            "hoshimi_hits": hits,
                        }
                else:
                    result = {
                        "title": title,
                        "url": url,
                        "content": snippet or "",
                        "engine": "hoshimi",
                        "category": cat,
                        "hoshimi_position": position,
                        "hoshimi_score": score_multiplier,
                        "hoshimi_hits": hits,
                    }
                    if thumbnail or img_src:
                        result["thumbnail"] = thumbnail or img_src
                    if img_src:
                        result["img_src"] = img_src

                if published_date:
                    try:
                        result["publishedDate"] = datetime.fromtimestamp(published_date)
                    except (ValueError, TypeError, OSError):
                        pass

                if "url" in result and "parsed_url" not in result:
                    try:
                        result["parsed_url"] = urllib.parse.urlparse(result["url"])
                    except Exception:
                        pass

                results.append(result)

            results.sort(key=lambda x: x.get("hoshimi_score", 0), reverse=True)

            if len(results) >= fallback_threshold:
                need_fallback = False

            if pageno == 1 and results:
                result_ids = [r.get("id", 0) for r in results if r.get("id")]
                if result_ids:
                    placeholders = ",".join(["?"] * len(result_ids))
                    conn.execute(
                        f"UPDATE cached_results SET last_used = ? WHERE id IN ({placeholders})",
                        [now] + result_ids
                    )
                    conn.commit()

            logger.debug("Hoshimi found %d cached results for: %s", len(results), query[:50])
        except Exception as e:
            logger.error("Hoshimi search error: %s", e)
        finally:
            conn.close()

    return {"results": results, "need_fallback": need_fallback}


def search(query, params):
    """Search cached results.

    This is called by OfflineProcessor for offline engines.
    Returns list of results (compatible with SearXNG offline engine API).
    """
    category = params.get("category", "general")
    pageno = params.get("pageno", 1)
    time_range = params.get("time_range")
    safesearch = params.get("safesearch", 0)

    db_path = _db_paths.get(category, _db_paths["general"])
    if not os.path.exists(db_path):
        init_db(category)

    result = search_cached(
        query,
        category=category,
        pageno=pageno,
        time_range=time_range,
        safesearch=safesearch
    )

    # Return only the results list (SearXNG expects list for offline engines)
    return result.get("results", [])


# Initialize on module load
try:
    for category in _db_paths.keys():
        init_db(category)
except Exception as e:
    logger.warning("Hoshimi init error (will retry on first search): %s", e)
