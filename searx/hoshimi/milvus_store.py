# SPDX-License-Identifier: AGPL-3.0-or-later
"""Milvus Lite vector store implementation for Hoshimi."""

from __future__ import annotations

import os
import threading
import typing as t

from searx import logger
from searx.hoshimi.vector_store import VectorStore

if t.TYPE_CHECKING:
    from searx.hoshimi import HoshimiResult

logger = logger.getChild("hoshimi.milvus")

# Milvus dimension for all-MiniLM-L6-v2
DEFAULT_DIM = 384
COLLECTION_NAME = "hoshimi_results"

# Module-level singleton for the Milvus client.
# Milvus Lite does not support multiple connections to the same database file,
# so we share a single client across all MilvusStore instances.
_shared_client = None
_shared_db_path = None
_shared_client_pid = None  # Track which PID created the client
_init_lock = threading.Lock()  # Thread-safe lock for client initialization
_collection_initialized = False  # Track if collection has been created


def _get_milvus_client(db_path: str, max_retries: int = 3, retry_delay: float = 0.5):
    """Get or create a shared Milvus client for the given database path."""
    global _shared_client, _shared_db_path, _shared_client_pid  # pylint: disable=global-statement

    current_pid = os.getpid()

    # Fast path: client already exists for this process
    if _shared_client is not None and _shared_db_path == db_path and _shared_client_pid == current_pid:
        return _shared_client

    # Slow path: need to initialize
    with _init_lock:
        # Double-check after acquiring lock
        if _shared_client is not None and _shared_db_path == db_path and _shared_client_pid == current_pid:
            return _shared_client

        # Detect if we're in a forked process (different PID from when client was created)
        if _shared_client is not None and _shared_client_pid != current_pid:
            logger.debug("Hoshimi: Detected forked process (PID %s != %s), resetting Milvus client", _shared_client_pid, current_pid)
            _shared_client = None
            _shared_db_path = None
            _shared_client_pid = None
            _collection_initialized = False  # Reset collection flag for new process

        from pymilvus import MilvusClient

        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        logger.info("Hoshimi: Initializing Milvus client with db_path=%s (PID %s)", db_path, current_pid)

        # Retry loop for Milvus Lite startup
        last_error = None
        for attempt in range(max_retries):
            try:
                _shared_client = MilvusClient(uri=db_path)
                # Test the connection
                _shared_client.has_collection(collection_name="__ping__")
                _shared_db_path = db_path
                _shared_client_pid = current_pid
                logger.info("Hoshimi: Milvus client initialized successfully")
                return _shared_client
            except Exception as e:  # pylint: disable=broad-except
                last_error = e
                logger.warning("Hoshimi: Milvus client init attempt %d/%d failed: %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)

        raise Exception(f"Failed to initialize Milvus client after {max_retries} attempts") from last_error


class MilvusStore(VectorStore):
    """Milvus Lite vector store for Hoshimi search results."""

    def __init__(self, config: dict[str, t.Any]):
        self.config = config
        db_path = config.get("milvus_uri", "./data/hoshimi.milvus")
        # Milvus Lite requires .db extension for local files
        if not db_path.endswith(".db"):
            db_path = db_path.rsplit(".", 1)[0] + ".db" if "." in db_path else db_path + ".db"
        self.db_path = db_path
        self.similarity_threshold = config.get("similarity_threshold", 0.6)
        self._collection = None
        self._dim = config.get("embedding_dim", DEFAULT_DIM)

    @property
    def client(self):
        return _get_milvus_client(self.db_path)

    def ensure_initialized(self) -> bool:
        global _collection_initialized  # pylint: disable=global-statement

        # Fast path: collection already initialized
        if _collection_initialized:
            self._collection = COLLECTION_NAME
            return True

        with _init_lock:
            # Double-check after acquiring lock
            if _collection_initialized:
                self._collection = COLLECTION_NAME
                return True

            try:
                if self.client.has_collection(collection_name=COLLECTION_NAME):
                    self._collection = COLLECTION_NAME
                    _collection_initialized = True
                    logger.debug("Hoshimi: Milvus collection '%s' already exists", COLLECTION_NAME)
                    return True

                from pymilvus import DataType

                schema = self.client.create_schema(
                    auto_id=True,
                    enable_dynamic_field=True,
                )
                schema.add_field("id", DataType.INT64, is_primary=True)
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
                schema.add_field("url", DataType.VARCHAR, max_length=2048)
                schema.add_field("title", DataType.VARCHAR, max_length=1024)
                schema.add_field("content", DataType.VARCHAR, max_length=4096)
                schema.add_field("thumbnail", DataType.VARCHAR, max_length=2048)
                schema.add_field("engine", DataType.VARCHAR, max_length=128)
                schema.add_field("category", DataType.VARCHAR, max_length=128)
                schema.add_field("engine_weight", DataType.FLOAT)

                index_params = self.client.prepare_index_params()
                index_params.add_index(
                    field_name="vector",
                    index_name="vector_idx",
                    index_type="AUTOINDEX",
                    metric_type="COSINE",
                )

                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    schema=schema,
                    index_params=index_params,
                )
                self._collection = COLLECTION_NAME
                _collection_initialized = True
                logger.info("Hoshimi: Milvus collection '%s' created", COLLECTION_NAME)
                return True
            except Exception as e:
                logger.error("Hoshimi: Failed to initialize Milvus store: %s", e)
                return False

    def _get_existing_urls(self, urls: list[str]) -> set[str]:
        """Check which URLs already exist in the database."""
        if not urls:
            return set()

        existing = set()
        # Query in batches to avoid overly long filter expressions
        batch_size = 50
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            # Build filter expression: url in ["url1", "url2", ...]
            url_list = '","'.join(batch)
            filter_expr = f'url in ["{url_list}"]'
            try:
                results = self.client.query(
                    collection_name=COLLECTION_NAME,
                    filter=filter_expr,
                    output_fields=["url"],
                    limit=len(batch),
                )
                for r in results:
                    existing.add(r.get("url", ""))
            except Exception as e:
                logger.debug("Hoshimi: Failed to query existing URLs: %s", e)

        return existing

    def add(self, results: list[HoshimiResult]) -> bool:
        if not results:
            return True

        # Ensure collection exists before adding
        if not self.ensure_initialized():
            logger.error("Hoshimi: Failed to initialize Milvus store")
            return False

        try:
            # Dedup: filter out URLs that already exist in the database
            urls_to_check = [r.url for r in results if r.vector]
            existing_urls = self._get_existing_urls(urls_to_check)
            if existing_urls:
                logger.debug("Hoshimi: Skipping %d existing URLs", len(existing_urls))

            data = []
            skipped = 0
            for r in results:
                if not r.vector:
                    continue
                if r.url in existing_urls:
                    skipped += 1
                    continue
                data.append({
                    "vector": r.vector,
                    "url": r.url,
                    "title": r.title,
                    "content": r.content,
                    "thumbnail": r.thumbnail,
                    "engine": r.engine,
                    "category": r.category,
                    "engine_weight": r.engine_weight,
                })

            if not data:
                if skipped:
                    logger.debug("Hoshimi: All %d results already exist, skipping", skipped)
                return True

            self.client.insert(
                collection_name=COLLECTION_NAME,
                data=data,
            )
            logger.debug("Hoshimi: Added %d results to Milvus (%d duplicates skipped)", len(data), skipped)
            return True
        except Exception as e:
            logger.error("Hoshimi: Failed to add results to Milvus: %s", e)
            return False

    def keyword_search(self, query: str, top_k: int = 20, category: str = "") -> list[dict[str, t.Any]]:
        """Search using keyword matching on title and content fields.

        Splits the query into individual terms and uses LIKE matching
        on title and content fields. Results are ranked by match count.
        """
        try:
            # Split query into individual terms
            terms = [t.strip().lower() for t in query.split() if len(t.strip()) > 1]
            if not terms:
                return []

            # Build category filter
            filter_expr = ""
            if category:
                filter_expr = f'category == "{category}"'

            # Query all records (with a reasonable limit)
            all_results = self.client.query(
                collection_name=COLLECTION_NAME,
                filter=filter_expr,
                limit=max(top_k * 5, 100),
                output_fields=["id", "url", "title", "content", "thumbnail", "engine", "category", "engine_weight"],
            )

            if not all_results:
                return []

            # Score each result based on keyword matches
            scored = []
            for r in all_results:
                title = (r.get("title", "") or "").lower()
                content = (r.get("content", "") or "").lower()

                score = 0.0
                matched_terms = 0
                for term in terms:
                    title_match = term in title
                    content_match = term in content
                    if title_match:
                        score += 2.0  # Title match is more important
                        matched_terms += 1
                    if content_match:
                        score += 1.0
                        matched_terms += 1

                if score > 0:
                    scored.append({
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "content": r.get("content", ""),
                        "thumbnail": r.get("thumbnail", ""),
                        "engine": r.get("engine", ""),
                        "category": r.get("category", ""),
                        "engine_weight": r.get("engine_weight", 1.0),
                        "score": score / len(terms),  # Normalize score
                        "matched_terms": matched_terms,
                    })

            # Sort by score descending and return top_k
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.error("Hoshimi: Failed to keyword search in Milvus: %s", e)
            return []

    def search(self, query_vector: list[float], top_k: int = 20, category: str = "") -> list[dict[str, t.Any]]:
        try:
            # Build category filter
            filter_expr = ""
            if category:
                filter_expr = f'category == "{category}"'

            results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector],
                limit=top_k,
                filter=filter_expr,
                output_fields=["url", "title", "content", "thumbnail", "engine", "category", "engine_weight"],
                search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            )

            if not results:
                return []

            ret = []
            for hits in results:
                for hit in hits:
                    distance = hit.get("distance", 0)
                    if distance < self.similarity_threshold:
                        continue
                    ret.append({
                        **hit.get("entity", {}),
                        "score": distance,
                    })
            return ret
        except Exception as e:
            logger.error("Hoshimi: Failed to search Milvus: %s", e)
            return []

    def close(self):
        global _shared_client, _shared_db_path, _shared_client_pid, _collection_initialized  # pylint: disable=global-statement
        if _shared_client:
            _shared_client.close()
            _shared_client = None
            _shared_db_path = None
            _shared_client_pid = None
            _collection_initialized = False
            self._collection = None
