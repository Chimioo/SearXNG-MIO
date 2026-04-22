#!/usr/bin/env python3
"""Migrate Hoshimi Milvus database to a new embedding model.

Reads existing records (title, content, url, etc.) from the old database,
re-embeds them with the new model, and writes to a new database.

Usage:
    python searx/hoshimi/migrate_model.py [old_db_path] [new_db_path]

Defaults:
    old_db_path: ./data/hoshimi.db
    new_db_path: ./data/hoshimi_new.db
"""

import sys
import os

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault('SEARXNG_SETTINGS_PATH', 'searx/settings.yml')

OLD_DB = sys.argv[1] if len(sys.argv) > 1 else "./data/hoshimi.db"
NEW_DB = sys.argv[2] if len(sys.argv) > 2 else "./data/hoshimi_new.db"
COLLECTION_NAME = "hoshimi_results"

# New model config
NEW_MODEL = "intfloat/multilingual-e5-small"
NEW_DIM = 384


def main():
    from pymilvus import connections, Collection, utility, MilvusClient, DataType
    from sentence_transformers import SentenceTransformer

    # Connect to old database
    print(f"Connecting to old database: {OLD_DB}")
    old_client = MilvusClient(uri=OLD_DB)

    if not old_client.has_collection(collection_name=COLLECTION_NAME):
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found in {OLD_DB}")
        sys.exit(1)

    # Read all existing records (batched, max 16384 per query)
    print("Reading existing records...")
    all_records = []
    batch_limit = 16000
    offset = 0
    while True:
        batch = old_client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            limit=batch_limit,
            offset=offset,
            output_fields=["url", "title", "content", "thumbnail", "engine", "category", "engine_weight"],
        )
        if not batch:
            break
        all_records.extend(batch)
        offset += len(batch)
        if len(batch) < batch_limit:
            break

    old_records = all_records
    print(f"Found {len(old_records)} records")

    if not old_records:
        print("No records to migrate")
        sys.exit(0)

    # Load new embedding model
    print(f"Loading new model: {NEW_MODEL}")
    model = SentenceTransformer(NEW_MODEL)

    # Re-embed all records
    print("Re-embedding records...")
    texts = [f"{r.get('title', '')} {r.get('content', '')}" for r in old_records]

    # Batch embed (handle empty texts)
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    valid_texts = [texts[i] for i in valid_indices]

    if valid_texts:
        embeddings = model.encode(valid_texts, convert_to_numpy=True)
        vectors_map = {i: embeddings[j].tolist() for j, i in enumerate(valid_indices)}
    else:
        vectors_map = {}

    # Create new database
    print(f"Creating new database: {NEW_DB}")
    if os.path.exists(NEW_DB):
        os.remove(NEW_DB)

    new_client = MilvusClient(uri=NEW_DB)
    os.makedirs(os.path.dirname(NEW_DB) if os.path.dirname(NEW_DB) else ".", exist_ok=True)

    schema = new_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=NEW_DIM)
    schema.add_field("url", DataType.VARCHAR, max_length=2048)
    schema.add_field("title", DataType.VARCHAR, max_length=1024)
    schema.add_field("content", DataType.VARCHAR, max_length=4096)
    schema.add_field("thumbnail", DataType.VARCHAR, max_length=2048)
    schema.add_field("engine", DataType.VARCHAR, max_length=128)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("engine_weight", DataType.FLOAT)

    index_params = new_client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_name="vector_idx",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    new_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    # Insert re-embedded data
    print("Inserting re-embedded records...")
    data = []
    skipped = 0
    for r in old_records:
        url = r.get("url", "")
        if url in vectors_map:
            data.append({
                "vector": vectors_map[url],
                "url": url,
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "thumbnail": r.get("thumbnail", ""),
                "engine": r.get("engine", ""),
                "category": r.get("category", ""),
                "engine_weight": r.get("engine_weight", 1.0),
            })
        else:
            skipped += 1

    if data:
        new_client.insert(
            collection_name=COLLECTION_NAME,
            data=data,
        )

    print(f"Migration complete!")
    print(f"  Total records: {len(old_records)}")
    print(f"  Migrated: {len(data)}")
    print(f"  Skipped (empty text): {skipped}")
    print(f"  New database: {NEW_DB}")
    print(f"\nTo use the new database, replace {OLD_DB} with {NEW_DB}")


if __name__ == "__main__":
    main()
