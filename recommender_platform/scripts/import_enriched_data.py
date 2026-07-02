import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db import models
from app.db.session import SessionLocal, engine


COLLECTION_NAME = "products"
EXPECTED_COLUMNS = [
    "product_id", "title", "brand", "category", "subcategory", "description",
    "features", "specifications", "keywords/tags", "price", "rating",
    "review_count", "popularity", "best_seller", "availability", "image_url",
    "product_url",
]


def _point_id(product_id: str) -> int:
    return int(hashlib.md5(product_id.encode("utf-8")).hexdigest()[:16], 16)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _migrate_schema(db: Session) -> None:
    statements = [
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS product_id VARCHAR",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS brand VARCHAR",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS subcategory VARCHAR",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS features JSONB",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS specifications JSONB",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS keywords_tags TEXT",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS rating DOUBLE PRECISION",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS review_count INTEGER",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS popularity DOUBLE PRECISION",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS best_seller BOOLEAN",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS availability VARCHAR",
        "ALTER TABLE items ADD COLUMN IF NOT EXISTS image_url VARCHAR",
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_items_product_id ON items (product_id)",
        "CREATE INDEX IF NOT EXISTS ix_items_brand ON items (brand)",
        "CREATE INDEX IF NOT EXISTS ix_items_subcategory ON items (subcategory)",
        "CREATE INDEX IF NOT EXISTS ix_items_popularity ON items (popularity)",
        "CREATE INDEX IF NOT EXISTS ix_items_availability ON items (availability)",
    ]
    for statement in statements:
        db.execute(text(statement))
    db.commit()


def _read_rows(csv_path: Path, bought_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected enriched CSV columns: {reader.fieldnames}")
        for line_number, source_row in enumerate(reader, start=2):
            product_id = source_row["product_id"].strip()
            if not product_id or product_id in seen:
                raise ValueError(f"Invalid or duplicate product_id at line {line_number}")
            features = json.loads(source_row["features"])
            specifications = json.loads(source_row["specifications"])
            if len(features) < 4 or len(specifications) < 3:
                raise ValueError(f"Insufficient enrichment at line {line_number}")
            rating = float(source_row["rating"])
            reviews = int(source_row["review_count"])
            best_seller = _parse_bool(source_row["best_seller"])
            row = {
                "asin": product_id,
                "product_id": product_id,
                "title": source_row["title"].strip(),
                "brand": source_row["brand"].strip(),
                "category": source_row["category"].strip(),
                "subcategory": source_row["subcategory"].strip(),
                "description": source_row["description"].strip(),
                "features": features,
                "specifications": specifications,
                "keywords_tags": source_row["keywords/tags"].strip(),
                "price": float(source_row["price"]),
                "stars": rating,
                "rating": rating,
                "reviews": reviews,
                "review_count": reviews,
                "popularity": float(source_row["popularity"]),
                "bought_in_last_month": bought_counts.get(product_id, 0),
                "is_best_seller": best_seller,
                "best_seller": best_seller,
                "availability": source_row["availability"].strip(),
                "product_url": source_row["product_url"].strip(),
                "img_url": source_row["image_url"].strip(),
                "image_url": source_row["image_url"].strip(),
            }
            if any(value == "" for value in row.values() if isinstance(value, str)):
                raise ValueError(f"Blank value at line {line_number}")
            rows.append(row)
            seen.add(product_id)
    return rows


def _embedding_text(row: Dict[str, Any]) -> str:
    features = ". ".join(row["features"])
    specifications = ". ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in row["specifications"].items()
    )
    return (
        f"{row['title']}. Brand: {row['brand']}. Category: {row['category']}. "
        f"Subcategory: {row['subcategory']}. Features: {features}. "
        f"Specifications: {specifications}. Keywords: {row['keywords_tags']}"
    )


def _reset_catalog(db: Session, qdrant: QdrantClient, vector_size: int) -> None:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=qmodels.Distance.COSINE,
            on_disk=True,
        ),
        hnsw_config=qmodels.HnswConfigDiff(on_disk=True),
        optimizers_config=qmodels.OptimizersConfigDiff(memmap_threshold=20_000),
        on_disk_payload=True,
    )
    db.execute(text("TRUNCATE TABLE interactions, items RESTART IDENTITY CASCADE"))
    db.commit()


def _write_batch(
    db: Session,
    qdrant: QdrantClient,
    model: SentenceTransformer,
    rows: List[Dict[str, Any]],
    embedding_batch_size: int,
) -> None:
    vectors = model.encode(
        [_embedding_text(row) for row in rows],
        batch_size=embedding_batch_size,
        show_progress_bar=False,
    ).tolist()
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=_point_id(row["product_id"]),
                vector=vector,
                payload=dict(row),
            )
            for row, vector in zip(rows, vectors)
        ],
    )
    db.execute(pg_insert(models.Item).values(rows))
    db.commit()


def _create_view_and_constraints(db: Session) -> None:
    required_columns = [
        "product_id", "title", "brand", "category", "subcategory", "description",
        "features", "specifications", "keywords_tags", "price", "rating",
        "review_count", "popularity", "best_seller", "availability", "image_url",
        "product_url",
    ]
    for column in required_columns:
        db.execute(text(f"ALTER TABLE items ALTER COLUMN {column} SET NOT NULL"))
    db.execute(text("""
        CREATE OR REPLACE VIEW enriched_products AS
        SELECT
            product_id,
            title,
            brand,
            category,
            subcategory,
            description,
            features,
            specifications,
            keywords_tags AS "keywords/tags",
            price,
            rating,
            review_count,
            popularity,
            best_seller,
            availability,
            image_url,
            product_url
        FROM items
    """))
    db.commit()


def _verify(db: Session, qdrant: QdrantClient, expected: int) -> None:
    db_count = int(db.query(func.count(models.Item.id)).scalar() or 0)
    distinct_products = int(db.query(func.count(func.distinct(models.Item.product_id))).scalar() or 0)
    categories = int(db.query(func.count(func.distinct(models.Item.category))).scalar() or 0)
    qdrant_count = int(qdrant.count(collection_name=COLLECTION_NAME, exact=True).count)
    missing_enrichment = int(db.execute(text("""
        SELECT count(*) FROM items
        WHERE product_id IS NULL OR brand IS NULL OR subcategory IS NULL
           OR description IS NULL OR features IS NULL OR specifications IS NULL
           OR keywords_tags IS NULL OR rating IS NULL OR review_count IS NULL
           OR popularity IS NULL OR best_seller IS NULL OR availability IS NULL
           OR image_url IS NULL OR product_url IS NULL
    """)).scalar() or 0)
    report = {
        "postgres_rows": db_count,
        "distinct_product_ids": distinct_products,
        "categories": categories,
        "qdrant_points": qdrant_count,
        "missing_enriched_values": missing_enrichment,
    }
    print("\nFinal enriched catalog verification")
    for key, value in report.items():
        print(f"  {key}: {value:,}")
    if db_count != expected or distinct_products != expected or qdrant_count != expected or missing_enrichment:
        raise RuntimeError(f"Enriched catalog verification failed: {report}")


def import_enriched(csv_path: Path, batch_size: int, embedding_batch_size: int) -> None:
    models.Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    try:
        _migrate_schema(db)
        bought_counts = dict(db.query(models.Item.asin, models.Item.bought_in_last_month).all())
        rows = _read_rows(csv_path, bought_counts)
        print(f"Validated {len(rows):,} enriched products before reset.")
        model = SentenceTransformer(settings.SBERT_MODEL)
        _reset_catalog(db, qdrant, int(model.get_embedding_dimension()))
        for start in tqdm(range(0, len(rows), batch_size), desc="Importing enriched products"):
            _write_batch(db, qdrant, model, rows[start:start + batch_size], embedding_batch_size)
        _create_view_and_constraints(db)
        _verify(db, qdrant, len(rows))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        qdrant.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import enriched products into PostgreSQL and Qdrant.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()
    import_enriched(args.csv, args.batch_size, args.embedding_batch_size)


if __name__ == "__main__":
    main()
