import argparse
import hashlib
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

# Add app to path to import models and session
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db import models
from app.db.session import SessionLocal, engine


COLLECTION_NAME = "products"


def _deterministic_point_id(asin: str) -> int:
    # Stable across runs/machines (unlike Python's built-in hash()).
    return int(hashlib.md5(asin.encode("utf-8")).hexdigest()[:16], 16)


def _clean_bool(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def _clean_int(val: Any) -> int:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return int(float(val))
    except Exception:
        return 0


def _clean_float(val: Any) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)
    except Exception:
        return 0.0


def _clean_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and np.isnan(val):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def _iter_valid_rows(df: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    for _, row in df.iterrows():
        asin = _clean_str(row.get("asin"))
        title = _clean_str(row.get("title"))
        img_url = _clean_str(row.get("imgUrl"))
        category = _clean_str(row.get("categoryName")) or "Unknown"
        product_url = _clean_str(row.get("productURL"))
        price = _clean_float(row.get("price"))
        stars = _clean_float(row.get("stars"))
        reviews = _clean_int(row.get("reviews"))
        bought = _clean_int(row.get("boughtInLastMonth"))
        is_best_seller = _clean_bool(row.get("isBestSeller"))

        if not asin or not title or price <= 0:
            continue
        # Keep img_url optional, but normalize empty to None downstream.
        yield {
            "asin": asin,
            "title": title,
            "category": category,
            "price": price,
            "stars": stars,
            "reviews": reviews,
            "bought_in_last_month": bought,
            "is_best_seller": is_best_seller,
            "product_url": product_url or None,
            "img_url": img_url or None,
        }


def _ensure_collection(q_client: QdrantClient, vector_size: int, reset: bool) -> None:
    if reset:
        q_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )
        return

    collections = {c.name for c in q_client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )


def _upsert_items(db: Session, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    stmt = pg_insert(models.Item).values(rows)
    update_cols = {
        "title": stmt.excluded.title,
        "category": stmt.excluded.category,
        "price": stmt.excluded.price,
        "stars": stmt.excluded.stars,
        "reviews": stmt.excluded.reviews,
        "bought_in_last_month": stmt.excluded.bought_in_last_month,
        "is_best_seller": stmt.excluded.is_best_seller,
        "product_url": stmt.excluded.product_url,
        "img_url": stmt.excluded.img_url,
    }
    stmt = stmt.on_conflict_do_update(index_elements=["asin"], set_=update_cols)
    db.execute(stmt)


def _upsert_qdrant(
    q_client: QdrantClient,
    rows: List[Dict[str, Any]],
    vectors: List[List[float]],
) -> None:
    points: List[qmodels.PointStruct] = []
    for row, vec in zip(rows, vectors):
        asin = row["asin"]
        payload = dict(row)
        points.append(
            qmodels.PointStruct(
                id=_deterministic_point_id(asin),
                vector=vec,
                payload=payload,
            )
        )
    q_client.upsert(collection_name=COLLECTION_NAME, points=points)


def ingest_data(
    csv_path: str,
    sample_size: int,
    chunksize: int,
    batch_size: int,
    reset: bool,
) -> None:
    print(f"Loading CSV: {csv_path}")
    models.Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

    print(f"Initializing SBERT: {settings.SBERT_MODEL}")
    model = SentenceTransformer(settings.SBERT_MODEL)
    vector_size = int(model.get_sentence_embedding_dimension())

    _ensure_collection(q_client, vector_size=vector_size, reset=reset)

    ingested = 0
    buffered_rows: List[Dict[str, Any]] = []

    reader = pd.read_csv(csv_path, chunksize=chunksize)
    for chunk in tqdm(reader, desc="Streaming CSV chunks"):
        for row in _iter_valid_rows(chunk):
            buffered_rows.append(row)
            if len(buffered_rows) >= batch_size:
                # 1) Postgres upsert
                _upsert_items(db, buffered_rows)
                db.commit()

                # 2) Qdrant upsert
                texts = [f"{r['title']}. Category: {r['category']}" for r in buffered_rows]
                vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
                _upsert_qdrant(q_client, buffered_rows, vectors=vectors)

                ingested += len(buffered_rows)
                buffered_rows = []

                if sample_size > 0 and ingested >= sample_size:
                    break
        if sample_size > 0 and ingested >= sample_size:
            break

    if buffered_rows and (sample_size <= 0 or ingested < sample_size):
        _upsert_items(db, buffered_rows)
        db.commit()
        texts = [f"{r['title']}. Category: {r['category']}" for r in buffered_rows]
        vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
        _upsert_qdrant(q_client, buffered_rows, vectors=vectors)
        ingested += len(buffered_rows)

    db.close()
    print(f"Done. Upserted {ingested} products into Postgres + Qdrant.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest product CSV into Postgres + Qdrant.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="/app/data/amz_uk_processed_data.csv",
        help="Path to amz_uk_processed_data.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=settings.INGEST_SAMPLE_SIZE,
        help="Number of products to ingest (0 = ingest all rows).",
    )
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=settings.INGEST_BATCH_SIZE)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="If set, recreates the Qdrant collection before ingesting.",
    )
    args = parser.parse_args()

    ingest_data(
        csv_path=args.csv_path,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
