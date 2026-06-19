import argparse
import hashlib
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tqdm import tqdm

# Add app to path to import models and session
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db import models
from app.db.session import SessionLocal, engine


COLLECTION_NAME = "products"
EXPECTED_COLUMNS = {
    "asin",
    "title",
    "imgUrl",
    "productURL",
    "stars",
    "reviews",
    "price",
    "isBestSeller",
    "boughtInLastMonth",
    "categoryName",
}


@dataclass
class IngestionStats:
    source_rows_read: int = 0
    accepted_rows: int = 0
    missing_asin: int = 0
    missing_title: int = 0
    invalid_price: int = 0

    @property
    def rejected_rows(self) -> int:
        return self.missing_asin + self.missing_title + self.invalid_price


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


def _validate_columns(columns: Iterable[str]) -> None:
    missing = sorted(EXPECTED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(
            "CSV schema validation failed. Missing required Amazon columns: "
            + ", ".join(missing)
        )


def _iter_valid_rows(
    df: pd.DataFrame,
    stats: IngestionStats,
) -> Iterable[Dict[str, Any]]:
    for _, row in df.iterrows():
        stats.source_rows_read += 1
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

        if not asin:
            stats.missing_asin += 1
            continue
        if not title:
            stats.missing_title += 1
            continue
        if price <= 0:
            stats.invalid_price += 1
            continue

        stats.accepted_rows += 1
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


def _ensure_collection(
    q_client: QdrantClient,
    vector_size: int,
    reset_qdrant: bool,
) -> None:
    vectors_config = qmodels.VectorParams(
        size=vector_size,
        distance=qmodels.Distance.COSINE,
        on_disk=True,
    )
    hnsw_config = qmodels.HnswConfigDiff(on_disk=True)
    optimizers_config = qmodels.OptimizersConfigDiff(memmap_threshold=20_000)

    if reset_qdrant:
        q_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            on_disk_payload=True,
        )
        return

    collections = {c.name for c in q_client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            on_disk_payload=True,
        )


def _reset_postgres_catalog(db: Session) -> None:
    # Interactions reference items, so both tables must be cleared together.
    db.execute(text("TRUNCATE TABLE interactions, items RESTART IDENTITY CASCADE"))
    db.commit()


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


def _verify_catalog(
    db: Session,
    q_client: QdrantClient,
    expected_count: Optional[int],
) -> Dict[str, int]:
    db_count = int(db.query(func.count(models.Item.id)).scalar() or 0)
    distinct_asins = int(
        db.query(func.count(func.distinct(models.Item.asin))).scalar() or 0
    )
    missing_asins = int(
        db.query(func.count(models.Item.id))
        .filter((models.Item.asin.is_(None)) | (models.Item.asin == ""))
        .scalar()
        or 0
    )
    missing_titles = int(
        db.query(func.count(models.Item.id))
        .filter((models.Item.title.is_(None)) | (models.Item.title == ""))
        .scalar()
        or 0
    )
    invalid_prices = int(
        db.query(func.count(models.Item.id))
        .filter((models.Item.price.is_(None)) | (models.Item.price <= 0))
        .scalar()
        or 0
    )
    invalid_stars = int(
        db.query(func.count(models.Item.id))
        .filter(
            (models.Item.stars.isnot(None))
            & ((models.Item.stars < 0) | (models.Item.stars > 5))
        )
        .scalar()
        or 0
    )
    negative_reviews = int(
        db.query(func.count(models.Item.id))
        .filter(models.Item.reviews < 0)
        .scalar()
        or 0
    )
    qdrant_count = int(
        q_client.count(collection_name=COLLECTION_NAME, exact=True).count
    )

    report = {
        "postgres_items": db_count,
        "postgres_distinct_asins": distinct_asins,
        "qdrant_points": qdrant_count,
        "missing_asins": missing_asins,
        "missing_titles": missing_titles,
        "invalid_prices": invalid_prices,
        "invalid_stars": invalid_stars,
        "negative_reviews": negative_reviews,
    }

    failures = []
    if db_count != distinct_asins:
        failures.append("PostgreSQL contains duplicate ASINs")
    if qdrant_count != db_count:
        failures.append(
            f"PostgreSQL/Qdrant count mismatch ({db_count} vs {qdrant_count})"
        )
    if expected_count is not None and db_count != expected_count:
        failures.append(
            f"expected {expected_count} products after a clean import, found {db_count}"
        )
    for field in (
        "missing_asins",
        "missing_titles",
        "invalid_prices",
        "invalid_stars",
        "negative_reviews",
    ):
        if report[field]:
            failures.append(f"{field}={report[field]}")

    print("\nPost-ingestion integrity report")
    for key, value in report.items():
        print(f"  {key}: {value:,}")

    if failures:
        raise RuntimeError("Catalog integrity verification failed: " + "; ".join(failures))

    return report


def _allocate_stratified_quotas(
    category_counts: Dict[str, int],
    target_size: int,
    minimum_per_category: int = 100,
) -> Dict[str, int]:
    """Allocate an exact, diversity-friendly sample across all categories."""
    target_size = min(target_size, sum(category_counts.values()))
    quotas = {
        category: min(count, minimum_per_category)
        for category, count in category_counts.items()
    }

    # This only matters when the requested sample is smaller than the number of
    # categories multiplied by the minimum allocation.
    while sum(quotas.values()) > target_size:
        category = max(quotas, key=lambda key: (quotas[key], category_counts[key]))
        quotas[category] -= 1

    remaining = target_size - sum(quotas.values())
    while remaining:
        active = [
            category
            for category, count in category_counts.items()
            if quotas[category] < count
        ]
        if not active:
            break

        # Square-root weighting preserves real catalog scale without allowing a
        # few giant categories to consume nearly the entire sample.
        weight_total = sum(math.sqrt(category_counts[category]) for category in active)
        allocations = []
        allocated = 0
        for category in active:
            capacity = category_counts[category] - quotas[category]
            exact = remaining * math.sqrt(category_counts[category]) / weight_total
            addition = min(capacity, int(exact))
            allocations.append((exact - int(exact), category, capacity, addition))
            allocated += addition

        if allocated:
            for _, category, _, addition in allocations:
                quotas[category] += addition
            remaining -= allocated
            continue

        # Finish small remainders by largest fractional entitlement.
        for _, category, capacity, _ in sorted(allocations, reverse=True):
            if remaining == 0:
                break
            if capacity > 0:
                quotas[category] += 1
                remaining -= 1

    return quotas


def _build_stratified_sample(
    csv_path: str,
    sample_size: int,
    chunksize: int,
    random_seed: int,
) -> tuple[List[Dict[str, Any]], IngestionStats, Dict[str, int]]:
    print("Pass 1/2: profiling valid products by category...")
    stats = IngestionStats()
    category_counts: Counter[str] = Counter()
    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize),
        desc="Profiling CSV chunks",
    ):
        for row in _iter_valid_rows(chunk, stats):
            category_counts[row["category"]] += 1

    target_size = min(sample_size, stats.accepted_rows)
    quotas = _allocate_stratified_quotas(dict(category_counts), target_size)
    print(
        f"Sampling {target_size:,} products across {len(quotas):,} categories "
        f"(seed={random_seed})."
    )

    print("Pass 2/2: selecting a reproducible random sample per category...")
    rng = random.Random(random_seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {
        category: [] for category in quotas
    }
    seen: Counter[str] = Counter()
    second_pass_stats = IngestionStats()
    for chunk in tqdm(
        pd.read_csv(csv_path, chunksize=chunksize),
        desc="Sampling CSV chunks",
    ):
        for row in _iter_valid_rows(chunk, second_pass_stats):
            category = row["category"]
            quota = quotas.get(category, 0)
            if quota == 0:
                continue
            seen[category] += 1
            reservoir = reservoirs[category]
            if len(reservoir) < quota:
                reservoir.append(row)
                continue
            replacement_index = rng.randrange(seen[category])
            if replacement_index < quota:
                reservoir[replacement_index] = row

    selected = [row for rows in reservoirs.values() for row in rows]
    rng.shuffle(selected)
    if len(selected) != target_size:
        raise RuntimeError(
            f"Stratified sampler selected {len(selected):,} rows; "
            f"expected {target_size:,}."
        )
    return selected, stats, quotas


def ingest_data(
    csv_path: str,
    sample_size: int,
    chunksize: int,
    batch_size: int,
    reset_all: bool,
    sampling: str,
    random_seed: int,
) -> None:
    if sample_size < 0:
        raise ValueError("--sample-size must be 0 or a positive integer")
    if chunksize <= 0 or batch_size <= 0:
        raise ValueError("--chunksize and --batch-size must be positive integers")

    print(f"Loading CSV: {csv_path}")
    csv_columns = pd.read_csv(csv_path, nrows=0).columns
    _validate_columns(csv_columns)
    print("CSV schema validation passed.")

    models.Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    stats = IngestionStats()

    try:
        sampled_rows: Optional[List[Dict[str, Any]]] = None
        sampled_quotas: Dict[str, int] = {}
        if sampling == "stratified":
            if sample_size == 0:
                raise ValueError("--sampling stratified requires --sample-size greater than 0")
            sampled_rows, stats, sampled_quotas = _build_stratified_sample(
                csv_path=csv_path,
                sample_size=sample_size,
                chunksize=chunksize,
                random_seed=random_seed,
            )

        print(f"Initializing SBERT: {settings.SBERT_MODEL}")
        model = SentenceTransformer(settings.SBERT_MODEL)
        vector_size = int(model.get_embedding_dimension())

        _ensure_collection(
            q_client,
            vector_size=vector_size,
            reset_qdrant=reset_all,
        )
        if reset_all:
            print("Resetting PostgreSQL items/interactions and Qdrant products.")
            _reset_postgres_catalog(db)

        ingested = 0
        buffered_rows: List[Dict[str, Any]] = []
        reached_limit = False

        def flush_rows(rows: List[Dict[str, Any]]) -> int:
            texts = [f"{r['title']}. Category: {r['category']}" for r in rows]
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
            ).tolist()
            _upsert_qdrant(q_client, rows, vectors=vectors)
            _upsert_items(db, rows)
            db.commit()
            return len(rows)

        if sampled_rows is not None:
            for start in tqdm(
                range(0, len(sampled_rows), batch_size),
                desc="Embedding and indexing sample",
            ):
                ingested += flush_rows(sampled_rows[start : start + batch_size])
        else:
            reader = pd.read_csv(csv_path, chunksize=chunksize)
            for chunk in tqdm(reader, desc="Streaming CSV chunks"):
                for row in _iter_valid_rows(chunk, stats):
                    buffered_rows.append(row)
                    if sample_size > 0 and ingested + len(buffered_rows) >= sample_size:
                        reached_limit = True
                        break

                    if len(buffered_rows) >= batch_size:
                        ingested += flush_rows(buffered_rows)
                        buffered_rows = []

                if reached_limit:
                    break

        if buffered_rows:
            ingested += flush_rows(buffered_rows)

        expected_count = ingested if reset_all else None
        _verify_catalog(db, q_client, expected_count=expected_count)

        print("\nCSV ingestion report")
        print(f"  source_rows_read: {stats.source_rows_read:,}")
        print(f"  accepted_rows_seen: {stats.accepted_rows:,}")
        print(f"  rejected_rows: {stats.rejected_rows:,}")
        print(f"  missing_asin: {stats.missing_asin:,}")
        print(f"  missing_title: {stats.missing_title:,}")
        print(f"  invalid_price: {stats.invalid_price:,}")
        if sampled_quotas:
            print(f"  sampled_categories: {len(sampled_quotas):,}")
            print(f"  smallest_category_sample: {min(sampled_quotas.values()):,}")
            print(f"  largest_category_sample: {max(sampled_quotas.values()):,}")
        print(f"Done. Upserted exactly {ingested:,} products into PostgreSQL + Qdrant.")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        q_client.close()


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
        help=(
            "Clean import: clears PostgreSQL items/interactions and recreates "
            "the Qdrant products collection before ingesting."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=("first", "stratified"),
        default="first",
        help="Select the first valid rows or a balanced random category sample.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed used by reproducible stratified sampling.",
    )
    args = parser.parse_args()

    ingest_data(
        csv_path=args.csv_path,
        sample_size=args.sample_size,
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        reset_all=args.reset,
        sampling=args.sampling,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
