# Amazon Dataset and Database Integrity Audit

## Audit Scope

This audit compares:

- Source file: `data/amz_uk_processed_data.csv`
- PostgreSQL table: `items`
- Qdrant collection: `products`

The purpose is to determine whether every valid Amazon product was captured and
whether stored fields accurately match the CSV normalization rules.

## Executive Result

The existing database is internally clean and matches Qdrant, but it is not a
complete import of the Amazon dataset.

| Metric | Result |
|---|---:|
| CSV rows | 2,222,742 |
| Valid CSV products | 2,222,724 |
| Invalid-price rows rejected | 18 |
| Duplicate valid ASINs in CSV | 0 |
| PostgreSQL products | 5,120 |
| Qdrant vectors | 5,120 |
| Missing valid products | 2,217,604 |
| Dataset coverage | 0.230348% |

**Conclusion:** 99.769652% of valid Amazon products have not yet been loaded.

## Source Dataset Structure

The Amazon CSV contains these columns:

1. `asin`
2. `title`
3. `imgUrl`
4. `productURL`
5. `stars`
6. `reviews`
7. `price`
8. `isBestSeller`
9. `boughtInLastMonth`
10. `categoryName`

Current validity rules require:

- non-empty `asin`
- non-empty `title`
- `price > 0`

Exactly 18 source rows fail these rules because their price is missing, zero,
negative, or otherwise invalid. No source rows are missing ASIN or title, and
no duplicate valid ASIN was found.

## PostgreSQL Integrity

The 5,120 stored records passed these checks:

- duplicate ASINs: 0
- missing ASINs: 0
- missing titles: 0
- price less than or equal to zero: 0
- star ratings outside 0 to 5: 0
- negative review counts: 0
- missing image URLs: 0
- missing product URLs: 0

Observed numeric ranges:

| Field | Minimum | Maximum | Average |
|---|---:|---:|---:|
| Price | 1.56 | 2,986.22 | 87.8358 |
| Stars | 0.0 | 5.0 | 2.0996 |
| Reviews | 0 | 151,739 | n/a |

## PostgreSQL and Qdrant Alignment

PostgreSQL contains 5,120 products and Qdrant contains exactly 5,120 vectors.
Sampled Qdrant payload ASINs all exist in PostgreSQL.

This means the relational and vector stores are synchronized for the current
subset.

## Evidence of Partial Ingestion

The database category distribution is:

| Category | Database records |
|---|---:|
| CD, Disc & Tape Players | 2,632 |
| Hi-Fi Speakers | 2,488 |

The first 5,120 valid rows of the CSV have exactly the same category counts.
The first and last stored ASINs also align with that source segment.

Therefore, the database contains the first 5,120 valid CSV rows. It is not a
random or representative sample of the full catalog.

Large categories later in the file are absent. For example, the full CSV
contains 826,072 `Sports & Outdoors` rows.

## Field Accuracy Check

A row-level comparison was performed for 25 stored products against their CSV
source rows after applying the current normalization rules.

Compared fields:

- ASIN
- title
- category
- price
- stars
- reviews
- bought in last month
- best-seller status
- product URL
- image URL

Result: **0 field mismatches across 25 checked records.**

The stored subset is therefore field-accurate based on the inspected rows.

## Ingestion Corrections Implemented

The ingestion command now:

1. validates all expected Amazon columns before changing either store;
2. treats `--reset` as a clean reset of PostgreSQL and Qdrant;
3. clears dependent interactions before clearing products;
4. respects the exact requested sample size;
5. counts rejected rows by reason;
6. checks duplicate ASINs and invalid database values after import;
7. requires exact PostgreSQL/Qdrant count equality;
8. requires the clean-import database count to match accepted rows;
9. exits with an error instead of reporting false success when verification
   fails.
10. stores Qdrant vectors, payloads, and the HNSW index on disk so the complete
    dataset does not require all vectors to fit inside Docker's memory limit.

## Recommended Import Sequence

The full dataset requires embedding 2,222,724 products. Perform staged imports
first to measure runtime, memory, database size, and Qdrant size on the current
machine.

### 100,000-product validation run

```bash
cd recommender_platform
SAMPLE_SIZE=100000 make ingest
```

### 500,000-product scale run

```bash
cd recommender_platform
SAMPLE_SIZE=500000 make ingest
```

### Complete clean import

```bash
cd recommender_platform
make ingest-all
```

`make ingest-all` uses `--sample-size 0 --reset`, meaning all valid source rows
are processed after both stores are cleared.

## Full-Import Acceptance Criteria

The complete import is successful only when the final report shows:

```text
postgres_items: 2,222,724
postgres_distinct_asins: 2,222,724
qdrant_points: 2,222,724
missing_asins: 0
missing_titles: 0
invalid_prices: 0
invalid_stars: 0
negative_reviews: 0
```

The source report should also show:

```text
source_rows_read: 2,222,742
rejected_rows: 18
invalid_price: 18
```

Anything else means the import is incomplete or inconsistent and must not be
treated as a successful catalog initialization.
