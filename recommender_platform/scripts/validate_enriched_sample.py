import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.generate_enriched_sample import (
    DOMAIN_PROFILES,
    FORBIDDEN_VALUES,
    OUTPUT_COLUMNS,
    TYPE_RULES,
    classify_domain,
)


def _is_forbidden(value: object) -> bool:
    return str(value).strip().lower() in FORBIDDEN_VALUES


def validate(csv_path: Path, expected_rows: int, expected_categories: int) -> None:
    ids = set()
    categories = Counter()
    domains = Counter()
    brands = Counter()
    with csv_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise AssertionError(f"Unexpected columns: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            bad_fields = [key for key, value in row.items() if _is_forbidden(value)]
            if bad_fields:
                raise AssertionError(f"Line {line_number} has forbidden values in {bad_fields}")
            product_id = row["product_id"]
            if product_id in ids:
                raise AssertionError(f"Duplicate product_id: {product_id}")
            ids.add(product_id)

            price = float(row["price"])
            rating = float(row["rating"])
            reviews = int(row["review_count"])
            popularity = float(row["popularity"])
            if price <= 0 or not 1 <= rating <= 5 or reviews <= 0 or not 0 <= popularity <= 100:
                raise AssertionError(f"Invalid numeric values for {product_id}")
            if row["availability"] not in {"in_stock", "limited_stock", "out_of_stock"}:
                raise AssertionError(f"Invalid availability for {product_id}")
            if row["best_seller"].lower() not in {"true", "false"}:
                raise AssertionError(f"Invalid best_seller for {product_id}")

            features = json.loads(row["features"])
            specs = json.loads(row["specifications"])
            if not isinstance(features, list) or len(features) < 4:
                raise AssertionError(f"Insufficient features for {product_id}")
            if not isinstance(specs, dict) or len(specs) < 3:
                raise AssertionError(f"Insufficient specifications for {product_id}")
            if any(_is_forbidden(value) for value in [*features, *specs.keys(), *specs.values()]):
                raise AssertionError(f"Forbidden nested value for {product_id}")

            domain = classify_domain(row["category"], row["title"])
            valid_subcategories = set(DOMAIN_PROFILES[domain]["subcategories"])
            valid_subcategories.update(value for _, value in TYPE_RULES.get(domain, []))
            if row["brand"] not in DOMAIN_PROFILES[domain]["brands"]:
                raise AssertionError(f"Brand/domain mismatch for {product_id}")
            if row["subcategory"] not in valid_subcategories:
                raise AssertionError(f"Subcategory/domain mismatch for {product_id}")
            if row["brand"].lower() not in row["description"].lower():
                raise AssertionError(f"Description omits brand for {product_id}")
            if row["brand"].lower() not in row["keywords/tags"].lower():
                raise AssertionError(f"Keywords omit brand for {product_id}")

            categories[row["category"]] += 1
            domains[domain] += 1
            brands[row["brand"]] += 1

    if len(ids) != expected_rows:
        raise AssertionError(f"Expected {expected_rows:,} rows, found {len(ids):,}")
    if len(categories) != expected_categories:
        raise AssertionError(f"Expected {expected_categories} categories, found {len(categories)}")
    print("Validation passed")
    print(f"  rows: {len(ids):,}")
    print(f"  columns: {len(OUTPUT_COLUMNS)}")
    print(f"  categories: {len(categories)}")
    print(f"  domains: {len(domains)}")
    print(f"  synthetic_brands: {len(brands)}")
    print(f"  forbidden_or_blank_values: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the enriched product sample.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--expected-rows", type=int, default=50_000)
    parser.add_argument("--expected-categories", type=int, default=296)
    args = parser.parse_args()
    validate(args.csv_path, args.expected_rows, args.expected_categories)


if __name__ == "__main__":
    main()
