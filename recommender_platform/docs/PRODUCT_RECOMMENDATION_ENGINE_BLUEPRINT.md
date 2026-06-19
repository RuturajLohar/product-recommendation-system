# Product Recommendation Engine Blueprint

## 1. Final Project Direction

The main goal of this project is no longer to build an Amazon/Flipkart-style production ecommerce system.

The main goal is:

```text
Build the most accurate product recommendation engine possible for uploaded product datasets.
```

The project should focus on three perfect core areas:

1. Product recommendation accuracy.
2. Universal product CSV ingestion.
3. Proper product data storage in PostgreSQL and vector storage.

Everything else is secondary.

This project should be treated as a recommendation-engine-first experimental system, not a full ecommerce SaaS platform.

---

## 2. Updated Vision

### Main Vision

Build a **Universal Product Recommendation Engine**.

The system should accept multiple product CSV datasets, normalize them into one fixed internal product schema, store them properly, generate embeddings, and recommend the best matching products across the combined product catalog.

Example:

```text
Dataset 1: Amazon electronics
Dataset 2: Flipkart mobile accessories
Dataset 3: Meesho fashion products
Dataset 4: Custom product catalog

All datasets are ingested into one product universe.

User searches:
"gaming mouse"

System searches all product data together and returns:
1. Logitech G305
2. Razer DeathAdder
3. SteelSeries Rival
4. HyperX Pulsefire
5. Corsair Katar
```

### Important Clarification

The system should **not** accept completely unrelated datasets right now.

It should not focus on:

```text
movies
jobs
travel
books
courses
restaurants
```

Those can be future modules.

For now:

```text
100% focus on product recommendation.
```

---

## 3. Non-Goals

Do not spend time on:

- Amazon clone frontend.
- Flipkart clone frontend.
- Checkout system.
- Cart system beyond demo event tracking.
- Payment system.
- Inventory management.
- Multi-tenant SaaS.
- Collaborative filtering.
- User-user similarity.
- Matrix factorization.
- LightFM.
- NMF.
- Deep user personalization.
- Production auth.
- Kubernetes.
- CI/CD.
- Billing.
- Admin SaaS dashboard.

The only serious goal is:

```text
Given product data and a user product query, return the nearest and most accurate product recommendations.
```

---

## 4. Recommendation Strategy

### Final Strategy For This Phase

Use **content-based recommendation only**.

This means recommendations are based on product information:

- product title
- description
- category
- brand
- price
- rating
- reviews
- popularity
- best-seller flag
- tags/specifications if available

No collaborative filtering should be used.

### Why Content-Based Only

Content-based recommendation is the right focus because:

1. It works immediately after CSV ingestion.
2. It does not need user interaction history.
3. It works across multiple product datasets.
4. It works for cold-start products.
5. It is easier to explain in a college project.
6. Accuracy can be improved using product metadata.
7. It aligns with the actual project goal: product similarity.

### Collaborative Filtering Status

Existing collaborative filtering files should be treated as legacy experimental code.

Files such as:

```text
collaborative_filtering.py
hybrid_recommender.py
```

should not be used in the main product recommendation pipeline.

Implementation rule:

```text
Do not import or call collaborative filtering in the production product engine.
```

The code can remain in the repository as old research/prototype code, but the active system should ignore it.

Recommended future cleanup:

```text
Move collaborative filtering files into legacy/
or add comments saying "Not used in current content-based product engine."
```

---

## 5. Core User Flow

### Flow 1: Product Dataset Upload

```text
User uploads product CSV
        ->
System detects columns
        ->
System maps columns into fixed product schema
        ->
System validates required fields
        ->
System cleans and normalizes data
        ->
System stores products in PostgreSQL
        ->
System creates embeddings
        ->
System stores vectors in Qdrant
        ->
Products become searchable and recommendable
```

### Flow 2: Product Search And Recommendation

```text
User enters product query:
"gaming mouse"
        ->
System embeds query
        ->
Qdrant retrieves semantically similar products
        ->
PostgreSQL provides full product metadata
        ->
Ranking layer scores products
        ->
MMR removes duplicates / improves diversity
        ->
Top 10 recommendations returned
```

### Flow 3: Product-To-Product Recommendation

```text
User selects product:
"Logitech G305"
        ->
System uses selected product text + metadata
        ->
Finds similar product vectors
        ->
Ranks by content similarity + metadata similarity
        ->
Returns nearest alternatives
```

---

## 6. Universal Product CSV Acceptor

### Goal

The system should accept product CSVs from different sources.

Example sources:

- Amazon product dataset
- Flipkart product dataset
- ecommerce export
- Shopify product CSV
- custom college dataset
- Kaggle product dataset
- scraped product catalog

But all datasets must be product-related.

### Accepted Dataset Type

Allowed:

```text
Product datasets only
```

Not allowed for this module:

```text
movies
jobs
travel
restaurants
books
courses
```

### Minimum Required Columns

Every product CSV must provide at least:

```text
product_id
title/name
```

The system should not require price, category, rating, or images.

Those fields improve accuracy but should remain optional.

### Internal Fixed Product Schema

Every CSV row should be converted into this internal schema:

```json
{
  "dataset_id": 1,
  "source_product_id": "B09B96TG33",
  "title": "Echo Dot 5th generation smart speaker",
  "description": "Smart speaker with Alexa and Bluetooth",
  "category": "Speakers",
  "subcategory": "Smart Speakers",
  "brand": "Amazon",
  "price": 21.99,
  "currency": "GBP",
  "rating": 4.7,
  "reviews": 15308,
  "popularity": 600,
  "is_best_seller": false,
  "image_url": "https://example.com/image.jpg",
  "product_url": "https://example.com/product",
  "tags": "wifi bluetooth alexa smart speaker",
  "raw_metadata": {}
}
```

### Required Internal Fields

```text
dataset_id
source_product_id
title
```

### Optional Internal Fields

```text
description
category
subcategory
brand
price
currency
rating
reviews
popularity
is_best_seller
image_url
product_url
tags
raw_metadata
```

### Default Values

If optional fields are missing:

```text
description      -> ""
category         -> "Unknown"
subcategory      -> "Unknown"
brand            -> "Unknown"
price            -> 0.0
currency         -> ""
rating           -> 0.0
reviews          -> 0
popularity       -> 0
is_best_seller   -> false
image_url        -> null
product_url      -> null
tags             -> ""
raw_metadata     -> original CSV row as JSON
```

---

## 7. Column Mapping Algorithm

### Purpose

Different product datasets use different column names.

Examples:

```text
Amazon:
asin, title, categoryName, price, stars, reviews, imgUrl

Flipkart:
pid, product_name, product_category_tree, retail_price, rating, image

Shopify:
Handle, Title, Body, Vendor, Type, Tags, Variant Price, Image Src
```

The system must map all of them into the fixed internal schema.

### Auto-Mapping Dictionary

Use column-name matching with lowercase normalization.

```python
FIELD_CANDIDATES = {
    "source_product_id": [
        "asin",
        "id",
        "product_id",
        "pid",
        "sku",
        "item_id",
        "handle",
        "product_code"
    ],
    "title": [
        "title",
        "name",
        "product_name",
        "item_name",
        "product title",
        "product"
    ],
    "description": [
        "description",
        "desc",
        "details",
        "body",
        "about",
        "product_description"
    ],
    "category": [
        "category",
        "categoryName",
        "product_category",
        "product_category_tree",
        "type",
        "department"
    ],
    "subcategory": [
        "subcategory",
        "sub_category",
        "sub category"
    ],
    "brand": [
        "brand",
        "vendor",
        "manufacturer",
        "company"
    ],
    "price": [
        "price",
        "retail_price",
        "discounted_price",
        "sale_price",
        "cost",
        "amount",
        "variant price"
    ],
    "currency": [
        "currency",
        "currency_code"
    ],
    "rating": [
        "stars",
        "rating",
        "average_rating",
        "product_rating",
        "score"
    ],
    "reviews": [
        "reviews",
        "review_count",
        "ratings_count",
        "number_of_reviews"
    ],
    "popularity": [
        "boughtInLastMonth",
        "sales",
        "views",
        "orders",
        "popularity"
    ],
    "is_best_seller": [
        "isBestSeller",
        "best_seller",
        "bestseller",
        "is_best_seller"
    ],
    "image_url": [
        "imgUrl",
        "image",
        "image_url",
        "image src",
        "thumbnail",
        "photo"
    ],
    "product_url": [
        "productURL",
        "url",
        "link",
        "product_url",
        "product link"
    ],
    "tags": [
        "tags",
        "keywords",
        "features",
        "specifications"
    ]
}
```

### Mapping Confidence

The mapper should return:

```json
{
  "field": "title",
  "matched_column": "product_name",
  "confidence": 0.95
}
```

Confidence rules:

```text
exact lowercase match       -> 1.00
normalized match            -> 0.90
contains keyword            -> 0.70
fuzzy match                 -> 0.60
not found                   -> 0.00
```

### User Confirmation

The system should not blindly ingest.

Recommended flow:

```text
Upload CSV
        ->
Preview columns
        ->
Auto-suggest mapping
        ->
User confirms or corrects mapping
        ->
Ingest
```

This prevents incorrect mappings from ruining recommendation quality.

---

## 8. Data Cleaning Rules

### Product ID

Rules:

- Convert to string.
- Trim whitespace.
- Reject if empty.
- Preserve original ID exactly after trimming.

### Title

Rules:

- Convert to string.
- Trim whitespace.
- Collapse repeated spaces.
- Reject if empty.
- Reject if title length is less than 3 characters.

### Description

Rules:

- Convert to string.
- Empty if missing.
- Remove repeated whitespace.

### Category / Subcategory / Brand

Rules:

- Convert to string.
- Trim whitespace.
- Default to `"Unknown"`.
- Normalize obvious missing values:
  ```text
  nan, none, null, n/a, -
  ```

### Price

Rules:

- Remove currency symbols.
- Remove commas.
- Convert to float.
- If invalid, set `0.0`.
- If negative, set `0.0`.

Examples:

```text
"INR 1,299" -> 1299.0
"$49.99" -> 49.99
"1,499" -> 1499.0
```

### Rating

Rules:

- Convert to float.
- Clamp to range `0.0` to `5.0` if product rating is out of 5.
- If dataset has rating out of 10, normalize to 5 if detected.
- If invalid, set `0.0`.

### Reviews

Rules:

- Convert to integer.
- Remove commas.
- If invalid, set `0`.

### Popularity

Rules:

- Convert to integer or float.
- If invalid, set `0`.
- Can represent:
  ```text
  bought last month
  sales
  orders
  views
  popularity count
  ```

### Best Seller

Accepted true values:

```text
true
1
yes
y
t
best seller
bestseller
```

All other values default to false.

### Raw Metadata

Every original CSV row must be preserved in `raw_metadata`.

Purpose:

- No information is lost.
- Future ranking features can use more columns later.
- Debugging ingestion becomes easier.

---

## 9. PostgreSQL Storage Blueprint

### Main Tables

The database should store:

```text
datasets
items
interactions
```

For current goals, the most important tables are:

```text
datasets
items
```

Interactions are useful for demo behavior but not required for content-based recommendations.

### datasets Table

Purpose:

Stores metadata about each uploaded product dataset.

Recommended columns:

```text
id                      integer primary key
name                    string
original_filename       string
source_type             string
status                  string
total_rows              integer
valid_rows              integer
invalid_rows            integer
duplicate_rows          integer
mapping_json            json/text
created_at              datetime
updated_at              datetime
```

Status values:

```text
previewed
ingesting
completed
failed
```

### items Table

Purpose:

Stores normalized products from every product dataset.

Recommended columns:

```text
id                      integer primary key
dataset_id              foreign key -> datasets.id
source_product_id       string
title                   string
description             text
category                string
subcategory             string
brand                   string
price                   float
currency                string
stars                   float
reviews                 integer
bought_in_last_month    integer
is_best_seller          boolean
product_url             string
img_url                 string
tags                    text
raw_metadata            text/json
embedding_text          text
created_at              datetime
updated_at              datetime
```

### Unique Constraint

Use:

```text
dataset_id + source_product_id
```

This allows different datasets to contain the same product ID without conflict.

### Search Indexes

Recommended indexes:

```text
items.dataset_id
items.source_product_id
items.title
items.category
items.brand
items.price
items.stars
items.reviews
```

### Backward Compatibility

Current code uses:

```text
asin
img_url
stars
bought_in_last_month
```

For a fast transition:

- keep `asin` as the API field temporarily
- internally treat `asin` as `source_product_id`
- later rename API fields to `source_product_id`

Recommended short-term mapping:

```text
asin                  -> source_product_id
stars                 -> rating
bought_in_last_month  -> popularity
img_url               -> image_url
```

---

## 10. Qdrant Vector Storage Blueprint

### Collection

Use one collection:

```text
products
```

Because all datasets are product datasets and should be searched together.

### Vector Payload

Each Qdrant point should store:

```json
{
  "dataset_id": 1,
  "source_product_id": "B09B96TG33",
  "title": "Echo Dot smart speaker",
  "description": "Smart speaker with Alexa",
  "category": "Speakers",
  "subcategory": "Smart Speakers",
  "brand": "Amazon",
  "price": 21.99,
  "rating": 4.7,
  "reviews": 15308,
  "popularity": 600,
  "is_best_seller": false,
  "image_url": "https://example.com/image.jpg",
  "product_url": "https://example.com/product"
}
```

### Point ID Strategy

Use deterministic hash:

```text
hash(dataset_id + ":" + source_product_id)
```

This prevents collisions across datasets.

Example:

```python
point_id = md5(f"{dataset_id}:{source_product_id}").hexdigest()
```

### Embedding Text

Do not embed only the title.

Use richer text:

```text
Title: {title}
Brand: {brand}
Category: {category}
Subcategory: {subcategory}
Description: {description}
Tags: {tags}
```

This improves semantic accuracy.

### Why Combined Qdrant Collection

Because the user wants:

```text
dataset 1 + dataset 2 + dataset 3 + ...
all searched together
best recommendations from all data
```

So all product vectors should live in the same product collection.

---

## 11. Content-Based Recommendation Engine

### Engine Goal

Given a product query or selected product, return the most relevant products across all ingested product datasets.

### Input Types

Support two main input types:

```text
text query
selected product ID
```

Examples:

```text
gaming mouse
iPhone 15 case
wireless earbuds
office chair
running shoes
```

### Query Recommendation Flow

```text
User query
        ->
Generate query embedding
        ->
Qdrant vector search across all products
        ->
Retrieve top 100 candidates
        ->
Fetch/merge metadata
        ->
Feature scoring
        ->
Final ranking
        ->
MMR diversity
        ->
Return top K
```

### Product-To-Product Flow

```text
Selected product
        ->
Build product embedding text
        ->
Find semantically similar products
        ->
Exclude selected product itself
        ->
Rank candidates
        ->
Return top K
```

---

## 12. Ranking Signals

The ranking layer should combine semantic similarity with product metadata.

### Signal 1: Semantic Similarity

Source:

```text
Qdrant cosine similarity
```

Most important signal.

Weight:

```text
0.45
```

### Signal 2: Category Similarity

Compare:

```text
query/seed category vs candidate category
```

For text query, category can be inferred from top candidates or ignored.

Weight:

```text
0.15
```

### Signal 3: Brand Similarity

Useful for products like:

```text
iPhone case
Nike shoes
Logitech mouse
Samsung charger
```

Weight:

```text
0.05
```

### Signal 4: Price Similarity

If seed product has price:

```text
candidate price close to seed price = higher score
```

Formula:

```text
price_similarity = exp(-abs(log(candidate_price) - log(seed_price)))
```

Weight:

```text
0.10
```

### Signal 5: Rating Score

Normalize:

```text
rating_score = rating / 5
```

Weight:

```text
0.10
```

### Signal 6: Review Confidence

Products with many reviews are more trustworthy.

Formula:

```text
review_confidence = min(log1p(reviews) / log1p(100000), 1.0)
```

Weight:

```text
0.05
```

### Signal 7: Popularity Score

Based on:

```text
sales
views
orders
bought in last month
```

Normalize with log scaling.

Weight:

```text
0.05
```

### Signal 8: Best Seller Boost

If `is_best_seller = true`:

```text
small boost
```

Weight:

```text
0.05
```

### Final Formula

Recommended v1 formula:

```text
final_score =
    0.45 * semantic_similarity
  + 0.15 * category_similarity
  + 0.05 * brand_similarity
  + 0.10 * price_similarity
  + 0.10 * rating_score
  + 0.05 * review_confidence
  + 0.05 * popularity_score
  + 0.05 * best_seller_boost
```

If a signal is missing:

```text
redistribute its weight to semantic_similarity
```

This keeps the engine strong even with minimal CSV data.

---

## 13. Diversity Layer

Use MMR after ranking.

Purpose:

- avoid recommending the same duplicate product repeatedly
- avoid near-identical title variants
- improve result variety

MMR should balance:

```text
relevance
diversity
```

Recommended value:

```text
lambda = 0.75
```

Meaning:

- 75% relevance
- 25% diversity

Since the goal is accuracy, relevance should remain higher than diversity.

---

## 14. Accuracy Improvements

### 1. Better Embedding Text

Current systems often embed only title.

For higher accuracy, embed:

```text
title + brand + category + description + tags
```

This helps when title is short.

### 2. Product Type Detection

Extract product type from:

```text
category
title keywords
description
```

Examples:

```text
gaming mouse
phone case
wireless earbuds
running shoes
office chair
```

Use product type to avoid bad matches.

Example:

```text
Query: iPhone 15 case
Bad match: iPhone 15 phone
Good match: iPhone 15 silicone case
```

### 3. Query Expansion

For product search query:

```text
"gaming mouse"
```

Create expanded text:

```text
gaming mouse, wireless mouse, RGB mouse, esports mouse, high DPI mouse
```

This can be done with a simple dictionary first.

### 4. Brand Boost

If query contains brand:

```text
Logitech gaming mouse
```

Boost products with:

```text
brand = Logitech
```

### 5. Category Hard Preference

If seed product has clear category:

```text
Mouse
```

Products from same category should rank higher.

Do not hard-filter because category data may be noisy.

### 6. Duplicate Suppression

Suppress near-duplicates using:

```text
same title normalized
same image URL
same source URL
very high text similarity
```

### 7. Price Range Awareness

For alternatives, users usually expect similar price range.

Example:

```text
Seed: 999 rupee mouse
Bad: 15000 rupee mouse
Good: 800-1500 rupee mouse
```

### 8. Strong Fallback For Minimal Data

If dataset only has:

```text
id,title
```

Use:

```text
semantic similarity only
```

This ensures every valid product CSV still works.

---

## 15. API Blueprint

### Dataset Preview

```http
POST /api/v1/product-datasets/preview
```

Input:

```text
CSV file
```

Output:

```json
{
  "filename": "products.csv",
  "columns": ["pid", "product_name", "category", "price"],
  "sample_rows": [],
  "suggested_mapping": {
    "source_product_id": "pid",
    "title": "product_name",
    "category": "category",
    "price": "price"
  },
  "required_fields_present": true
}
```

### Dataset Ingest

```http
POST /api/v1/product-datasets/ingest
```

Input:

```text
CSV file
dataset_name
confirmed_mapping
```

Output:

```json
{
  "dataset_id": 3,
  "dataset_name": "Flipkart Electronics",
  "status": "completed",
  "total_rows": 100000,
  "valid_rows": 98500,
  "invalid_rows": 1500,
  "message": "Products ingested successfully."
}
```

### Dataset List

```http
GET /api/v1/product-datasets
```

Output:

```json
[
  {
    "dataset_id": 1,
    "name": "Amazon Products",
    "valid_rows": 1400000,
    "status": "completed"
  },
  {
    "dataset_id": 2,
    "name": "Flipkart Products",
    "valid_rows": 500000,
    "status": "completed"
  }
]
```

### Query-Based Recommendation

```http
GET /api/v1/recommend/products?q=gaming%20mouse&limit=10
```

Output:

```json
{
  "query": "gaming mouse",
  "recommendations": [
    {
      "source_product_id": "G305",
      "title": "Logitech G305 Wireless Gaming Mouse",
      "category": "Gaming Mouse",
      "brand": "Logitech",
      "price": 2499,
      "rating": 4.6,
      "score": 0.91,
      "reason": "High semantic match, same product type, strong rating."
    }
  ]
}
```

### Product-To-Product Recommendation

```http
GET /api/v1/recommend/products/{source_product_id}/similar?limit=10
```

Output:

```json
{
  "seed_product_id": "B09B96TG33",
  "recommendations": []
}
```

---

## 16. Frontend Blueprint

Frontend should support only the product module for now.

### Main Screens

```text
Product Recommendation Dashboard
Product Dataset Upload
Product Search
Product Results
Product Details
```

### Product Dataset Upload UI

Flow:

```text
Upload CSV
        ->
Show column preview
        ->
Show auto mapping
        ->
User confirms mapping
        ->
Ingest
        ->
Show ingestion summary
```

### Product Search UI

Input:

```text
Search Product: [ gaming mouse ]
```

Output:

```text
Top 10 Similar Products
```

Each card should show:

```text
title
image
category
brand
price
rating
score
reason
dataset source
```

### Explainability UI

Show:

```text
Recommended because:
- Semantic match: 92%
- Same category: Gaming Mouse
- Similar price range
- Strong rating: 4.6
- Popular product
```

This makes the engine look more accurate and understandable.

---

## 17. Implementation Phases

### Phase 1: Product-Only Blueprint Cleanup

Goal:

Make the project direction clear.

Tasks:

1. Mark collaborative filtering as legacy.
2. Update docs to say product-only for now.
3. Update README to remove "any dataset" ambiguity.
4. Keep UI focused on product recommendation.

### Phase 2: Universal Product Schema

Goal:

Create fixed internal product schema.

Tasks:

1. Add `datasets` table.
2. Extend `items` table for universal product fields.
3. Add `raw_metadata`.
4. Add `embedding_text`.
5. Add dataset-aware unique key.

### Phase 3: Universal Product CSV Ingestion

Goal:

Accept product CSVs from different sources.

Tasks:

1. Build product column mapper.
2. Build preview endpoint.
3. Build confirmed ingest endpoint.
4. Normalize rows.
5. Store in PostgreSQL.
6. Store vectors in Qdrant.
7. Track valid/invalid/duplicate rows.

### Phase 4: Content-Based Recommendation Upgrade

Goal:

Improve accuracy.

Tasks:

1. Use richer embedding text.
2. Add query-based product recommendation.
3. Add product-to-product recommendation.
4. Add metadata scoring.
5. Add duplicate suppression.
6. Add MMR with relevance-heavy setting.

### Phase 5: Frontend Product Engine UI

Goal:

Make system usable and demo-ready.

Tasks:

1. Add product CSV upload screen.
2. Add mapping confirmation UI.
3. Add ingestion summary.
4. Add query search.
5. Add explanation cards.
6. Add dataset source badge.

### Phase 6: Accuracy Evaluation

Goal:

Show recommendation quality.

Tasks:

1. Create small labeled test queries.
2. Measure whether top results match expected product type.
3. Add manual evaluation table.
4. Compare:
   ```text
   title-only embedding
   vs
   title + metadata embedding
   vs
   metadata reranking
   ```

---

## 18. Testing Strategy

### CSV Mapping Tests

Test with:

```text
Amazon-style product CSV
Flipkart-style product CSV
Shopify-style product CSV
Minimal id,title CSV
Invalid CSV missing title
Invalid CSV missing product ID
```

Expected:

- valid product CSVs accepted
- invalid CSVs rejected
- optional fields default correctly
- raw metadata preserved

### Ingestion Tests

Verify:

```text
dataset row created
items inserted
duplicates handled
invalid rows counted
Qdrant points created
embedding_text generated
```

### Recommendation Tests

Queries:

```text
gaming mouse
wireless earbuds
iPhone 15 case
office chair
running shoes
bluetooth speaker
```

Expected:

- top results match product intent
- seed product is excluded
- unrelated categories are lower ranked
- results include products from all datasets
- duplicate products are reduced

### API Tests

Check:

```text
/product-datasets/preview
/product-datasets/ingest
/product-datasets
/recommend/products?q=
/recommend/products/{id}/similar
```

### Frontend Tests

Manual flow:

```text
Upload product CSV
Confirm mapping
Ingest
Search product
View recommendations
Open recommendation details
Read explanation
```

---

## 19. Success Criteria

This module is successful when:

1. Multiple product CSV datasets can be ingested.
2. All datasets are stored together in PostgreSQL.
3. All product vectors are searchable together.
4. A query searches across all product datasets.
5. Recommendations are content-based only.
6. Results are accurate for product intent.
7. Results include understandable explanations.
8. Missing optional columns do not break ingestion.
9. DB stores normalized fields and original raw metadata.
10. The system can demo product recommendations without user history.

---

## 20. Final System Identity

The project should now be described as:

```text
Universal Product Recommendation Engine
```

One-line description:

```text
A content-based AI product recommendation engine that accepts multiple product CSV datasets, normalizes them into a unified product schema, stores them in PostgreSQL, indexes them with vector embeddings, and recommends the most semantically and metadata-similar products across the combined catalog.
```

Resume-worthy title:

```text
Universal Content-Based Product Recommendation Engine
```

Resume description:

```text
Designed and developed a universal content-based product recommendation engine that ingests multiple product CSV datasets, maps heterogeneous schemas into a fixed product format, stores normalized product data in PostgreSQL, indexes rich product embeddings in Qdrant, and ranks recommendations using semantic similarity, category similarity, price affinity, ratings, popularity, and explainable metadata-based scoring.
```
