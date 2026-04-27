import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add app to path to import models and session
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import models
from app.db.session import SessionLocal, engine
from app.core.config import settings

def ingest_data(csv_path: str, sample_size: int = 5000):
    print(f"📥 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.sample(min(sample_size, len(df)), random_state=42)
    
    # Clean up
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
    df['stars'] = pd.to_numeric(df['stars'], errors='coerce').fillna(0.0)
    df['categoryName'] = df['categoryName'].fillna("Unknown")
    
    db: Session = SessionLocal()
    q_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    # 1. Setup Qdrant Collection
    collection_name = "products"
    vector_size = 384 # SBERT miniLM
    
    q_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )
    
    print("🧠 Initializing SBERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🚀 Starting ingestion...")
    
    items_to_save = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing items"):
        # a. Create Postgres Record
        item = models.Item(
            asin=str(row['asin']),
            title=str(row['title']),
            category=str(row['categoryName']),
            price=float(row['price']),
            stars=float(row['stars']),
            img_url=str(row.get('imgUrl', ''))
        )
        items_to_save.append(item)
        
        # We'll batch save to Postgres and Qdrant for speed
        if len(items_to_save) >= 100:
            # Postgres
            db.add_all(items_to_save)
            db.commit()
            
            # Qdrant
            titles = [it.title for it in items_to_save]
            asins = [it.asin for it in items_to_save]
            embeddings = model.encode(titles).tolist()
            
            q_client.upsert(
                collection_name=collection_name,
                points=[
                    qmodels.PointStruct(
                        id=hash(asin) & 0xFFFFFFFFFFFFFFFF, # Simplistic ID mapping
                        vector=emb,
                        payload={
                            "asin": asin,
                            "title": title,
                            "category": cat,
                            "price": price
                        }
                    )
                    for asin, title, emb, cat, price in zip(
                        asins, 
                        titles, 
                        embeddings, 
                        [it.category for it in items_to_save],
                        [it.price for it in items_to_save]
                    )
                ]
            )
            items_to_save = []

    # Final batch
    if items_to_save:
        db.add_all(items_to_save)
        db.commit()
        
    print("✅ Items ingested into Postgres and Qdrant.")
    
    # 2. Simulate Users and Interactions
    print("👤 Creating synthetic users and interactions...")
    # Create 100 users
    users = []
    for i in range(100):
        user = models.User(external_id=f"USER_{i}")
        users.append(user)
    db.add_all(users)
    db.commit()
    
    # Get all items from DB to link them
    db_items = db.query(models.Item).all()
    item_ids = [it.id for it in db_items]
    user_ids = [u.id for u in users]
    
    interactions = []
    for _ in range(5000):
        uid = np.random.choice(user_ids)
        iid = np.random.choice(item_ids)
        itype = np.random.choice(['view', 'click', 'purchase'], p=[0.7, 0.2, 0.1])
        
        interaction = models.Interaction(
            user_id=uid,
            item_id=iid,
            interaction_type=itype,
            timestamp=pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
        )
        interactions.append(interaction)
        
        if len(interactions) >= 500:
            db.add_all(interactions)
            db.commit()
            interactions = []
            
    if interactions:
        db.add_all(interactions)
        db.commit()
        
    print("✅ Users and Interactions ingested.")
    db.close()

if __name__ == "__main__":
    csv_path = "amz_uk_processed_data.csv"
    ingest_data(csv_path)
