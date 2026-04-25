import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import sys

# Add app to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.db.models import Base, Item, User, Interaction

# Configuration
DB_URL = "postgresql://admin:secretpassword@localhost:5432/recommender_db"
QDRANT_HOST = "localhost"
CSV_PATH = "../amz_uk_processed_data.csv"
MODEL_NAME = 'all-MiniLM-L12-v2'

def ingest():
    print("🚀 Starting Data Ingestion...")
    
    # 1. Setup Connections
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    q_client = QdrantClient(host=QDRANT_HOST, port=6333)
    model = SentenceTransformer(MODEL_NAME)
    
    # 2. Create Tables & Collection
    Base.metadata.create_all(engine)
    
    q_client.recreate_collection(
        collection_name="products",
        vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.IP),
    )
    
    # 3. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found at {CSV_PATH}. Please ensure the file exists.")
        return

    df = pd.read_csv(CSV_PATH).head(5000) # Sample for speed in demo
    print(f"📈 Processing {len(df)} items...")

    items_to_add = []
    points = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Create DB Item
        item = Item(
            asin=row['asin'],
            title=row['title'],
            category=row.get('categoryName', 'Unknown'),
            price=float(row['price']),
            stars=float(row.get('stars', 0)),
            img_url=row.get('imgUrl', '')
        )
        items_to_add.append(item)
        
        # Prepare Qdrant Point
        emb = model.encode(str(row['title'])).tolist()
        points.append(qmodels.PointStruct(
            id=i,
            vector=emb,
            payload={
                "asin": row['asin'],
                "title": row['title'],
                "category": row.get('categoryName', 'Unknown'),
                "price": float(row['price']),
                "stars": float(row.get('stars', 0)),
                "popularity_score": float(row.get('boughtInLastMonth', 0))
            }
        ))

    # 4. Batch Upload
    print("💾 Saving to PostgreSQL...")
    session.add_all(items_to_add)
    session.commit()
    
    print("🧠 Uploading to Qdrant...")
    q_client.upsert(collection_name="products", points=points)
    
    # 5. Create Mock User & Interactions
    print("👤 Creating mock user 'user_demo'...")
    user = User(external_id="user_demo")
    session.add(user)
    session.commit()
    
    # Add some interactions for the user
    for item in items_to_add[:5]:
        interaction = Interaction(
            user_id=user.id,
            item_id=item.id,
            interaction_type="view",
            rating=5.0
        )
        session.add(interaction)
    session.commit()

    print("✅ Ingestion Complete!")

if __name__ == "__main__":
    ingest()
