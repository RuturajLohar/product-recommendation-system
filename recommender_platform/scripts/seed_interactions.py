import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import models
from app.db.session import SessionLocal, engine

def seed_interactions():
    # Ensure tables exist in this session
    models.Base.metadata.create_all(bind=engine)
    
    db: Session = SessionLocal()
    
    print("👤 Fetching users and items...")
    users = db.query(models.User).all()
    if not users:
        print("Creating users...")
        for i in range(100):
            db.add(models.User(external_id=f"USER_{i}"))
        db.commit()
        users = db.query(models.User).all()
        
    items = db.query(models.Item).all()
    if not items:
        print("❌ No items found. Run ingest_data.py first.")
        return

    user_ids = [u.id for u in users]
    item_ids = [it.id for it in items]
    
    print(f"🚀 Seeding 10,000 interactions for {len(users)} users and {len(items)} items...")
    
    interactions = []
    for i in range(10000):
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
        
        if len(interactions) >= 1000:
            db.add_all(interactions)
            db.commit()
            print(f"   - Inserted {i+1} interactions...")
            interactions = []
            
    if interactions:
        db.add_all(interactions)
        db.commit()
        
    print("✅ Interactions seeded successfully.")
    db.close()

if __name__ == "__main__":
    seed_interactions()
