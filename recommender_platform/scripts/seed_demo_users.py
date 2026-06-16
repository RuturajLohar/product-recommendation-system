import os
import sys
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import models
from app.db.session import SessionLocal, engine


DEMO_USERS = {
    "USER_0": ["speaker", "sound", "bluetooth"],
    "USER_1": ["echo", "smart", "alexa"],
    "USER_2": ["mini", "portable", "budget"],
    "USER_3": ["home", "wireless", "stereo"],
    "CODEx_SMOKE_USER": ["anker", "soundcore", "speaker"],
}


def _find_demo_items(db: Session, keywords: list[str], limit: int = 5) -> list[models.Item]:
    seen = set()
    items: list[models.Item] = []
    for keyword in keywords:
        rows = (
            db.query(models.Item)
            .filter(models.Item.title.ilike(f"%{keyword}%"))
            .order_by(models.Item.stars.desc(), models.Item.reviews.desc())
            .limit(limit)
            .all()
        )
        for item in rows:
            if item.id not in seen:
                seen.add(item.id)
                items.append(item)
        if len(items) >= limit:
            break
    if len(items) < limit:
        fallback = db.query(models.Item).order_by(models.Item.stars.desc()).limit(limit).all()
        for item in fallback:
            if item.id not in seen:
                items.append(item)
    return items[:limit]


def seed_demo_users() -> None:
    models.Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    try:
        if db.query(models.Item).count() == 0:
            print("No products found. Run ingest_data.py first.")
            return

        for external_id, keywords in DEMO_USERS.items():
            user = db.query(models.User).filter(models.User.external_id == external_id).first()
            if not user:
                user = models.User(external_id=external_id)
                db.add(user)
                db.commit()
                db.refresh(user)

            items = _find_demo_items(db, keywords)
            for offset, item in enumerate(items):
                already_seeded = (
                    db.query(models.Interaction)
                    .filter(
                        models.Interaction.user_id == user.id,
                        models.Interaction.item_id == item.id,
                    )
                    .first()
                )
                if already_seeded:
                    continue
                db.add(
                    models.Interaction(
                        user_id=user.id,
                        item_id=item.id,
                        interaction_type="click" if offset < 3 else "view",
                        timestamp=datetime.utcnow() - timedelta(minutes=offset),
                    )
                )
            db.commit()
            print(f"Seeded {external_id} with {len(items)} demo items.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo_users()
