from fastapi import APIRouter, Depends
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db


router = APIRouter()


@router.get("/summary")
def get_summary(db: Session = Depends(get_db)):
    total_products = db.query(func.count(models.Item.id)).scalar() or 0
    total_users = db.query(func.count(models.User.id)).scalar() or 0
    total_interactions = db.query(func.count(models.Interaction.id)).scalar() or 0

    top_category = (
        db.query(models.Item.category, func.count(models.Item.id).label("count"))
        .group_by(models.Item.category)
        .order_by(desc("count"))
        .first()
    )

    most_interacted = (
        db.query(
            models.Item.asin,
            models.Item.title,
            models.Item.category,
            func.count(models.Interaction.id).label("count"),
        )
        .join(models.Interaction, models.Interaction.item_id == models.Item.id)
        .group_by(models.Item.asin, models.Item.title, models.Item.category)
        .order_by(desc("count"))
        .first()
    )

    return {
        "total_products": int(total_products),
        "total_users": int(total_users),
        "total_interactions": int(total_interactions),
        "top_category": {
            "name": top_category.category if top_category else "No data",
            "count": int(top_category.count) if top_category else 0,
        },
        "most_interacted_product": {
            "asin": most_interacted.asin if most_interacted else None,
            "title": most_interacted.title if most_interacted else "No interactions yet",
            "category": most_interacted.category if most_interacted else None,
            "count": int(most_interacted.count) if most_interacted else 0,
        },
        "active_strategy": "SBERT + Qdrant retrieval, XGBoost-style ranking fallback, MMR diversity",
    }
