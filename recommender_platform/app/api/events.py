from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import models
from ..db.session import get_db


router = APIRouter()


class EventIn(BaseModel):
    user_id: str
    asin: str
    type: Literal["view", "click", "purchase"]


@router.post("/events", status_code=204)
def create_event(payload: EventIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.external_id == payload.user_id).first()
    if not user:
        user = models.User(external_id=payload.user_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    item = db.query(models.Item).filter(models.Item.asin == payload.asin).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    interaction = models.Interaction(
        user_id=user.id,
        item_id=item.id,
        interaction_type=payload.type,
        timestamp=datetime.utcnow(),
    )
    db.add(interaction)
    db.commit()

    return None

