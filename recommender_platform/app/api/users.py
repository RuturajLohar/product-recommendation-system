from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List

from ..db import models
from ..db.session import get_db
from ..schemas.recommendation import ItemDetail

router = APIRouter()

class UserCreateIn(BaseModel):
    external_id: str


@router.post("/", status_code=201)
def create_user(payload: UserCreateIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.external_id == payload.external_id).first()
    if user:
        return {"external_id": user.external_id}
    user = models.User(external_id=payload.external_id)
    db.add(user)
    db.commit()
    return {"external_id": user.external_id}


@router.get("/{external_id}/history", response_model=List[ItemDetail])
def get_user_history(
    external_id: str,
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.external_id == external_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    items = (
        db.query(models.Item)
        .join(models.Interaction)
        .filter(models.Interaction.user_id == user.id)
        .order_by(models.Interaction.timestamp.desc())
        .limit(limit)
        .all()
    )
    return items
