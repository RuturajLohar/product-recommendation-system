from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from ..db.session import get_db
from ..db import models
from ..schemas.recommendation import ItemDetail

router = APIRouter()

@router.get("/", response_model=List[ItemDetail])
def get_items(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, min_length=1, max_length=200),
    db: Session = Depends(get_db)
):
    query = db.query(models.Item)
    if q:
        query = query.filter(models.Item.title.ilike(f"%{q}%")).order_by(models.Item.bought_in_last_month.desc())
    items = query.offset(offset).limit(limit).all()
    return items

@router.get("/{asin}", response_model=ItemDetail)
def get_item(asin: str, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.asin == asin).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item
