from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List
from ..db.session import get_db
from ..db import models
from ..schemas.recommendation import ItemDetail

router = APIRouter()

@router.get("/", response_model=List[ItemDetail])
def get_items(
    limit: int = 20, 
    offset: int = 0,
    db: Session = Depends(get_db)
):
    items = db.query(models.Item).offset(offset).limit(limit).all()
    return items

@router.get("/{asin}", response_model=ItemDetail)
def get_item(asin: str, db: Session = Depends(get_db)):
    item = db.query(models.Item).filter(models.Item.asin == asin).first()
    return item
