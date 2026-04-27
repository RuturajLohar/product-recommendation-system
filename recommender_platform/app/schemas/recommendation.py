from pydantic import BaseModel
from typing import List, Optional, Dict

class ItemDetail(BaseModel):
    asin: str
    title: str
    price: float
    stars: Optional[float] = 0.0
    category: Optional[str] = "General"
    image_url: Optional[str] = None
    score: Optional[float] = None
    explanation: Optional[Dict[str, float]] = None

class RecResponse(BaseModel):
    user_id: Optional[int] = None
    seed_item: Optional[Dict] = None
    recommendations: List[ItemDetail]
    strategy: str
    diversity_score: Optional[float] = None
