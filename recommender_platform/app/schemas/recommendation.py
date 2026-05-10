from pydantic import BaseModel
from pydantic import ConfigDict
from typing import List, Optional, Dict

class ItemDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asin: str
    title: str
    price: float
    stars: Optional[float] = 0.0
    category: Optional[str] = "General"
    reviews: int = 0
    bought_in_last_month: int = 0
    is_best_seller: bool = False
    product_url: Optional[str] = None
    img_url: Optional[str] = None
    score: Optional[float] = None
    explanation: Optional[Dict[str, float]] = None
    explanation_text: Optional[str] = None

class RecResponse(BaseModel):
    user_id: Optional[str] = None
    seed_item: Optional[Dict] = None
    recommendations: List[ItemDetail]
    strategy: str
    diversity_score: Optional[float] = None
