from pydantic import BaseModel
from pydantic import ConfigDict
from typing import List, Optional, Dict

class ItemDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    asin: str
    product_id: Optional[str] = None
    title: str
    brand: Optional[str] = None
    price: float
    stars: Optional[float] = 0.0
    rating: Optional[float] = 0.0
    category: Optional[str] = "General"
    subcategory: Optional[str] = None
    description: Optional[str] = None
    features: Optional[List[str]] = None
    specifications: Optional[Dict[str, str]] = None
    keywords_tags: Optional[str] = None
    reviews: int = 0
    review_count: int = 0
    popularity: float = 0.0
    bought_in_last_month: int = 0
    is_best_seller: bool = False
    best_seller: bool = False
    availability: Optional[str] = None
    product_url: Optional[str] = None
    img_url: Optional[str] = None
    image_url: Optional[str] = None
    score: Optional[float] = None
    explanation: Optional[Dict[str, float]] = None
    explanation_text: Optional[str] = None

class RecResponse(BaseModel):
    user_id: Optional[str] = None
    seed_item: Optional[Dict] = None
    recommendations: List[ItemDetail]
    strategy: str
    diversity_score: Optional[float] = None
