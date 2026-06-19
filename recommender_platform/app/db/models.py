from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "rec_users"
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True) # ID from client system
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    interactions = relationship("Interaction", back_populates="user")

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True, index=True)
    asin = Column(String, unique=True, index=True)
    product_id = Column(String, unique=True, index=True)
    title = Column(String, index=True)
    brand = Column(String, index=True)
    category = Column(String, index=True)
    subcategory = Column(String, index=True)
    description = Column(Text, nullable=True)
    features = Column(JSON, nullable=True)
    specifications = Column(JSON, nullable=True)
    keywords_tags = Column(Text, nullable=True)
    price = Column(Float)
    stars = Column(Float)
    rating = Column(Float)
    reviews = Column(Integer, default=0)
    review_count = Column(Integer, default=0)
    popularity = Column(Float, default=0.0, index=True)
    bought_in_last_month = Column(Integer, default=0)
    is_best_seller = Column(Boolean, default=False)
    best_seller = Column(Boolean, default=False)
    availability = Column(String, index=True)
    product_url = Column(String, nullable=True)
    img_url = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    
    interactions = relationship("Interaction", back_populates="item")

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("rec_users.id"))
    item_id = Column(Integer, ForeignKey("items.id"))
    interaction_type = Column(String) # click, purchase, view
    rating = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")
