from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
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
    title = Column(String, index=True)
    category = Column(String, index=True)
    price = Column(Float)
    stars = Column(Float)
    reviews = Column(Integer, default=0)
    bought_in_last_month = Column(Integer, default=0)
    is_best_seller = Column(Boolean, default=False)
    product_url = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    img_url = Column(String, nullable=True)
    
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
