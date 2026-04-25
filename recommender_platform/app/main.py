from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import time

from .api import recommendations, users, items
from .core.config import settings
from .db.session import SessionLocal, engine
from .db import models

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EliteRec SaaS Platform",
    description="Production-grade Product Recommendation API",
    version="1.0.0"
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Include Routers
app.include_router(recommendations.router, prefix="/api/v1/recommend", tags=["recommendations"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(items.router, prefix="/api/v1/items", tags=["items"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
def root():
    return {"message": "Welcome to EliteRec SaaS Platform API"}
