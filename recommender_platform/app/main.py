from contextlib import asynccontextmanager
from fastapi import FastAPI
import time

from .api import recommendations, users, items, events
from .db.session import engine
from .db import models
from .ml.engine.hybrid import HybridEngine
from .ml.ranking.ranker import XGBRanker

# Create tables
models.Base.metadata.create_all(bind=engine)

from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = HybridEngine()
    app.state.ranker = XGBRanker()
    yield


app = FastAPI(
    title="EliteRec SaaS Platform",
    description="Production-grade Product Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(recommendations.router, prefix="/api/v1/recommend", tags=["recommendations"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])
app.include_router(items.router, prefix="/api/v1/items", tags=["items"])
app.include_router(events.router, prefix="/api/v1", tags=["events"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
def root():
    return {"message": "Welcome to EliteRec SaaS Platform API"}
