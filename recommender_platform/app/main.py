import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import recommendations, users, items, events, analytics
from .core.config import settings
from .db.session import SessionLocal, engine
from .db import models
from .ml.engine.hybrid import HybridEngine
from .ml.ranking.ranker import XGBRanker

# Create tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EliteRec SaaS Platform",
    description="Production-grade Product Recommendation API",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_recommendation_components():
    app.state.engine = HybridEngine()
    app.state.ranker = XGBRanker(model_path=settings.RANKER_MODEL_PATH)


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
app.include_router(events.router, prefix="/api/v1", tags=["events"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/health/ready")
def readiness_check():
    engine_loaded = hasattr(app.state, "engine")
    ranker_loaded = hasattr(app.state, "ranker")
    return {
        "status": "ready" if engine_loaded and ranker_loaded else "starting",
        "engine_loaded": engine_loaded,
        "ranker_loaded": ranker_loaded,
        "timestamp": time.time(),
    }


@app.get("/")
def root():
    return {"message": "Welcome to EliteRec SaaS Platform API"}
