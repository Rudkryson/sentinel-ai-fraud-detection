import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.api.api import api_router
from app.api.deps import limiter
from app.services.utils import get_logger

logger = get_logger("api")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Enterprise-grade fraud detection and risk scoring API by Rudkryson Tech.",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve Frontend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.get("/")
def read_root():
    return FileResponse(os.path.join(FRONTEND_DIR, "landing.html"))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/dashboard")
async def read_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# We must mount static files AFTER other specific routes
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
