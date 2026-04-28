from fastapi import APIRouter
from app.api.routes import auth, predict, dashboard

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(predict.router, prefix="/predict", tags=["predict"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
