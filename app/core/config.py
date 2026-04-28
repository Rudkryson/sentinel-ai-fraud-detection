import os
import secrets
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Sentinel AI"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = os.getenv("SECRET_KEY", secrets.token_hex(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./sentinel.db"

settings = Settings()

if os.getenv("ENVIRONMENT") == "production" and not os.getenv("SECRET_KEY"):
    raise ValueError("SECRET_KEY environment variable is required in production environment")
