from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    full_name: str | None = Field(default=None, max_length=100)

class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128, description="Password must be at least 8 characters long")

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None
