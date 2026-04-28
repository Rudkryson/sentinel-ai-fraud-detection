from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0, example=5000.0)
    transaction_hour: int = Field(..., ge=0, le=23, example=2)
    transaction_day: int = Field(..., ge=0, le=6, example=6)
    location: str = Field(..., max_length=100, example="international")
    transaction_type: str = Field(..., max_length=50, example="wire_transfer")
    transaction_freq_7d: int = Field(..., ge=0, example=1)
    avg_amount_7d: float = Field(..., ge=0, example=200.0)
    amount_deviation: float = Field(..., example=24.0)
    is_night: int = Field(..., ge=0, le=1, example=1)
    is_weekend: int = Field(..., ge=0, le=1, example=1)

class TransactionCreate(TransactionBase):
    pass

class Transaction(TransactionBase):
    id: int
    risk_score: Optional[float] = None
    is_fraud: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    fraud: int
    risk_score: float
    risk_level: str
    model: str
    threshold: float
    timestamp: str
