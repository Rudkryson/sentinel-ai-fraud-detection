from pydantic import BaseModel
from datetime import datetime

class AlertBase(BaseModel):
    transaction_id: int
    risk_score: float
    risk_level: str
    status: str = "NEW"

class AlertCreate(AlertBase):
    pass

class Alert(AlertBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True
