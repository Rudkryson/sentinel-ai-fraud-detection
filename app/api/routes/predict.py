from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from sqlalchemy.orm import Session
from datetime import datetime
import pandas as pd

from app.api.deps import get_db, get_current_user, limiter
from app.models.user import User
from app.models.transaction import Transaction
from app.models.alert import Alert
from app.schemas.transaction import TransactionCreate, PredictionResponse
from app.services.predict import predict, predict_batch
from app.services.utils import get_logger

router = APIRouter()
logger = get_logger("api_predict")

@router.post("/", response_model=PredictionResponse)
@limiter.limit("20/minute")
def api_predict(
    request: Request,
    payload: TransactionCreate, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    try:
        # Run ML model
        result = predict(payload.dict())
        
        # Save Transaction
        db_txn = Transaction(
            **payload.dict(),
            risk_score=result["risk_score"],
            is_fraud=result["fraud"]
        )
        db.add(db_txn)
        db.commit()
        db.refresh(db_txn)
        
        # Save Alert if Fraud
        if result["fraud"] == 1:
            db_alert = Alert(
                transaction_id=db_txn.id,
                risk_score=result["risk_score"],
                risk_level=result["risk_level"]
            )
            db.add(db_alert)
            db.commit()
            
        result["timestamp"] = datetime.utcnow().isoformat()
        return result
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch")
@limiter.limit("5/minute")
def api_batch_predict(
    request: Request,
    file: UploadFile = File(...), 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        df = pd.read_csv(file.file)
        required = list(TransactionCreate.model_fields.keys())
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        
        transactions = df[required].to_dict(orient='records')
        results = predict_batch(transactions)
        
        # We could save all batch transactions to DB here, but for performance 
        # we might just save the fraudulent ones as alerts, or batch insert.
        # For simplicity in this demo, we'll return the results.
        
        for i, res in enumerate(results):
            if "error" not in res:
                res["amount"] = transactions[i]["amount"]
                res["type"] = transactions[i]["transaction_type"]
                res["timestamp"] = datetime.utcnow().isoformat()
        
        return {"total": len(results), "results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
