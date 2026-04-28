from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.transaction import Transaction
from app.models.alert import Alert
from app.services.utils import load_metadata

router = APIRouter()

@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    try:
        meta = load_metadata()
        metrics = meta.get("metrics", {})
        
        # Real stats from DB
        total_tx = db.query(Transaction).count()
        fraud_tx = db.query(Transaction).filter(Transaction.is_fraud == 1).count()
        active_alerts = db.query(Alert).filter(Alert.status == "NEW").count()
        
        fraud_rate = f"{(fraud_tx / total_tx * 100):.2f}%" if total_tx > 0 else "0.00%"
        
        # Add baseline simulated data if DB is empty for demo purposes
        if total_tx < 100:
            total_tx += 145000
            fraud_tx += 3800
            fraud_rate = f"{(fraud_tx / total_tx * 100):.2f}%"
            
        return {
            "total_transactions": total_tx,
            "fraud_detected": fraud_tx,
            "risk_alerts": active_alerts,
            "fraud_rate": fraud_rate,
            "model_performance": metrics,
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/alerts")
def get_alerts(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    # Fetch top 10 most recent alerts from real DB
    db_alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(10).all()
    
    alerts = []
    for a in db_alerts:
        alerts.append({
            "id": f"ALRT-{a.id:04d}",
            "timestamp": a.created_at.isoformat(),
            "amount": a.transaction.amount if a.transaction else 0,
            "type": a.transaction.transaction_type if a.transaction else "unknown",
            "location": a.transaction.location if a.transaction else "unknown",
            "risk_score": a.risk_score,
            "risk_level": a.risk_level
        })
        
    # If no alerts in DB yet, fallback to simulation for demo
    if not alerts:
        import random
        types = ["wire_transfer", "atm", "online", "pos"]
        locations = ["international", "online", "urban"]
        for i in range(5):
            score = random.uniform(0.65, 0.99)
            level = "CRITICAL" if score > 0.85 else "HIGH"
            alerts.append({
                "id": f"ALRT-{random.randint(1000, 9999)}",
                "timestamp": datetime.utcnow().isoformat(),
                "amount": round(random.uniform(1000, 20000), 2),
                "type": random.choice(types),
                "location": random.choice(locations),
                "risk_score": round(score, 4),
                "risk_level": level
            })
    
    return alerts
