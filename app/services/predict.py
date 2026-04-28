"""
predict.py — Production-grade prediction & risk-scoring module.

Public API:
    predict(input_data: dict) -> dict
        Input:  raw transaction dict
        Output: {"fraud": 0|1, "risk_score": float, "risk_level": str,
                 "model": str, "threshold": float}

Can be imported directly by a REST API framework (FastAPI, Flask) or
called from app.py / any script.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import get_logger, validate_input, load_metadata

logger = get_logger("predict")

# ─────────────────────────────────────────────
# SINGLETONS (loaded once at module import)
# ─────────────────────────────────────────────
_model      = None
_pipeline   = None
_metadata   = None


def _load_artifacts():
    """Load model + preprocessing pipeline into module-level singletons."""
    global _model, _pipeline, _metadata

    if _model is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {config.MODEL_PATH}. "
                "Run train.py first."
            )
        _model = joblib.load(config.MODEL_PATH)
        logger.info(f"Model loaded ← {config.MODEL_PATH}")

    if _pipeline is None:
        if not os.path.exists(config.PIPELINE_PATH):
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at {config.PIPELINE_PATH}. "
                "Run train.py first."
            )
        _pipeline = joblib.load(config.PIPELINE_PATH)
        logger.info(f"Preprocessing pipeline loaded ← {config.PIPELINE_PATH}")

    if _metadata is None and os.path.exists(config.METADATA_PATH):
        _metadata = load_metadata()

    return _model, _pipeline, _metadata


# ─────────────────────────────────────────────
# HELPER — RISK LEVEL LABEL
# ─────────────────────────────────────────────

def _risk_level(score: float) -> str:
    if score >= 0.80:
        return "CRITICAL"
    elif score >= 0.60:
        return "HIGH"
    elif score >= 0.40:
        return "MEDIUM"
    elif score >= 0.20:
        return "LOW"
    else:
        return "MINIMAL"


# ─────────────────────────────────────────────
# CORE PREDICT FUNCTION
# ─────────────────────────────────────────────

def predict(input_data: dict) -> dict:
    """
    Score a single transaction.

    Parameters
    ----------
    input_data : dict
        Must contain the keys listed in utils.REQUIRED_FIELDS.
        Example:
            {
                "amount":              5000.0,
                "transaction_hour":    2,
                "transaction_day":     6,
                "location":            "international",
                "transaction_type":    "online",
                "transaction_freq_7d": 1,
                "avg_amount_7d":       200.0,
                "amount_deviation":    24.0,
                "is_night":            1,
                "is_weekend":          1,
            }

    Returns
    -------
    dict
        {
            "fraud":       0 or 1,
            "risk_score":  float (0.0 – 1.0),
            "risk_level":  "MINIMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "model":       str,
            "threshold":   float,
        }

    Raises
    ------
    ValueError   — if input_data fails validation
    FileNotFoundError — if model artefacts are missing
    """
    # 1. Validate + coerce input
    cleaned = validate_input(input_data)

    # 2. Load artefacts (cached after first call)
    model, pipeline, metadata = _load_artifacts()
    threshold = (metadata or {}).get("threshold", config.FRAUD_THRESHOLD)
    model_name = (metadata or {}).get("model_name", "Unknown")

    # 3. Convert to DataFrame with correct column order
    feature_order = config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS
    row = {col: cleaned.get(col, np.nan) for col in feature_order}
    df_input = pd.DataFrame([row])

    # 4. Check whether model is a full sklearn/imblearn Pipeline
    #    (has its own preprocessor step) or a bare IsolationForestWrapper
    if hasattr(model, "predict_proba"):
        # sklearn / imblearn pipeline — preprocessing is baked in
        risk_score = float(model.predict_proba(df_input)[0, 1])
    elif hasattr(model, "decision_function"):
        # IsolationForestWrapper
        raw = model.decision_function(df_input)[0]
        # Normalise with sigmoid-like mapping (single sample — use stored stats)
        risk_score = float(1 / (1 + np.exp(raw * 10)))   # approx
    else:
        risk_score = float(model.predict(df_input)[0])

    fraud_flag = int(risk_score >= threshold)

    result = {
        "fraud":       fraud_flag,
        "risk_score":  round(risk_score, 6),
        "risk_level":  _risk_level(risk_score),
        "model":       model_name,
        "threshold":   threshold,
    }

    logger.info(
        f"Prediction → fraud={fraud_flag}  "
        f"risk_score={risk_score:.4f}  "
        f"level={result['risk_level']}"
    )
    return result


# ─────────────────────────────────────────────
# BATCH PREDICTION
# ─────────────────────────────────────────────

def predict_batch(transactions: list) -> list:
    """
    Score a list of transaction dicts using vectorized operations.
    Returns list of result dicts in the same order.
    Each failed validation is caught individually and returned as an error entry.
    """
    if not transactions:
        return []

    # 1. Load artefacts
    model, _, metadata = _load_artifacts()
    threshold = (metadata or {}).get("threshold", config.FRAUD_THRESHOLD)
    model_name = (metadata or {}).get("model_name", "Unknown")

    # 2. Validate and clean all inputs
    cleaned_txns = []
    errors = {}
    for i, txn in enumerate(transactions):
        try:
            cleaned_txns.append(validate_input(txn))
        except (ValueError, KeyError) as e:
            errors[i] = str(e)
            cleaned_txns.append(None)

    # 3. Create DataFrame for inference (only for valid ones)
    valid_indices = [i for i, v in enumerate(cleaned_txns) if v is not None]
    if not valid_indices:
        return [{"error": errors[i], "index": i} for i in range(len(transactions))]

    valid_data = [cleaned_txns[i] for i in valid_indices]
    feature_order = config.NUMERICAL_COLUMNS + config.CATEGORICAL_COLUMNS
    df_input = pd.DataFrame(valid_data)
    # Ensure correct feature order and handle missing cols if any (though validate_input ensures them)
    df_input = df_input.reindex(columns=feature_order)

    # 4. Vectorized Prediction
    if hasattr(model, "predict_proba"):
        risk_scores = model.predict_proba(df_input)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(df_input)
        # Normalise (approx sigmoid)
        risk_scores = 1 / (1 + np.exp(raw * 10))
    else:
        risk_scores = model.predict(df_input)

    # 5. Assemble results
    final_results = [None] * len(transactions)
    for idx, score in enumerate(risk_scores):
        real_idx = valid_indices[idx]
        s = float(score)
        fraud_flag = int(s >= threshold)
        final_results[real_idx] = {
            "fraud":       fraud_flag,
            "risk_score":  round(s, 6),
            "risk_level":  _risk_level(s),
            "model":       model_name,
            "threshold":   threshold,
        }

    for i in range(len(transactions)):
        if final_results[i] is None:
            final_results[i] = {"error": errors[i], "index": i}

    logger.info(f"Batch prediction complete: {len(valid_indices)} successful, {len(errors)} failed.")
    return final_results


# ─────────────────────────────────────────────
# CLI USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "amount":              5000.0,
        "transaction_hour":    2,
        "transaction_day":     6,
        "location":            "international",
        "transaction_type":    "online",
        "transaction_freq_7d": 1,
        "avg_amount_7d":       200.0,
        "amount_deviation":    24.0,
        "is_night":            1,
        "is_weekend":          1,
    }
    print(json.dumps(predict(sample), indent=2))
