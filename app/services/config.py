"""
config.py — Central configuration for the Fraud Detection System.
All paths, hyperparameters, thresholds, and constants live here.
Edit this file to adapt the system to a new dataset or deployment environment.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR        = os.path.join(BASE_DIR, "data")
MODEL_DIR       = os.path.join(BASE_DIR, "model")
PIPELINE_DIR    = os.path.join(BASE_DIR, "pipeline")
VIZ_DIR         = os.path.join(BASE_DIR, "visualizations")
NOTEBOOK_DIR    = os.path.join(BASE_DIR, "notebook")

DATASET_PATH    = os.path.join(DATA_DIR, "transactions.csv")
MODEL_PATH      = os.path.join(MODEL_DIR, "best_model.joblib")
PIPELINE_PATH   = os.path.join(PIPELINE_DIR, "preprocessing_pipeline.joblib")
METADATA_PATH   = os.path.join(MODEL_DIR, "model_metadata.json")

# ─────────────────────────────────────────────
# DATASET SCHEMA
# ─────────────────────────────────────────────
TARGET_COLUMN = "is_fraud"

# Columns to drop before modelling (IDs, timestamps in raw form, etc.)
DROP_COLUMNS = ["transaction_id", "customer_id", "raw_timestamp"]

# Categorical columns to be one-hot encoded
CATEGORICAL_COLUMNS = ["location", "transaction_type"]

# Numerical columns to be scaled
NUMERICAL_COLUMNS = [
    "amount",
    "transaction_hour",
    "transaction_day",
    "transaction_freq_7d",
    "avg_amount_7d",
    "amount_deviation",
    "is_night",
    "is_weekend",
]

# ─────────────────────────────────────────────
# RANDOM SEED (for reproducibility)
# ─────────────────────────────────────────────
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TEST_SIZE = 0.20

# ─────────────────────────────────────────────
# FRAUD CLASSIFICATION THRESHOLD
# Higher = fewer fraud alerts (more precise), Lower = more alerts (more recall)
# In production, tune this based on business cost of false negatives vs false positives
# ─────────────────────────────────────────────
FRAUD_THRESHOLD = 0.40   # probability >= 0.40 → labelled fraud

# ─────────────────────────────────────────────
# CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────
IMBALANCE_STRATEGY = "smote"   # options: "smote" | "class_weight" | "both"
SMOTE_SAMPLING_STRATEGY = 0.3  # minority class will be 30 % of majority after SMOTE

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETER GRIDS  (used in RandomizedSearchCV)
# ─────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 400],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
    "class_weight":      ["balanced", None],
}

GB_PARAM_GRID = {
    "n_estimators":   [100, 200, 300],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2],
    "max_depth":      [3, 5, 7],
    "subsample":      [0.7, 0.8, 1.0],
    "min_samples_split": [2, 5],
}

LR_PARAM_GRID = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "solver":  ["lbfgs", "saga"],
    "max_iter": [200, 500, 1000],
    "class_weight": ["balanced", None],
}

ISOLATION_FOREST_PARAMS = {
    "n_estimators":  200,
    "contamination": 0.05,   # expected fraction of anomalies
    "random_state":  RANDOM_STATE,
}

# CV folds and iterations for RandomizedSearchCV
CV_FOLDS       = 5
SEARCH_ITER    = 30          # number of random combinations to try
SCORING_METRIC = "f1"        # primary optimisation metric

# ─────────────────────────────────────────────
# MODEL SELECTION METRIC
# ─────────────────────────────────────────────
SELECTION_METRIC = "roc_auc"  # options: "f1" | "roc_auc" | "recall"

# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATION (used when no real dataset provided)
# ─────────────────────────────────────────────
SYNTHETIC_N_SAMPLES  = 50_000
SYNTHETIC_FRAUD_RATE = 0.04   # 4 % fraud rate — realistic for payments

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(BASE_DIR, "fraud_detection.log")
