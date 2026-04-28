"""
pipeline/build_pipeline.py
--------------------------
Constructs the full sklearn preprocessing + modelling pipelines.
Uses only scikit-learn (no imblearn dependency) with:
  - Manual SMOTE (pure numpy)
  - class_weight="balanced" on all supervised classifiers
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger

logger = get_logger("pipeline")


# ─────────────────────────────────────────────
# MANUAL SMOTE  (pure numpy)
# ─────────────────────────────────────────────

def manual_smote(X, y, sampling_strategy=0.3, k=5, random_state=42):
    rng = np.random.default_rng(random_state)
    minority_idx = np.where(y == 1)[0]
    majority_idx = np.where(y == 0)[0]
    n_minority = len(minority_idx)
    n_majority = len(majority_idx)
    target_minority = int(n_majority * sampling_strategy)

    if target_minority <= n_minority:
        logger.info("SMOTE skipped — minority class already at target ratio.")
        return X, y

    n_synthetic = target_minority - n_minority
    X_min = X[minority_idx]
    synthetic_samples = []

    for _ in range(n_synthetic):
        idx = rng.integers(0, n_minority)
        sample = X_min[idx]
        dists = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf
        nn_indices = np.argpartition(dists, min(k, n_minority - 1))[:k]
        chosen = X_min[rng.choice(nn_indices)]
        alpha = rng.uniform(0, 1)
        synthetic_samples.append(sample + alpha * (chosen - sample))

    X_syn = np.array(synthetic_samples)
    y_syn = np.ones(n_synthetic, dtype=y.dtype)
    X_out = np.vstack([X, X_syn])
    y_out = np.concatenate([y, y_syn])
    logger.info(f"SMOTE generated {n_synthetic:,} synthetic fraud samples → shape {X_out.shape}")
    return X_out, y_out


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def build_preprocessing_pipeline(
    numerical_cols=None,
    categorical_cols=None,
):
    numerical_cols   = numerical_cols   or config.NUMERICAL_COLUMNS
    categorical_cols = categorical_cols or config.CATEGORICAL_COLUMNS

    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer,   numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor):
    num_features = list(preprocessor.transformers_[0][2])
    cat_encoder  = preprocessor.transformers_[1][1].named_steps["encoder"]
    cat_features = list(cat_encoder.get_feature_names_out(
        preprocessor.transformers_[1][2]
    ))
    return num_features + cat_features


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_model_definitions():
    return {
        "Logistic Regression": LogisticRegression(
            random_state=config.RANDOM_STATE, max_iter=1000,
            class_weight="balanced", C=1.0, solver="saga",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=config.RANDOM_STATE,
            n_jobs=-1, class_weight="balanced", max_depth=20,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=config.RANDOM_STATE, subsample=0.8,
        ),
        "Isolation Forest": IsolationForest(**config.ISOLATION_FOREST_PARAMS),
    }


# ─────────────────────────────────────────────
# SMOTE WRAPPER CLASSIFIER
# ─────────────────────────────────────────────

class SMOTEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, sampling_strategy=0.3, k=5, random_state=42):
        self.estimator         = estimator
        self.sampling_strategy = sampling_strategy
        self.k                 = k
        self.random_state      = random_state

    def fit(self, X, y):
        X_res, y_res = manual_smote(
            X, y,
            sampling_strategy=self.sampling_strategy,
            k=self.k, random_state=self.random_state,
        )
        self.estimator.fit(X_res, y_res)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def feature_importances_(self):
        return getattr(self.estimator, "feature_importances_", None)

    @property
    def coef_(self):
        return getattr(self.estimator, "coef_", None)


# ─────────────────────────────────────────────
# FULL PIPELINE BUILDER
# ─────────────────────────────────────────────

def build_full_pipeline(preprocessor, estimator, use_smote=True):
    is_anomaly = isinstance(estimator, IsolationForest)
    if use_smote and not is_anomaly and config.IMBALANCE_STRATEGY in ("smote", "both"):
        clf = SMOTEClassifier(
            estimator=estimator,
            sampling_strategy=config.SMOTE_SAMPLING_STRATEGY,
            random_state=config.RANDOM_STATE,
        )
    else:
        clf = estimator

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_model(pipeline, X_train, y_train, model_name):
    clf_step = pipeline.named_steps["classifier"]
    if isinstance(clf_step, SMOTEClassifier):
        prefix = "classifier__estimator__"
    else:
        prefix = "classifier__"

    tunable = {
        "Random Forest":       config.RF_PARAM_GRID,
        "Gradient Boosting":   config.GB_PARAM_GRID,
        "Logistic Regression": config.LR_PARAM_GRID,
    }
    if model_name not in tunable:
        logger.info(f"No tuning grid for {model_name}; skipping.")
        return pipeline

    param_grid = {f"{prefix}{k}": v for k, v in tunable[model_name].items()}
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=config.SEARCH_ITER,
        cv=cv,
        scoring=config.SCORING_METRIC,
        n_jobs=-1,
        refit=True,
        verbose=1,
        random_state=config.RANDOM_STATE,
        error_score=0,
    )
    logger.info(f"RandomizedSearchCV — {model_name} ({config.SEARCH_ITER} iter × {config.CV_FOLDS}-fold)…")
    search.fit(X_train, y_train)
    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best CV {config.SCORING_METRIC}: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────
# ISOLATION FOREST WRAPPER
# ─────────────────────────────────────────────

class IsolationForestWrapper:
    def __init__(self, preprocessor, params=None):
        self.preprocessor = preprocessor
        self.model = IsolationForest(**(params or config.ISOLATION_FOREST_PARAMS))

    def fit(self, X, y=None):
        X_t = self.preprocessor.transform(X)
        self.model.fit(X_t)
        return self

    def predict(self, X):
        X_t = self.preprocessor.transform(X)
        raw = self.model.predict(X_t)
        return np.where(raw == -1, 1, 0)

    def decision_function(self, X):
        X_t = self.preprocessor.transform(X)
        return self.model.decision_function(X_t)


# ─────────────────────────────────────────────
# SERIALISATION
# ─────────────────────────────────────────────

def save_pipeline(pipeline, path=config.PIPELINE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    logger.info(f"Pipeline saved → {path}")


def load_pipeline(path=config.PIPELINE_PATH):
    pipeline = joblib.load(path)
    logger.info(f"Pipeline loaded ← {path}")
    return pipeline
