"""
train.py — Full training orchestrator.

Responsibilities:
  1. Load or generate dataset
  2. Clean + feature-engineer the data
  3. Split train / test (stratified)
  4. Build and fit preprocessing pipeline
  5. Train all four models (with SMOTE)
  6. Hyperparameter-tune top candidates
  7. Evaluate all models
  8. Select + serialise the best model
  9. Generate and save all visualisations
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# ── Make sure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    get_logger, ensure_dirs, generate_synthetic_dataset,
    summarise_data, evaluate_model, select_best_model,
    save_metadata, feature_engineering,
    plot_class_distribution, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance, plot_model_comparison,
    plot_risk_score_distribution,
)
from pipeline.build_pipeline import (
    build_preprocessing_pipeline, build_full_pipeline,
    tune_model, get_model_definitions, get_feature_names,
    IsolationForestWrapper, save_pipeline,
)

logger = get_logger("train")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if os.path.exists(config.DATASET_PATH):
        logger.info(f"Loading dataset from {config.DATASET_PATH}…")
        df = pd.read_csv(config.DATASET_PATH)
    else:
        logger.info("No dataset found — generating synthetic data…")
        df = generate_synthetic_dataset()
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(config.DATASET_PATH, index=False)
        logger.info(f"Synthetic dataset saved → {config.DATASET_PATH}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply derived / behavioural features that may not already be in the raw data.
    Safe to call even if columns already exist (idempotent).
    """
    # Amount deviation from personal mean (if avg_amount_7d column exists)
    if "avg_amount_7d" in df.columns and "amount_deviation" not in df.columns:
        df["amount_deviation"] = (
            (df["amount"] - df["avg_amount_7d"]) / (df["avg_amount_7d"] + 1e-9)
        ).round(4)

    # Binary time flags
    if "transaction_hour" in df.columns:
        if "is_night" not in df.columns:
            df["is_night"]   = (df["transaction_hour"] < 6).astype(int)
    if "transaction_day" in df.columns:
        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["transaction_day"] >= 5).astype(int)

    return df


# ─────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    • Drop irrelevant columns
    • Remove duplicate rows
    • Cap extreme outliers at the 1st/99th percentile for numerical columns
      (instead of dropping, to preserve fraud records which may be extreme)
    """
    # Drop configured irrelevant columns (silently skip missing ones)
    drop_cols = [c for c in config.DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    if before != after:
        logger.info(f"Dropped {before - after:,} duplicate rows.")

    # Winsorise numerical outliers (cap rather than drop — preserves fraud samples)
    for col in config.NUMERICAL_COLUMNS:
        if col not in df.columns:
            continue
        p01 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p01, upper=p99)

    return df


# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,       # maintain class ratio in both splits
    )
    logger.info(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    logger.info(f"Train fraud rate: {y_train.mean():.2%}  |  "
                f"Test fraud rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 5. TRAIN ALL MODELS
# ─────────────────────────────────────────────

def train_all_models(X_train, y_train, preprocessor):
    """
    Trains all models defined in get_model_definitions().
    Returns dict of {name: fitted_pipeline_or_wrapper}.
    """
    model_defs  = get_model_definitions()
    fitted_models = {}

    for name, estimator in model_defs.items():
        logger.info(f"\n{'─'*50}")
        logger.info(f"Training: {name}")

        if name == "Isolation Forest":
            # Anomaly detection — fit on full training set (no labels needed)
            wrapper = IsolationForestWrapper(preprocessor)
            wrapper.fit(X_train)
            fitted_models[name] = wrapper
        else:
            pipeline = build_full_pipeline(preprocessor, estimator)
            pipeline.fit(X_train, y_train)
            fitted_models[name] = pipeline

        logger.info(f"✓  {name} trained.")

    return fitted_models


# ─────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING (top 2 models)
# ─────────────────────────────────────────────

def tune_top_models(
    initial_results: list,
    X_train, y_train,
    preprocessor,
    top_n: int = 2,
) -> dict:
    """
    Re-train the top N supervised models with hyperparameter search.
    Returns dict of {name: tuned_pipeline}.
    """
    # Rank by selection metric (exclude Isolation Forest from tuning)
    ranked = sorted(
        [r for r in initial_results if r["model"] != "Isolation Forest"],
        key=lambda r: r[config.SELECTION_METRIC],
        reverse=True,
    )[:top_n]

    tuned = {}
    model_defs = get_model_definitions()

    for res in ranked:
        name = res["model"]
        logger.info(f"\nTuning {name}…")
        pipeline = build_full_pipeline(preprocessor, model_defs[name])
        tuned_pipeline = tune_model(pipeline, X_train, y_train, name)
        tuned[name] = tuned_pipeline

    return tuned


# ─────────────────────────────────────────────
# MAIN TRAINING RUN
# ─────────────────────────────────────────────

def run_training():
    ensure_dirs()
    logger.info("=" * 60)
    logger.info("  FRAUD DETECTION — TRAINING PIPELINE START")
    logger.info("=" * 60)

    # ── Load data
    df = load_data()
    logger.info(f"\nData summary:\n{json.dumps(summarise_data(df), indent=2, default=str)}")

    # ── Visualise class distribution
    plot_class_distribution(
        df[config.TARGET_COLUMN],
        save_path=os.path.join(config.VIZ_DIR, "01_class_distribution.png"),
    )

    # ── Feature engineering + cleaning
    df = feature_engineering(df)
    df = clean_data(df)
    logger.info(f"After cleaning — shape: {df.shape}")

    # ── Split
    X_train, X_test, y_train, y_test = split_data(df)

    # ── Build + fit preprocessor (fit on training set ONLY)
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)
    logger.info(f"Feature names after encoding ({len(feature_names)}): {feature_names}")

    # Persist preprocessor pipeline for inference
    save_pipeline(preprocessor, config.PIPELINE_PATH)

    # ── Train all models (initial run)
    logger.info("\n──── INITIAL MODEL TRAINING ────")
    fitted_models = train_all_models(X_train, y_train, preprocessor)

    # ── Evaluate initial models
    logger.info("\n──── INITIAL EVALUATION ────")
    initial_results = []
    for name, model in fitted_models.items():
        res = evaluate_model(model, X_test, y_test, model_name=name)
        initial_results.append(res)

        plot_confusion_matrix(
            y_test, res["y_pred"], model_name=name,
            save_path=os.path.join(config.VIZ_DIR, f"02_cm_{name.replace(' ', '_')}.png"),
        )

    # ── Tune top 2 models
    logger.info("\n──── HYPERPARAMETER TUNING ────")
    tuned_models = tune_top_models(initial_results, X_train, y_train, preprocessor)

    # ── Evaluate tuned models
    logger.info("\n──── TUNED MODEL EVALUATION ────")
    tuned_results = []
    for name, model in tuned_models.items():
        res = evaluate_model(model, X_test, y_test, model_name=f"{name} (Tuned)")
        tuned_results.append(res)
        fitted_models[f"{name} (Tuned)"] = model   # add to pool

    # ── Merge all results
    all_results = initial_results + tuned_results

    # ── Select best model
    logger.info("\n──── MODEL SELECTION ────")
    best_result = select_best_model(all_results, metric=config.SELECTION_METRIC)
    best_model  = fitted_models[best_result["model"]]

    # ── Save best model
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, config.MODEL_PATH, compress=3)
    logger.info(f"Best model saved → {config.MODEL_PATH}")

    # ── Save metadata
    save_metadata(best_result, feature_names)

    # ── Advanced visualisations
    logger.info("\n──── GENERATING VISUALISATIONS ────")

    # ROC + PR curves (all models)
    plot_roc_curve(
        all_results, y_test,
        save_path=os.path.join(config.VIZ_DIR, "03_roc_curves.png"),
    )
    plot_precision_recall_curve(
        all_results, y_test,
        save_path=os.path.join(config.VIZ_DIR, "04_pr_curves.png"),
    )
    plot_model_comparison(
        all_results,
        save_path=os.path.join(config.VIZ_DIR, "05_model_comparison.png"),
    )

    # Feature importance for best supervised model
    _best_clf = (
        best_model.named_steps.get("classifier") or
        best_model.named_steps.get("classifier", None)
    )
    if _best_clf is not None:
        plot_feature_importance(
            _best_clf, feature_names,
            model_name=best_result["model"],
            save_path=os.path.join(config.VIZ_DIR, "06_feature_importance.png"),
        )

    # Risk score distribution
    y_prob = best_result["y_prob"]
    y_test_arr = np.array(y_test)
    plot_risk_score_distribution(
        y_prob[y_test_arr == 0], y_prob[y_test_arr == 1],
        model_name=best_result["model"],
        save_path=os.path.join(config.VIZ_DIR, "07_risk_score_distribution.png"),
    )

    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info(f"  Best model : {best_result['model']}")
    logger.info(f"  ROC-AUC    : {best_result['roc_auc']:.4f}")
    logger.info(f"  Recall     : {best_result['recall']:.4f}")
    logger.info(f"  F1 Score   : {best_result['f1']:.4f}")
    logger.info("=" * 60)

    return best_model, feature_names


if __name__ == "__main__":
    run_training()
