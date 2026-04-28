"""
utils.py — Shared utilities: logging, synthetic data generation,
           feature engineering, evaluation helpers, and visualisation.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score,
)

import config

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Return a configured logger that writes to both console and file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    os.makedirs(os.path.dirname(config.LOG_FILE) or ".", exist_ok=True)
    fh = logging.FileHandler(config.LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


logger = get_logger()


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
# DIRECTORY HELPERS
# ─────────────────────────────────────────────

def ensure_dirs():
    """Create all required output directories if they don't exist."""
    for d in [config.DATA_DIR, config.MODEL_DIR, config.PIPELINE_DIR, config.VIZ_DIR]:
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = config.SYNTHETIC_N_SAMPLES,
    fraud_rate: float = config.SYNTHETIC_FRAUD_RATE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic transaction dataset with:
    - Temporal patterns (hour, day)
    - Behavioural features (frequency, running averages)
    - Location and transaction-type categories
    - Class imbalance matching real-world fraud rates
    """
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    logger.info(f"Generating {n_samples:,} synthetic transactions "
                f"({n_fraud:,} fraud, {n_legit:,} legitimate)…")

    def _make_block(n, is_fraud):
        f = int(is_fraud)
        # Fraudulent transactions: higher amounts, unusual hours, online/card-not-present
        amounts = (
            rng.lognormal(mean=7.5, sigma=1.2, size=n)
            if f else rng.lognormal(mean=5.5, sigma=0.9, size=n)
        )
        hours = (
            rng.choice(np.concatenate([np.arange(0, 6), np.arange(22, 24)]), size=n)
            if f else rng.integers(8, 22, size=n)
        )
        days = rng.integers(0, 7, size=n)
        locations = (
            rng.choice(["international", "online"], p=[0.4, 0.6], size=n)
            if f else rng.choice(["urban", "suburban", "online", "rural", "international"],
                                  p=[0.35, 0.25, 0.25, 0.10, 0.05], size=n)
        )
        txn_types = (
            rng.choice(["online", "atm", "wire_transfer"], p=[0.5, 0.3, 0.2], size=n)
            if f else rng.choice(["online", "pos", "atm", "wire_transfer", "contactless"],
                                  p=[0.30, 0.35, 0.15, 0.10, 0.10], size=n)
        )
        freq   = rng.integers(1, 5, size=n) if f else rng.integers(1, 20, size=n)
        avg7d  = amounts * rng.uniform(0.5, 1.5, size=n)
        deviation = (amounts - avg7d) / (avg7d + 1e-9)

        return pd.DataFrame({
            "transaction_id":      [f"TXN{rng.integers(1e8, 1e9)}" for _ in range(n)],
            "customer_id":         [f"CUST{rng.integers(1e6, 1e7)}" for _ in range(n)],
            "amount":              np.round(amounts, 2),
            "transaction_hour":    hours.astype(int),
            "transaction_day":     days.astype(int),
            "location":            locations,
            "transaction_type":    txn_types,
            "transaction_freq_7d": freq.astype(int),
            "avg_amount_7d":       np.round(avg7d, 2),
            "amount_deviation":    np.round(deviation, 4),
            "is_night":            (hours < 6).astype(int),
            "is_weekend":          (days >= 5).astype(int),
            config.TARGET_COLUMN:  f,
        })

    df = pd.concat([_make_block(n_legit, False), _make_block(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Inject ~1 % missing values in amount (realistic for corrupt records)
    mask = rng.random(len(df)) < 0.01
    df.loc[mask, "amount"] = np.nan

    logger.info(f"Dataset shape: {df.shape}  |  "
                f"Fraud rate: {df[config.TARGET_COLUMN].mean():.2%}")
    return df


# ─────────────────────────────────────────────
# DATA QUALITY HELPERS
# ─────────────────────────────────────────────

def summarise_data(df: pd.DataFrame) -> dict:
    """Return a dict with shape, dtypes, missing counts, and class balance."""
    summary = {
        "shape":           df.shape,
        "columns":         list(df.columns),
        "missing_counts":  df.isnull().sum().to_dict(),
        "missing_pct":     (df.isnull().mean() * 100).round(2).to_dict(),
        "dtypes":          df.dtypes.astype(str).to_dict(),
    }
    if config.TARGET_COLUMN in df.columns:
        vc = df[config.TARGET_COLUMN].value_counts()
        summary["class_distribution"] = vc.to_dict()
        summary["fraud_rate"]         = float(df[config.TARGET_COLUMN].mean())
    return summary


def detect_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 3.0) -> pd.Series:
    """
    Return a boolean mask of rows that are extreme outliers in ANY of the given
    numerical columns (|z-score via IQR| > factor * IQR).
    """
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask |= (df[col] < lower) | (df[col] > upper)
    return mask


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    threshold: float = config.FRAUD_THRESHOLD,
) -> dict:
    """
    Full evaluation suite. Returns a metrics dict.
    Works for classifiers with predict_proba or decision_function.
    """
    # Probability or decision score
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        # Normalise Isolation Forest scores to [0, 1]
        y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        y_prob = 1 - y_prob          # invert: higher = more anomalous
    else:
        y_prob = model.predict(X_test).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model":         model_name,
        "threshold":     threshold,
        "accuracy":      round(accuracy_score(y_test, y_pred),  4),
        "precision":     round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_test, y_pred, zero_division=0),    4),
        "f1":            round(f1_score(y_test, y_pred, zero_division=0),        4),
        "roc_auc":       round(roc_auc_score(y_test, y_prob),  4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "y_prob":        y_prob,
        "y_pred":        y_pred,
    }

    logger.info(
        f"\n{'='*55}\n"
        f"  {model_name}\n"
        f"{'='*55}\n"
        f"  Accuracy   : {metrics['accuracy']:.4f}\n"
        f"  Precision  : {metrics['precision']:.4f}\n"
        f"  Recall     : {metrics['recall']:.4f}\n"
        f"  F1 Score   : {metrics['f1']:.4f}\n"
        f"  ROC-AUC    : {metrics['roc_auc']:.4f}\n"
        f"  Avg Prec.  : {metrics['avg_precision']:.4f}\n"
        f"{'='*55}\n"
        f"{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}"
    )
    return metrics


def select_best_model(results: list, metric: str = config.SELECTION_METRIC) -> dict:
    """Return the result dict with the highest value for `metric`."""
    best = max(results, key=lambda r: r[metric])
    logger.info(f"Best model: {best['model']}  |  {metric.upper()}: {best[metric]:.4f}")
    return best


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

PALETTE = {"Legit": "#2ecc71", "Fraud": "#e74c3c"}
FRAUD_COLOR  = "#e74c3c"
LEGIT_COLOR  = "#2ecc71"
BG_COLOR     = "#0d1117"
TEXT_COLOR   = "#c9d1d9"
GRID_COLOR   = "#21262d"

def _dark_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.6)
    return fig, ax


def plot_class_distribution(y: pd.Series, save_path: str = None):
    counts = y.value_counts()
    labels = ["Legitimate", "Fraud"]
    colors = [LEGIT_COLOR, FRAUD_COLOR]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    # Bar chart
    bars = axes[0].bar(labels, [counts.get(0, 0), counts.get(1, 0)],
                       color=colors, edgecolor="#ffffff22", linewidth=0.8)
    for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f"{count:,}", ha="center", va="bottom",
                     color=TEXT_COLOR, fontsize=10, fontweight="bold")
    axes[0].set_title("Transaction Class Distribution", color=TEXT_COLOR, fontsize=13, pad=12)
    axes[0].set_ylabel("Count", color=TEXT_COLOR)
    axes[0].tick_params(colors=TEXT_COLOR)
    axes[0].grid(color=GRID_COLOR, axis="y", linestyle="--", alpha=0.6)
    for spine in axes[0].spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=labels, colors=colors,
        autopct="%1.2f%%", startangle=90,
        wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
    )
    for t in texts + autotexts:
        t.set_color(TEXT_COLOR)
    axes[1].set_title("Class Ratio", color=TEXT_COLOR, fontsize=13, pad=12)

    plt.tight_layout(pad=2)
    _save_or_show(fig, save_path)


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        linewidths=0.5, linecolor=BG_COLOR,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Actual", color=TEXT_COLOR, fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curve(results: list, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    colours = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=colours[i % len(colours)],
                label=f"{res['model']}  (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", color=TEXT_COLOR)
    ax.set_ylabel("True Positive Rate", color=TEXT_COLOR)
    ax.set_title("ROC Curves — All Models", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#161b22", labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(color=GRID_COLOR, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_precision_recall_curve(results: list, y_test, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    colours = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, res in enumerate(results):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, lw=2, color=colours[i % len(colours)],
                label=f"{res['model']}  (AP={res['avg_precision']:.3f})")
    ax.axhline(y=sum(y_test)/len(y_test), color="#555", linestyle="--",
               lw=1, label="Baseline (fraud rate)")
    ax.set_xlabel("Recall", color=TEXT_COLOR)
    ax.set_ylabel("Precision", color=TEXT_COLOR)
    ax.set_title("Precision-Recall Curves — All Models", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#161b22", labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(color=GRID_COLOR, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importance(model, feature_names: list, model_name="Model", top_n=20, save_path=None):
    """Works for tree-based models with feature_importances_ and LogReg with coef_."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title_suffix = "(Gini Importance)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title_suffix = "(|Coefficient|)"
    else:
        logger.warning(f"{model_name} does not expose feature importances.")
        return

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    colours = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(top_features)))
    bars = ax.barh(top_features[::-1], top_values[::-1], color=colours[::-1],
                   edgecolor="#ffffff11", linewidth=0.5)
    ax.set_xlabel("Importance", color=TEXT_COLOR)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name} {title_suffix}",
                 color=TEXT_COLOR, fontsize=12, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(color=GRID_COLOR, axis="x", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_model_comparison(results: list, save_path=None):
    """Grouped bar chart comparing all models across all metrics."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    model_names = [r["model"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    colours = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    for i, (res, color) in enumerate(zip(results, colours)):
        vals = [res[m] for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        rects = ax.bar(x + offset, vals, width, label=res["model"],
                       color=color, alpha=0.85, edgecolor="#ffffff22")
        for rect, v in zip(rects, vals):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom",
                    color=TEXT_COLOR, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics], color=TEXT_COLOR)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", color=TEXT_COLOR)
    ax.set_title("Model Comparison — All Metrics", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#161b22", labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(color=GRID_COLOR, axis="y", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_risk_score_distribution(y_prob_legit, y_prob_fraud, model_name="Best Model", save_path=None):
    """KDE + histogram of predicted fraud probabilities split by true class."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.hist(y_prob_legit, bins=60, alpha=0.6, color=LEGIT_COLOR,
            density=True, label="Legitimate", edgecolor="none")
    ax.hist(y_prob_fraud, bins=60, alpha=0.6, color=FRAUD_COLOR,
            density=True, label="Fraud", edgecolor="none")
    ax.axvline(config.FRAUD_THRESHOLD, color="#f39c12", lw=2, linestyle="--",
               label=f"Threshold = {config.FRAUD_THRESHOLD}")
    ax.set_xlabel("Predicted Fraud Probability", color=TEXT_COLOR)
    ax.set_ylabel("Density", color=TEXT_COLOR)
    ax.set_title(f"Risk Score Distribution — {model_name}", color=TEXT_COLOR, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT_COLOR)
    ax.legend(facecolor="#161b22", labelcolor=TEXT_COLOR)
    ax.grid(color=GRID_COLOR, linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Saved plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# METADATA  (saved alongside the model)
# ─────────────────────────────────────────────

def save_metadata(metrics: dict, feature_names: list, path: str = config.METADATA_PATH):
    meta = {
        "trained_at":     datetime.utcnow().isoformat() + "Z",
        "model_name":     metrics["model"],
        "threshold":      metrics["threshold"],
        "feature_names":  feature_names,
        "metrics": {
            k: float(v) for k, v in metrics.items()
            if k not in ("y_prob", "y_pred", "model", "threshold")
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved → {path}")


def load_metadata(path: str = config.METADATA_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# INPUT VALIDATION
# ─────────────────────────────────────────────

REQUIRED_FIELDS = {
    "amount":              (float, int),
    "transaction_hour":    (float, int),
    "transaction_day":     (float, int),
    "location":            str,
    "transaction_type":    str,
    "transaction_freq_7d": (float, int),
    "avg_amount_7d":       (float, int),
    # Derived fields below are now OPTIONAL (calculated if missing)
    "amount_deviation":    (float, int, type(None)),
    "is_night":            (float, int, type(None)),
    "is_weekend":          (float, int, type(None)),
}

def validate_input(data: dict) -> dict:
    """
    Validate and coerce a raw prediction request dict.
    Returns cleaned dict or raises ValueError with details.
    """
    errors = []
    cleaned = {}

    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue
        val = data[field]
        if not isinstance(val, expected_types):
            try:
                val = expected_types[0](val) if isinstance(expected_types, tuple) else expected_types(val)
            except (ValueError, TypeError):
                errors.append(f"Field '{field}' must be {expected_types}, got {type(val).__name__}")
                continue
        cleaned[field] = val

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors))

    # 3. Apply feature engineering for missing derived fields
    df_temp = pd.DataFrame([cleaned])
    df_temp = feature_engineering(df_temp)
    cleaned = df_temp.iloc[0].to_dict()

    # Derived guard-rails
    if cleaned.get("amount", 0) < 0:
        raise ValueError("'amount' cannot be negative.")
    if not (0 <= cleaned.get("transaction_hour", 0) <= 23):
        raise ValueError("'transaction_hour' must be 0–23.")
    if not (0 <= cleaned.get("transaction_day", 0) <= 6):
        raise ValueError("'transaction_day' must be 0–6.")

    return cleaned
