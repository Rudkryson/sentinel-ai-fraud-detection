---
title: Sentinel AI Fraud Detection
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# 🛡️ Sentinel AI — Advanced Fraud Detection System

Sentinel AI is a premium, enterprise-grade fraud detection and risk scoring platform. It leverages machine learning to identify fraudulent transactions in real-time, providing deep behavioral insights and predictive analytics through a modern executive dashboard.

---

## 🚀 Quick Start

The fastest way to get Sentinel AI running is using the automated runner script:

```bash
# 1. Setup & Run (Automated)
python run.py
```

*This single command will install dependencies, setup environment variables, train the model (if missing), run database migrations, and start the server.*

---

## 🛠️ Manual Setup

If you prefer to run steps manually:

### 1. Prerequisites
- **Python 3.9+**
- **Virtual Environment** (recommended)

### 2. Installation
```bash
pip install -r requirements.txt
cp .env.example .env
```

### 3. Initialization & Run
```bash
python train.py               # Train models
python -m alembic upgrade head # Setup database
python -m uvicorn app.main:app # Start server
```

- **Frontend Dashboard**: [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Landing Page**: [http://localhost:8000/](http://localhost:8000/)

---

## 🏗️ Project Architecture

```text
fraud_detection/
├── app/                 # FastAPI Backend Application
│   ├── api/             # API Routes (Endpoints)
│   ├── core/            # Configuration & Security
│   ├── models/          # Database Models (SQLAlchemy)
│   ├── schemas/         # Pydantic Schemas
│   └── services/        # Business Logic & ML Inference
├── frontend/            # Vanilla JS/CSS/HTML Frontend
├── alembic/             # Database Migration Scripts
├── data/                # Synthetic Datasets
├── model/               # Serialised ML Models (.joblib)
├── pipeline/            # Scikit-learn Preprocessing Pipelines
├── visualizations/      # Generated Performance Metrics & Plots
├── train.py             # Model Training Orchestrator
├── requirements.txt     # Python Dependencies
└── .env.example         # Environment Variable Template
```

---

## 🧠 Machine Learning Engine

Sentinel AI uses an ensemble approach for maximum recall:

| Model              | Recall | ROC-AUC | Strategy |
|--------------------|--------|---------|----------|
| **Random Forest**  | 1.000  | 1.000   | Best Overall |
| **Isolation Forest**| 0.925  | 0.967   | Anomaly Detection |
| **SMOTE**          | -      | -       | Synthetic Oversampling |

### Key Features:
- **Real-time Scoring**: Instant risk assessment for incoming transactions.
- **Behavioral Analysis**: Tracks deviations from user spending patterns.
- **Explainable AI**: Provides feature importance and risk level justification.
- **Automated Pipeline**: End-to-end training, tuning, and evaluation.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Alembic, Pydantic
- **AI/ML**: Scikit-Learn, Pandas, NumPy, Joblib, Imbalanced-Learn (SMOTE)
- **Frontend**: Vanilla HTML5, CSS3 (Modern Glassmorphism), JavaScript (ES6+)
- **Database**: SQLite (Production-ready abstraction via SQLAlchemy)
- **DevOps**: Docker ready, Logging integrated

---

## 🤝 Contribution & Support

For enterprise support or integration queries, contact the **Rudkryson Tech** engineering team.

---
*Created by Antigravity AI for Rudkryson Tech.*
