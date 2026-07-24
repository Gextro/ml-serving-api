# ML Model Serving API + Monitoring Dashboard

[![CI](https://github.com/Gextro/ml-serving-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Gextro/ml-serving-api/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-8B5CF6.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

**🔗 Live demo:** https://ml-serving-api.onrender.com

A production-style **MLOps** project: train a machine-learning model, serve it behind a REST
API, and monitor it live (latency, error rate, prediction drift) with a real-time dashboard.
This is the skill that separates "did a Kaggle notebook" from "can ship ML to production" —
and it's among the highest-paid engineering skills.

> Runs instantly — the model trains itself on first startup using the built-in Iris dataset
> (no downloads, no GPU).

---

## Architecture

```
   Dashboard ──POST /predict──►  FastAPI  ──►  IrisModel (train/persist/predict)
   (frontend) ◄── prediction ──  (main.py)     │
              ──GET /metrics──►      │          ▼
              ◄── live stats ──   Monitor (latency, errors, drift)
```

- **Model lifecycle** ([backend/model.py](backend/model.py)) — train → evaluate → version → persist → load → predict.
- **Monitoring** ([backend/monitoring.py](backend/monitoring.py)) — request count, latency avg/p95, prediction distribution.
- **API** ([backend/main.py](backend/main.py)) — `/predict`, `/retrain`, `/metrics`, `/health`.

---

## Tech stack
- **ML:** scikit-learn (RandomForest), joblib for model persistence
- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Monitoring:** custom metrics (Prometheus-style), auto-refreshing dashboard
- **Infra:** Docker, Render blueprint

---

## Run locally

> Requires **Python 3.11–3.13** (scikit-learn wheels are not yet published for 3.14).

```bash
cd backend
py -3.13 -m venv .venv        # Windows: pick an installed 3.11–3.13
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000 — make predictions, retrain the model, watch metrics update live.

Test the API directly:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"features\":[5.1,3.5,1.4,0.2]}"
```

---

## Deploy (free)
Push to GitHub → on [Render](https://render.com): **New → Blueprint → select repo** (reads `render.yaml`).

---

## What to study next
See [`LEARN.md`](LEARN.md) — the full MLOps lifecycle explained, plus **interview Q&A** on model
serving, monitoring, and drift.
