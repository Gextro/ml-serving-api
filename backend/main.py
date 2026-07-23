"""
main.py — FastAPI service that serves the ML model + a monitoring dashboard.

Endpoints:
    GET  /health          -> liveness + current model version
    POST /predict         -> classify a flower from 4 features
    POST /retrain         -> retrain the model (returns new version + accuracy)
    GET  /metrics         -> live monitoring snapshot (latency, drift, errors)
    GET  /                -> the dashboard UI
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import IrisModel, FEATURE_NAMES
from monitoring import Monitor

app = FastAPI(title="ML Model Serving API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

model = IrisModel()
model.ensure_ready()  # load existing model or train one on startup
monitor = Monitor()

FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class PredictRequest(BaseModel):
    # Four flower measurements in centimetres.
    features: list[float] = Field(..., min_length=4, max_length=4,
                                  examples=[[5.1, 3.5, 1.4, 0.2]])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": model.version, "accuracy": model.accuracy}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    start = time.perf_counter()
    prediction = None
    error = False
    try:
        result = model.predict(req.features)
        prediction = result["prediction"]
        return result
    except ValueError as exc:
        error = True
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        error = True
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        monitor.record(latency_ms, prediction, error)


@app.post("/retrain")
def retrain() -> dict:
    result = model.train()
    return {
        "version": result.version,
        "accuracy": result.accuracy,
        "n_samples": result.n_samples,
    }


@app.get("/metrics")
def metrics() -> dict:
    snap = monitor.snapshot()
    snap["model_version"] = model.version
    snap["model_accuracy"] = model.accuracy
    snap["feature_names"] = FEATURE_NAMES
    return snap


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND)
