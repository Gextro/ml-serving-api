# LEARN.md — ML Model Serving: Your Complete Study Guide

Training a model is 10% of the job; **deploying and operating it** is the other 90%. This guide
teaches **MLOps** — the highest-leverage, highest-paid ML skill — using your own project.

---

## Table of contents
1. [The problem: notebook ≠ product](#1-the-problem-notebook--product)
2. [How the system works](#2-how-the-system-works)
3. [Skills this project teaches](#3-skills-this-project-teaches)
4. [Skill tutorials (from scratch)](#4-skill-tutorials-from-scratch)
5. [Monitoring & model drift](#5-monitoring--model-drift)
6. [Interview questions & model answers](#6-interview-questions--model-answers)
7. [60-second pitch](#7-60-second-pitch)
8. [Resume bullets](#8-resume-bullets)

---

## 1. The problem: notebook ≠ product
A model in a Jupyter notebook helps no one. To create value it must be:
- **served** behind an API other systems can call,
- **versioned** so you know exactly which model made a prediction,
- **monitored** so you catch failures and performance decay,
- **retrainable** without redeploying everything.

That end-to-end discipline is **MLOps**.

---

## 2. How the system works

| Stage | File | What happens |
|---|---|---|
| Train | `model.py` → `train()` | Fit a RandomForest on Iris, evaluate accuracy on a held-out test set. |
| Version | `model.py` | Stamp a version (`vYYYYMMDD-HHMMSS`) so predictions are traceable. |
| Persist | `model.py` → `save()`/`load()` | Serialize with joblib so restarts reuse the trained model. |
| Serve | `main.py` → `/predict` | Validate input, run inference, return class + probabilities. |
| Monitor | `monitoring.py` | Record latency, errors, and prediction distribution per request. |
| Retrain | `main.py` → `/retrain` | Produce a new versioned model on demand. |

---

## 3. Skills this project teaches

**Machine learning**
- Train/test split & why you evaluate on unseen data
- Classification, probabilities, confidence
- RandomForest basics
- Model persistence (serialization)

**MLOps / production ML**
- Model versioning & reproducibility (fixed random seed)
- Serving models behind a REST API
- Monitoring: latency (avg/p95), error rate, prediction drift
- Retraining workflows

**Backend engineering**
- FastAPI + Pydantic validation
- Thread-safe metrics (locking)
- Docker packaging & health checks

---

## 4. Skill tutorials (from scratch)

### 4.1 Train/test split
We train on 80% of the data and evaluate on the held-out 20%. Testing on unseen data is the
only honest measure of how the model will perform in production. Reporting accuracy on training
data is a classic beginner mistake (it hides overfitting).

### 4.2 What a RandomForest is (in one paragraph)
It's an ensemble of many decision trees, each trained on a random subset of data/features. Each
tree votes; the majority wins. Averaging many weak, decorrelated trees yields a strong, robust
model that resists overfitting — which is why it's a great default classifier.

### 4.3 Probabilities & confidence
`predict_proba` returns a probability for each class that sums to 1. We return the top class and
its probability as "confidence". Exposing confidence lets downstream systems set thresholds
(e.g. "route low-confidence predictions to a human").

### 4.4 Model versioning & reproducibility
Every trained model gets a version stamp, and we fix `random_state=42` so training is
reproducible. In an incident you must know *which* model served a bad prediction — versioning
makes that possible.

### 4.5 Serialization (why joblib)
A trained model lives in memory. `joblib.dump/load` writes it to disk so a server restart
doesn't lose it and you don't retrain on every boot. joblib is preferred over pickle for large
NumPy arrays.

### 4.6 Percentile latency (p95)
Averages hide pain. If 95% of requests are fast but 5% take 2s, users feel that 5%. **p95** = the
latency 95% of requests are under. Production SLAs are written in percentiles (p95/p99), not
averages. See `Monitor.snapshot()`.

### 4.7 Thread safety
A web server handles many requests at once. The metrics counters are guarded by a `Lock` so
concurrent updates don't corrupt them — a real concurrency concern in serving code.

---

## 5. Monitoring & model drift

A deployed model silently decays as the real world changes ("drift"):
- **Data drift** — the input distribution shifts (e.g. new user behavior).
- **Concept drift** — the relationship between inputs and the correct label changes.

You can't see drift from accuracy alone (you often lack live labels), so we track the
**prediction distribution**. If a model that used to predict 3 classes evenly suddenly predicts
one class 95% of the time, something changed — investigate and possibly **retrain**. That's why
`/metrics` exposes `prediction_distribution` and we provide a `/retrain` endpoint.

---

## 6. Interview questions & model answers

**Q1. How do you deploy an ML model to production?**
> Wrap it in a service (FastAPI), validate inputs, load a versioned/persisted model, expose a
> `/predict` endpoint, containerize it, and add monitoring + a retrain path. That's this project.

**Q2. Why evaluate on a test set?**
> To estimate real-world performance on unseen data and detect overfitting. Training accuracy is
> optimistic and misleading.

**Q3. What is model drift and how do you detect it?**
> Drift is degradation as real-world data diverges from training data. Detect it by monitoring the
> input and prediction distributions (and accuracy when labels are available). We track prediction
> distribution in `/metrics`.

**Q4. Why report p95 latency, not average?**
> Averages hide tail latency. p95/p99 capture the slow requests users actually feel and are the
> basis of SLAs.

**Q5. How do you version models and why?**
> Each trained model gets a unique version stamp and is persisted. Versioning gives traceability
> ("which model made this prediction?") and safe rollback.

**Q6. How do you make training reproducible?**
> Fix random seeds, pin data and library versions, and log hyperparameters. We set
> `random_state=42`.

**Q7. Pickle vs joblib?**
> Both serialize Python objects; joblib is more efficient for large NumPy arrays (sklearn models),
> so it's preferred for ML persistence.

**Q8. How would you scale this service?**
> Run multiple stateless replicas behind a load balancer, move the model to a shared store/model
> registry, add caching for repeated inputs, and use async/batch inference. For heavy models, use a
> GPU inference server (e.g. Triton) or ONNX runtime.

**Q9. How do you handle bad input?**
> Validate at the boundary with Pydantic + explicit checks (feature count/types) and return 4xx
> errors — never let malformed input reach the model. See `/predict`.

**Q10. What's your retraining strategy?**
> Retrain on a schedule or when drift/accuracy triggers fire, evaluate the candidate, and promote
> it only if it beats the current model (champion/challenger). The `/retrain` endpoint is the hook.

**Q11. RandomForest — why does it work well?**
> It averages many decorrelated decision trees (bagging), reducing variance and overfitting while
> needing little tuning — a strong baseline.

**Q12. What would you add for real production?**
> A model registry (MLflow), Prometheus/Grafana dashboards, A/B or shadow deployments,
> authentication, input/feature logging for audits, and automated retraining pipelines.

---

## 7. 60-second pitch

> "I built an ML serving platform that owns the full lifecycle: it trains a classifier, evaluates
> it on a held-out set, versions and persists it, and serves predictions with confidence scores
> over a FastAPI endpoint. Crucially, it's monitored — I track request latency at p95, error rate,
> and the live prediction distribution to detect drift — and it exposes a retrain endpoint to
> produce a new versioned model on demand. It's containerized and deploys with a Render blueprint.
> This is the MLOps discipline that turns a model into a reliable product."

---

## 8. Resume bullets

- Built an end-to-end **ML model serving platform** (FastAPI, scikit-learn) covering the full MLOps
  lifecycle: train → evaluate → **version** → persist → serve, with confidence-scored predictions.
- Implemented **production monitoring** (p95 latency, error rate, prediction-distribution drift
  detection) and an on-demand **retraining** endpoint, surfaced in a live auto-refreshing dashboard.
- **Containerized with Docker** and deployed via a health-checked Render blueprint; validated inputs
  with Pydantic and made metrics **thread-safe** for concurrent serving.
