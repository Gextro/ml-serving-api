"""
model.py — trains, persists, and serves a machine-learning model.

We use the classic Iris flower dataset (ships with scikit-learn, no download) and a
RandomForest classifier. The point isn't the dataset — it's demonstrating the full
**MLOps** lifecycle that production ML engineers own:

    train -> evaluate -> version -> persist -> load -> serve predictions

This "model as a service" pattern is exactly how companies deploy ML behind an API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


@dataclass
class TrainResult:
    version: str
    accuracy: float
    trained_at: float
    n_samples: int


class IrisModel:
    """Wraps a scikit-learn classifier with train / save / load / predict."""

    def __init__(self) -> None:
        self.clf: RandomForestClassifier | None = None
        self.version: str = "untrained"
        self.accuracy: float = 0.0

    # --- training --------------------------------------------------------
    def train(self, n_estimators: int = 100, seed: int = 42) -> TrainResult:
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        clf.fit(X_train, y_train)

        acc = float(accuracy_score(y_test, clf.predict(X_test)))
        self.clf = clf
        self.accuracy = acc
        # A simple, human-readable version stamp (time-based).
        self.version = time.strftime("v%Y%m%d-%H%M%S")
        self.save()
        return TrainResult(
            version=self.version,
            accuracy=acc,
            trained_at=time.time(),
            n_samples=len(X_train),
        )

    # --- persistence -----------------------------------------------------
    def save(self) -> None:
        joblib.dump(
            {"clf": self.clf, "version": self.version, "accuracy": self.accuracy},
            MODEL_PATH,
        )

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            return False
        data = joblib.load(MODEL_PATH)
        self.clf = data["clf"]
        self.version = data["version"]
        self.accuracy = data["accuracy"]
        return True

    def ensure_ready(self) -> None:
        """Load an existing model, or train a fresh one on first run."""
        if self.clf is None and not self.load():
            self.train()

    # --- inference -------------------------------------------------------
    def predict(self, features: List[float]) -> dict:
        if self.clf is None:
            raise RuntimeError("Model is not trained.")
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features: {FEATURE_NAMES}")

        x = np.array(features, dtype=float).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return {
            "prediction": CLASS_NAMES[idx],
            "confidence": round(float(proba[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)},
            "model_version": self.version,
        }
