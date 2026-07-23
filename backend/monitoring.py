"""
monitoring.py — lightweight production monitoring for the model service.

Production ML isn't "train once and forget" — you must watch the live service. This
module tracks the metrics that matter, which you'd normally export to Prometheus/Grafana:

    - request count & error count
    - latency (average + p95)
    - prediction distribution (to detect data/label drift)
"""

from __future__ import annotations

import time
from collections import Counter, deque
from threading import Lock


class Monitor:
    def __init__(self, window: int = 500) -> None:
        self._lock = Lock()
        self.requests = 0
        self.errors = 0
        self.latencies_ms: deque[float] = deque(maxlen=window)
        self.prediction_counts: Counter[str] = Counter()
        self.started_at = time.time()

    def record(self, latency_ms: float, prediction: str | None, error: bool) -> None:
        with self._lock:
            self.requests += 1
            if error:
                self.errors += 1
            else:
                self.latencies_ms.append(latency_ms)
                if prediction:
                    self.prediction_counts[prediction] += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies_ms)
            avg = sum(lat) / len(lat) if lat else 0.0
            p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
            return {
                "uptime_seconds": round(time.time() - self.started_at, 1),
                "total_requests": self.requests,
                "errors": self.errors,
                "error_rate": round(self.errors / self.requests, 4) if self.requests else 0.0,
                "avg_latency_ms": round(avg, 2),
                "p95_latency_ms": round(p95, 2),
                "prediction_distribution": dict(self.prediction_counts),
            }
