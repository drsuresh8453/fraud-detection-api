"""
log_predictions.py — saves every scored payment (features + score + band) to S3.

Without these logs there is nothing to compare against the training data, so drift could never be detected.
Logs are buffered in memory and written in small files:
    s3://<bucket>/logs/predictions/date=2026-11-14/part-<time>-<id>.jsonl
"""
import json
import threading
import time
import uuid
from datetime import datetime, timezone

from src import config, storage


class PredictionLogger:
    def __init__(self, flush_every=None, flush_seconds=None, enabled=None):
        self.buf = []
        self.lock = threading.Lock()
        self.flush_every = flush_every or config.LOG_FLUSH_EVERY
        self.flush_seconds = flush_seconds or config.LOG_FLUSH_SECONDS
        self.enabled = config.LOG_PREDICTIONS if enabled is None else enabled
        self.last_flush = time.time()
        self.written = 0

    def log(self, result: dict, features: dict, request: dict):
        if not self.enabled:
            return
        rec = {"logged_at": datetime.now(timezone.utc).isoformat(), "txn_id": result["txn_id"],
               "txn_timestamp": str(request.get("txn_timestamp")), "customer_id": request.get("customer_id"),
               "amount": request.get("amount"), "channel": request.get("channel"),
               "fraud_score": result["fraud_score"], "risk_band": result["risk_band"],
               "model_version": result["model_version"], "latency_ms": result["latency_ms"],
               **{k: round(float(v), 6) for k, v in features.items()}}
        with self.lock:
            self.buf.append(rec)
            due = len(self.buf) >= self.flush_every or time.time() - self.last_flush > self.flush_seconds
        if due:
            self.flush()

    def flush(self):
        with self.lock:
            batch, self.buf = self.buf, []
            self.last_flush = time.time()
        if not batch:
            return 0
        day = batch[0]["logged_at"][:10]
        key = f"{config.PREDICTION_LOG_PREFIX}date={day}/part-{int(time.time())}-{uuid.uuid4().hex[:6]}.jsonl"
        body = "\n".join(json.dumps(r, default=str) for r in batch).encode()
        try:
            storage.write_bytes(key, body)
            self.written += len(batch)
        except Exception as e:      # never break scoring because logging failed
            print("⚠️ prediction log write failed:", e)
        return len(batch)
