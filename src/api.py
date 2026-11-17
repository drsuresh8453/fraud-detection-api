"""
api.py — the Fraud Scoring API (FastAPI).

Run locally:   uvicorn src.api:app --host 0.0.0.0 --port 8000
Docs:          http://localhost:8000/docs   (try every endpoint from the browser)

Endpoints
  GET  /health        → is the service alive and the model loaded? (used by Kubernetes probes)
  GET  /model-info    → model name, version, threshold, bands, test results
  POST /score         → score ONE payment  → score, band, action, top reasons
  POST /score/batch   → score up to 1,000 payments
  POST /admin/reload  → reload the model from S3 now (normally automatic every 5 minutes)
"""
import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from monitoring.log_predictions import PredictionLogger
from src import config
from src.predict import FraudModel
from src.validate import BatchRequest, PaymentRequest, ScoreResponse, business_rules

MODEL = FraudModel()
LOGGER = PredictionLogger()
STATS = {"started": time.time(), "scored": 0, "rejected": 0}


async def _reload_loop():
    while True:
        await asyncio.sleep(config.MODEL_RELOAD_SECONDS)
        await asyncio.to_thread(MODEL.check_for_new_model)
        await asyncio.to_thread(LOGGER.flush)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL.load()                                   # once at start-up: model, encoders, fill values, feature store
    task = asyncio.create_task(_reload_loop())
    yield
    task.cancel()
    LOGGER.flush()                                 # don't lose buffered logs on shutdown


app = FastAPI(title="Real-Time Fraud Detection API",
              description="Scores UPI / card / net-banking payments. Created by Suresh D R | AI Product Developer & Technology Mentor",
              version="1.0.0", lifespan=lifespan)


def _score(reqs):
    dicts = []
    for r in reqs:
        errors, warnings = business_rules(r)
        if errors:
            STATS["rejected"] += 1
            raise HTTPException(status_code=422, detail={"txn_id": r.txn_id, "errors": errors})
        d = r.model_dump()
        d["txn_timestamp"] = r.txn_timestamp.replace(tzinfo=None).isoformat()
        d["_warnings"] = warnings
        dicts.append(d)
    results, X = MODEL.score(dicts)
    for i, (res, d) in enumerate(zip(results, dicts)):
        res["warnings"] = d["_warnings"] + res["warnings"]
        LOGGER.log(res, X.iloc[i].to_dict(), d)
    STATS["scored"] += len(results)
    return results


@app.get("/health")
def health():
    ok = MODEL.version is not None and MODEL.store.loaded
    body = {"status": "ok" if ok else "loading", "model_version": MODEL.version,
            "uptime_seconds": round(time.time() - STATS["started"]), "scored": STATS["scored"], "rejected": STATS["rejected"]}
    return JSONResponse(body, status_code=200 if ok else 503)


@app.get("/model-info")
def model_info():
    cfg = MODEL.cfg
    return {"model_name": cfg.get("model_name"), "model_version": MODEL.version, "threshold": MODEL.threshold,
            "bands": MODEL.bands, "n_features": len(MODEL.features), "trained_until": cfg.get("trained_until"),
            "test_results": cfg.get("test"), "model_card": cfg.get("model_card")}


@app.post("/score", response_model=ScoreResponse)
def score(payment: PaymentRequest):
    return _score([payment])[0]


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(batch: BatchRequest):
    return _score(batch.payments)


@app.post("/admin/reload")
def reload_model():
    """Reload model + feature store from storage now (normally automatic every MODEL_RELOAD_SECONDS)."""
    changed = MODEL.check_for_new_model(force=True)
    LOGGER.flush()
    return {"reloaded": changed, "model_version": MODEL.version}
