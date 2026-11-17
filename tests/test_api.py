import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api import app

SAMPLES = json.loads((Path(__file__).parent.parent / "client" / "sample_requests.json").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # runs start-up: loads model + feature store
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["n_features"] == 71 and 0 < body["threshold"] < 1


def test_genuine_payment_is_approved(client):
    body = client.post("/score", json=SAMPLES["1_genuine_upi_payment"]).json()
    assert body["risk_band"] == "APPROVE"


@pytest.mark.parametrize("name", ["2_account_takeover", "4_sim_swap"])
def test_obvious_fraud_is_stopped(client, name):
    body = client.post("/score", json=SAMPLES[name]).json()
    assert body["risk_band"] in ("REVIEW", "BLOCK") and body["reasons"]


def test_invalid_request_rejected(client):
    assert client.post("/score", json=SAMPLES["6_invalid_request"]).status_code == 422


def test_batch(client):
    body = {"payments": [SAMPLES["1_genuine_upi_payment"], SAMPLES["2_account_takeover"]]}
    out = client.post("/score/batch", json=body).json()
    assert [o["txn_id"] for o in out] == [p["txn_id"] for p in body["payments"]]


def test_latency_within_limit(client):
    """CI gate: a model that scores slower than LATENCY_LIMIT_MS is not deployed."""
    req = SAMPLES["3_vishing_kyc_scam"]
    for _ in range(5):
        client.post("/score", json=req)
    t0 = time.perf_counter()
    for _ in range(30):
        client.post("/score", json=req)
    ms = (time.perf_counter() - t0) / 30 * 1000
    assert ms < config.LATENCY_LIMIT_MS * 2, f"{ms:.1f} ms per request"
