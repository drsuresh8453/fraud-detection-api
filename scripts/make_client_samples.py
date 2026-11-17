"""
make_client_samples.py — creates test requests for Postman and the Python client from the TEST period.

Writes:
  client/sample_requests.json     ← named example payments (genuine, account takeover, scam, new live payment, invalid)
  client/sample_payments.csv      ← 1,000 test-period payments (what the bank would send) + true label, for replay
  client/postman_collection.json  ← import into Postman
Usage: python -m scripts.make_client_samples
"""
import json

import numpy as np
import pandas as pd

from src import storage
from src.predict import SENT_FIELDS


def to_req(row):
    r = {k: (None if pd.isna(row[k]) else row[k]) for k in SENT_FIELDS}
    r["txn_timestamp"] = pd.Timestamp(r["txn_timestamp"]).isoformat()
    for k in ["is_international", "is_rooted_or_emulator", "vpn_or_proxy"]:
        r[k] = int(r[k] or 0)
    r["amount"] = float(r["amount"])
    return r


def main():
    clean = storage.read_csv("data/02_cleaned.csv", low_memory=False)
    feats = storage.read_csv("data/06_features.csv", low_memory=False, usecols=["txn_id", "split"])
    d = clean.merge(feats, on="txn_id")      # fraud_type is used only to PICK examples, never sent to the API
    test = d[d["split"] == "test"]
    rng = np.random.default_rng(42)

    def pick(mask):
        rows = test[mask]
        return rows.iloc[int(rng.integers(len(rows)))]

    samples = {
        "1_genuine_upi_payment": to_req(pick((test["is_fraud"] == 0) & (test["channel"] == "UPI") & (test["amount"] < 3000))),
        "2_account_takeover": to_req(pick((test["fraud_type"] == "ACCOUNT_TAKEOVER") & (test["is_new_device"] == 1))),
        "3_vishing_kyc_scam": to_req(pick(test["fraud_type"] == "VISHING_KYC_CALL")),
        "4_sim_swap": to_req(pick(test["fraud_type"] == "SIM_SWAP")),
    }
    live = dict(samples["1_genuine_upi_payment"])                 # a brand-new payment (not in history)
    live.update({"txn_id": "TXN99000001", "txn_timestamp": "2026-11-14T02:40:00", "amount": 49999.0,
                 "payee_id": "quickcash642@paytm", "device_id": "D-NEW-PHONE-001", "device_os": "Android",
                 "ip_country": "NG", "remarks": "urgent kyc verification", "login_to_pay_seconds": 6.0,
                 "vpn_or_proxy": 1, "password_reset_last_24h": 1, "sim_changed_last_7d": 1})
    samples["5_new_live_payment_takeover_signals"] = live
    bad = dict(samples["1_genuine_upi_payment"])
    bad.update({"txn_id": "TXN99000002", "amount": 0, "channel": "BITCOIN"})
    samples["6_invalid_request"] = bad
    with open("client/sample_requests.json", "w") as f:
        json.dump(samples, f, indent=2)

    replay = pd.concat([test[test["is_fraud"] == 1], test[test["is_fraud"] == 0].sample(1000 - int(test["is_fraud"].sum()), random_state=1)])
    replay = replay.sort_values("txn_timestamp")
    replay[SENT_FIELDS + ["is_fraud"]].to_csv("client/sample_payments.csv", index=False)
    write_postman(samples)
    print(f"✅ client/sample_requests.json ({len(samples)}), client/sample_payments.csv ({len(replay)} rows), client/postman_collection.json")


def write_postman(samples):
    def item(name, method, path, body=None, tests=None):
        it = {"name": name, "request": {"method": method, "header": [{"key": "Content-Type", "value": "application/json"}],
                                        "url": {"raw": "{{baseUrl}}" + path, "host": ["{{baseUrl}}"], "path": path.strip("/").split("/")}}}
        if body is not None:
            it["request"]["body"] = {"mode": "raw", "raw": json.dumps(body, indent=2), "options": {"raw": {"language": "json"}}}
        if tests:
            it["event"] = [{"listen": "test", "script": {"type": "text/javascript", "exec": tests}}]
        return it

    ok = ["pm.test('status 200', () => pm.response.to.have.status(200));",
          "const r = pm.response.json();",
          "pm.test('has score and band', () => { pm.expect(r.fraud_score).to.be.a('number'); pm.expect(r.risk_band).to.be.a('string'); });",
          "console.log(r.txn_id, r.fraud_score, r.risk_band, r.reasons);"]
    items = [item("Health check", "GET", "/health", tests=["pm.test('status 200', () => pm.response.to.have.status(200));"]),
             item("Model info", "GET", "/model-info")]
    for name, body in samples.items():
        tests = ["pm.test('status 422', () => pm.response.to.have.status(422));"] if "invalid" in name else ok
        items.append(item(f"Score — {name}", "POST", "/score", body, tests))
    items.append(item("Score — batch of 3", "POST", "/score/batch",
                      {"payments": [samples["1_genuine_upi_payment"], samples["2_account_takeover"], samples["3_vishing_kyc_scam"]]}))
    col = {"info": {"name": "Fraud Detection API", "description": "Created by Suresh D R | AI Product Developer & Technology Mentor",
                    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
           "variable": [{"key": "baseUrl", "value": "http://localhost:8000"}], "item": items}
    with open("client/postman_collection.json", "w") as f:
        json.dump(col, f, indent=2)


if __name__ == "__main__":
    main()
