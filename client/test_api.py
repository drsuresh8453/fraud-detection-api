"""
test_api.py — a Python program that CONSUMES the API, like the bank's payment system would.

  python client/test_api.py                          # local API
  python client/test_api.py --url http://<k8s-ip>    # API in Kubernetes
  python client/test_api.py --replay 1000            # also replay 1,000 test-period payments and measure recall

Part 1 sends the named example payments one by one and prints score, band, action and reasons.
Part 2 replays real test-period payments and compares the API's decisions with the true labels.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent


def show(name, resp):
    print(f"\n── {name} → HTTP {resp.status_code}")
    body = resp.json()
    if resp.status_code != 200:
        print("   ", json.dumps(body.get("detail", body))[:300])
        return
    print(f"   score {body['fraud_score']:.4f} | band {body['risk_band']} | {body['action']}")
    for r in body["reasons"]:
        print(f"     • {r}")
    print(f"   model {body['model_version']} | {body['latency_ms']} ms | warnings: {body['warnings']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--replay", type=int, default=0, help="how many payments to replay")
    ap.add_argument("--file", default=str(HERE / "sample_payments.csv"), help="CSV of payments to replay (e.g. client/drift_traffic.csv)")
    ap.add_argument("--skip-examples", action="store_true")
    a = ap.parse_args()
    url = a.url.rstrip("/")

    print("HEALTH:", requests.get(f"{url}/health", timeout=10).json())
    info = requests.get(f"{url}/model-info", timeout=10).json()
    print(f"MODEL : {info['model_name']} {info['model_version']} | threshold {info['threshold']} | {info['n_features']} features")

    if not a.skip_examples:
        samples = json.loads((HERE / "sample_requests.json").read_text())
        for name, body in samples.items():
            show(name, requests.post(f"{url}/score", json=body, timeout=10))

    if a.replay:
        df = pd.read_csv(a.file).head(a.replay)
        y = df.pop("is_fraud").values
        reqs = json.loads(df.to_json(orient="records"))
        t0, out = time.time(), []
        for i in range(0, len(reqs), 100):                     # batches of 100, like a bank's batch endpoint
            r = requests.post(f"{url}/score/batch", json={"payments": reqs[i:i + 100]}, timeout=60)
            r.raise_for_status()
            out += r.json()
        secs = time.time() - t0
        res = pd.DataFrame(out)
        res["is_fraud"] = y
        res["flagged"] = res["risk_band"] != "APPROVE"
        tp = int((res.flagged & (res.is_fraud == 1)).sum())
        print(f"\nREPLAY of {len(res):,} payments from {Path(a.file).name} in {secs:.1f}s ({len(res) / secs:.0f} payments/s)")
        print(f"  fraud caught      : {tp} of {int(res.is_fraud.sum())}  (recall {tp / max(res.is_fraud.sum(), 1):.1%})")
        print(f"  genuine disturbed : {int((res.flagged & (res.is_fraud == 0)).sum())} of {int((res.is_fraud == 0).sum())}")
        print("  bands:", res["risk_band"].value_counts().to_dict())


if __name__ == "__main__":
    main()
