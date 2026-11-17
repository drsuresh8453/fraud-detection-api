"""
detect_drift.py — weekly drift check with Evidently AI (runs as a Kubernetes CronJob).

  REFERENCE = monitoring/reference_features.parquet   (training period: "what normal looks like")
  CURRENT   = logs/predictions/… of the last DRIFT_WINDOW_DAYS (what the API saw in production)

Checks
  A. Data drift        — PSI of every model feature (Evidently DataDriftPreset, method = PSI)
  B. Prediction drift  — PSI of the fraud score + alert rate (share of payments not APPROVED)
  C. Performance       — when mature labels exist (new_data/ exports): recall & fraud money caught at the live threshold
                         (Evidently ClassificationPreset)

Decision (limits in src/config.py)
  retrain if  any TOP-10 important feature has PSI > 0.25   OR  > 30 % of features drifted
          OR  alert rate > 5 %                              OR  recall on mature labels < 90 %

Outputs  reports/drift/<date>/data_drift.html, performance.html, summary.json   +  reports/drift/latest.json
Then     SNS e-mail alert  +  monitoring/trigger_retrain.py starts the SageMaker retraining notebook.

Usage    python -m monitoring.detect_drift                  # check + trigger SageMaker if needed
         python -m monitoring.detect_drift --no-trigger     # only report
         python -m monitoring.detect_drift --local-retrain  # retrain right here instead of SageMaker (laptop demo)
"""
import argparse
import io
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from src import config, storage


# ── load data ────────────────────────────────────────────────────────────────────────────
def load_current(days=config.DRIFT_WINDOW_DAYS) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    wanted = {f"date={today - timedelta(days=i)}" for i in range(days)}
    keys = [k for k in storage.list_keys(config.PREDICTION_LOG_PREFIX) if any(w in k for w in wanted)]
    if not keys:
        return pd.DataFrame()
    parts = [pd.read_json(io.BytesIO(storage.read_bytes(k)), lines=True) for k in keys]
    cur = pd.concat(parts, ignore_index=True).drop_duplicates("txn_id", keep="last")
    print(f"Current window: {len(cur):,} scored payments from {len(keys)} log file(s) (last {days} days)")
    return cur


def load_labels() -> pd.DataFrame:
    keys = [k for k in storage.list_keys(config.NEW_DATA_PREFIX) if k.endswith((".csv", ".parquet", ".xlsx"))]
    if not keys:
        return pd.DataFrame(columns=["txn_id", "is_fraud"])
    lab = pd.concat([storage.read_table(k)[["txn_id", "is_fraud", "label_status"]] for k in keys])
    lab = lab[lab["label_status"].isin(["NO_COMPLAINT_90D", "CONFIRMED_FRAUD", "COMPLAINT_REJECTED"])]
    return lab.drop_duplicates("txn_id", keep="last")[["txn_id", "is_fraud"]]


def feature_importance(features):
    model = storage.read_pickle(config.MODEL_KEY)
    imp = getattr(model, "feature_importances_", np.ones(len(features)))
    return pd.Series(imp, index=features).sort_values(ascending=False)


# ── checks ───────────────────────────────────────────────────────────────────────────────
def data_drift(ref: pd.DataFrame, cur: pd.DataFrame, columns: list):
    # Features with few distinct values (0/1 flags, one-hot, handle risk) are compared as CATEGORIES,
    # continuous ones (amounts, ages, seconds) in 10 quantile bins — both with PSI.
    ref, cur = ref[columns].astype(float).round(6), cur[columns].astype(float).round(6)   # 0.00378 == 0.0037800000000000004
    cat = [c for c in columns if ref[c].nunique() <= 15]
    num = [c for c in columns if c not in cat]
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    report = Report([DataDriftPreset(columns=columns, method="psi", threshold=config.PSI_LIMIT)])
    snap = report.run(current_data=Dataset.from_pandas(cur[columns], data_definition=dd),
                      reference_data=Dataset.from_pandas(ref[columns], data_definition=dd))
    psi = {}
    for m in snap.dict()["metrics"]:
        cfg = m.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            psi[cfg["column"]] = float(m["value"])
    return snap, pd.Series(psi).sort_values(ascending=False)


def performance(cur: pd.DataFrame, labels: pd.DataFrame, threshold: float):
    j = cur.merge(labels, on="txn_id", how="inner")
    if len(j) < 200 or j["is_fraud"].sum() < 10:
        return None, None
    j["is_fraud"] = j["is_fraud"].astype(int)
    j["flagged"] = (j["fraud_score"] >= threshold).astype(int)
    fr = j["is_fraud"] == 1
    amount = j["amount"].astype(float)
    perf = {"labelled_payments": int(len(j)), "frauds": int(fr.sum()),
            "recall": round(float(j.loc[fr, "flagged"].mean()), 4),
            "precision": round(float(j.loc[j["flagged"] == 1, "is_fraud"].mean()) if j["flagged"].any() else 0.0, 4),
            "money_caught": round(float(amount[fr & (j["flagged"] == 1)].sum() / amount[fr].sum()), 4)}
    dd = DataDefinition(classification=[BinaryClassification(target="is_fraud", prediction_labels="flagged")])
    snap = Report([ClassificationPreset()]).run(current_data=Dataset.from_pandas(j[["is_fraud", "flagged"]], data_definition=dd))
    return perf, snap


# ── main ─────────────────────────────────────────────────────────────────────────────────
def run(trigger=True, local_retrain=False, days=config.DRIFT_WINDOW_DAYS):
    cfg = storage.read_json(config.MODEL_CONFIG_KEY)
    features, threshold = cfg["features"], float(cfg["threshold"])
    cur = load_current(days)
    if "model_version" in cur and (cur["model_version"].astype(str) == str(cfg.get("model_version"))).sum() >= config.MIN_ROWS_FOR_DRIFT:
        cur = cur[cur["model_version"].astype(str) == str(cfg.get("model_version"))]      # only payments scored by the live model
        print(f"Using {len(cur):,} payments scored by the current model {cfg.get('model_version')}")
    if len(cur) < config.MIN_ROWS_FOR_DRIFT:
        print(f"Only {len(cur)} payments in the window (< {config.MIN_ROWS_FOR_DRIFT}) — not enough for a reliable check.")
        return {"status": "skipped", "rows": int(len(cur))}
    ref = storage.read_parquet(config.REFERENCE_KEY)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = f"{config.DRIFT_REPORT_PREFIX}{day}/"

    # A + B. data drift and prediction drift
    cols = [f for f in features if f in cur.columns and f in ref.columns] + ["fraud_score"]
    snap, psi = data_drift(ref, cur, cols)
    storage.write_bytes(out + "data_drift.html", _html(snap))
    imp = feature_importance(features)
    top = [f for f in imp.index[:config.TOP_FEATURES_FOR_DRIFT]]
    drifted = psi.drop("fraud_score", errors="ignore")
    drifted = drifted[drifted > config.PSI_LIMIT]
    share = len(drifted) / max(len(cols) - 1, 1)
    ref_alert = float((ref["fraud_score"] >= threshold).mean())
    cur_alert = float((cur["fraud_score"] >= threshold).mean())

    # C. performance on mature labels
    perf, perf_snap = performance(cur, load_labels(), threshold)
    if perf_snap is not None:
        storage.write_bytes(out + "performance.html", _html(perf_snap))

    reasons = []
    if any(f in top for f in drifted.index):
        reasons.append(f"important feature(s) drifted: {[f for f in drifted.index if f in top]}")
    if share > config.DRIFT_SHARE_LIMIT:
        reasons.append(f"{share:.0%} of features drifted (limit {config.DRIFT_SHARE_LIMIT:.0%})")
    if cur_alert > config.ALERT_RATE_LIMIT:
        reasons.append(f"alert rate {cur_alert:.1%} > {config.ALERT_RATE_LIMIT:.0%}")
    if perf and perf["recall"] < config.RECALL_LIMIT:
        reasons.append(f"recall on mature labels {perf['recall']:.1%} < {config.RECALL_LIMIT:.0%}")

    summary = {"date": day, "model_version": cfg.get("model_version"), "rows": int(len(cur)),
               "psi_top": {k: round(v, 3) for k, v in psi.head(10).items()},
               "drifted_features": {k: round(v, 3) for k, v in drifted.items()}, "drift_share": round(share, 3),
               "score_psi": round(float(psi.get("fraud_score", 0.0)), 3),
               "alert_rate": {"reference": round(ref_alert, 4), "current": round(cur_alert, 4)},
               "performance": perf, "retrain_needed": bool(reasons), "reasons": reasons,
               "reports": {"data_drift": out + "data_drift.html", "performance": out + "performance.html" if perf else None}}
    storage.write_json(summary, out + "summary.json")
    storage.write_json(summary, config.DRIFT_REPORT_PREFIX + "latest.json")
    _print(summary)

    if reasons:
        from monitoring.trigger_retrain import notify, start_retraining
        notify(summary)
        last = _last_retraining()
        if last is not None and (datetime.now(timezone.utc).replace(tzinfo=None) - last).days < config.RETRAIN_COOLDOWN_DAYS and not local_retrain:
            print(f"⏸️ Retraining ran on {last:%Y-%m-%d} (< {config.RETRAIN_COOLDOWN_DAYS} days ago) — alert sent, no new job started.")
            return summary
        if local_retrain:
            from src.train import retrain
            summary["retraining"] = retrain(promote=True, reason="; ".join(reasons))
        elif trigger:
            summary["retraining"] = start_retraining(out + "summary.json")
    return summary


def _last_retraining():
    keys = [k for k in storage.list_keys("reports/retraining/") if k.endswith("report.json")]
    if not keys:
        return None
    run_id = sorted(keys)[-1].split("/")[2]          # reports/retraining/<YYYYmmdd-HHMMSS>/report.json
    try:
        return datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _html(snap) -> bytes:
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "r.html")
        snap.save_html(p)
        return open(p, "rb").read()


def _print(s):
    print("\n══════════ DRIFT CHECK", s["date"], "— model", s["model_version"], "══════════")
    print(f"Payments checked: {s['rows']:,}")
    print("Highest PSI:", json.dumps(s["psi_top"]))
    print(f"Drifted features (PSI > {config.PSI_LIMIT}): {s['drifted_features'] or 'none'}  | share {s['drift_share']:.0%}")
    print(f"Score PSI: {s['score_psi']} | alert rate reference {s['alert_rate']['reference']:.2%} → current {s['alert_rate']['current']:.2%}")
    if s["performance"]:
        p = s["performance"]
        print(f"Performance on {p['labelled_payments']:,} labelled payments: recall {p['recall']:.1%}, "
              f"money caught {p['money_caught']:.1%}, precision {p['precision']:.1%}")
    print("DECISION:", "🚨 RETRAIN — " + " | ".join(s["reasons"]) if s["retrain_needed"] else "✅ no action needed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-trigger", action="store_true")
    ap.add_argument("--local-retrain", action="store_true")
    ap.add_argument("--days", type=int, default=config.DRIFT_WINDOW_DAYS)
    a = ap.parse_args()
    run(trigger=not a.no_trigger, local_retrain=a.local_retrain, days=a.days)
