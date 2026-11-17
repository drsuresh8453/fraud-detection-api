"""
train.py — automatic retraining with CHAMPION vs CHALLENGER (Notebooks 2, 3, 6, 7 as one pipeline).

Called by the SageMaker retraining notebook (sagemaker/retrain_pipeline.ipynb) when drift is detected,
or by hand:  python -m src.train --promote

 1. Load history (data/history_cleaned.parquet or data/02_cleaned.csv) + every new export in new_data/
 2. Clean new exports (preprocess.clean_batch) and run the data-quality gate  → STOP if labels look broken
 3. Time split on MATURE labels only: train | valid (VALID_MONTHS) | test (TEST_MONTHS)
 4. Fill values + encoders learned on TRAIN only → features (features.build_features — same code as the API)
 5. Train LightGBM (Notebook 7 settings) → threshold by cost with the 3 % alert cap
 6. Compare with the CURRENT model on the same latest months
 7. Promote only if the challenger is better → new version in S3 (API pods reload automatically)
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

from src import config, storage
from src.feature_store import build_store_tables, save_store_tables
from src.features import add_basic_features, build_features, fit_encoders
from src.preprocess import clean_batch, data_quality_gate, fill_missing, fit_fill_values

FINAL_LABELS = ["NO_COMPLAINT_90D", "CONFIRMED_FRAUD", "COMPLAINT_REJECTED"]
LGBM_PARAMS = dict(n_estimators=400, num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=42, verbose=-1)


# ── 1–2. data ────────────────────────────────────────────────────────────────────────────
def load_history() -> pd.DataFrame:
    if storage.exists("data/history_cleaned.parquet"):
        base = storage.read_parquet("data/history_cleaned.parquet")
        src = "data/history_cleaned.parquet"
    else:
        base = storage.read_csv("data/02_cleaned.csv", low_memory=False)
        src = "data/02_cleaned.csv"
    base["txn_timestamp"] = pd.to_datetime(base["txn_timestamp"])
    base["account_open_date"] = pd.to_datetime(base["account_open_date"], errors="coerce")
    new_keys = [k for k in storage.list_keys(config.NEW_DATA_PREFIX) if k.endswith((".csv", ".xlsx", ".parquet"))]
    parts = [base]
    for k in new_keys:
        parts.append(clean_batch(storage.read_table(k), verbose=True))
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates("txn_id", keep="last").sort_values("txn_timestamp").reset_index(drop=True)   # newer labels win
    print(f"History: {src} + {len(new_keys)} new export(s) → {len(df):,} payments "
          f"({df['txn_timestamp'].min():%Y-%m-%d} → {df['txn_timestamp'].max():%Y-%m-%d})")
    return df


def time_split(df: pd.DataFrame, valid_months=config.VALID_MONTHS, test_months=config.TEST_MONTHS) -> pd.DataFrame:
    """Train on the past, test on the newest MATURE months (Notebook 6, Step 4)."""
    last = df.loc[df["label_status"] == "NO_COMPLAINT_90D", "txn_timestamp"].max()
    month = last.to_period("M") if last.day >= 25 else last.to_period("M") - 1        # last COMPLETE mature month
    end_of = lambda p: p.to_timestamp(how="end").floor("s")                          # 2026-07 → 2026-07-31 23:59:59
    mature_end, valid_end, train_end = end_of(month), end_of(month - test_months), end_of(month - test_months - valid_months)
    d = df.copy()
    d["split"] = np.select([d["txn_timestamp"] <= train_end, d["txn_timestamp"] <= valid_end, d["txn_timestamp"] <= mature_end],
                           ["train", "valid", "test"], default="not_mature")
    d.attrs.update(train_end=str(train_end), valid_end=str(valid_end), mature_end=str(mature_end))
    return d


# ── 4. features ──────────────────────────────────────────────────────────────────────────
def make_xy(rows: pd.DataFrame, fill: dict, enc: dict, features: list):
    X = build_features(fill_missing(rows, fill), enc, features)
    return X, rows["is_fraud"].astype(int).to_numpy(), rows["amount"].to_numpy(dtype=float)


# ── 5. threshold + metrics (Notebook 7) ──────────────────────────────────────────────────
def choose_threshold(p, y, amt, cost=config.FALSE_ALARM_COST, cap=config.ALERT_CAP):
    ths = np.round(np.concatenate([np.arange(0.001, 0.02, 0.001), np.linspace(0.02, 0.98, 97)]), 3)
    costs = np.array([amt[(p < t) & (y == 1)].sum() + ((p >= t) & (y == 0)).sum() * cost for t in ths])
    alerts = np.array([(p >= t).mean() for t in ths])
    ok = alerts <= cap
    return float(ths[ok][np.argmin(costs[ok])]) if ok.any() else float(ths[np.argmin(costs)])


def metrics(p, y, amt, threshold, alert_budget=0.03):
    flag = p >= threshold
    k = max(int(len(p) * alert_budget), 1)
    top = np.zeros(len(p), bool)
    top[np.argsort(-p, kind="stable")[:k]] = True
    fr = y == 1
    return {"pr_auc": round(float(average_precision_score(y, p)), 4),
            "recall": round(float(flag[fr].mean()), 4), "precision": round(float(y[flag].mean()) if flag.any() else 0.0, 4),
            "alert_rate": round(float(flag.mean()), 4), "money_caught": round(float(amt[flag & fr].sum() / max(amt[fr].sum(), 1)), 4),
            "money_caught_at3": round(float(amt[top & fr].sum() / max(amt[fr].sum(), 1)), 4),
            "frauds": int(fr.sum()), "payments": int(len(y))}


def next_version(v: str) -> str:
    try:
        major, minor = v.lstrip("v").split(".")
        return f"v{major}.{int(minor) + 1}"
    except Exception:
        return f"v{datetime.now(timezone.utc):%Y%m%d%H%M}"


# ── main pipeline ────────────────────────────────────────────────────────────────────────
def retrain(promote=True, valid_months=config.VALID_MONTHS, test_months=config.TEST_MONTHS, reason="manual", force=False):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    hist = load_history()

    gate = data_quality_gate(hist)
    print(gate.to_string(index=False))
    if (gate["status"] == "FAILED").any():
        raise RuntimeError("Data-quality gate FAILED — training stopped (fix the export, then retrain).")

    d = time_split(hist[hist["label_status"].isin(FINAL_LABELS) & hist["is_fraud"].notna()], valid_months, test_months)
    print(f"Split: train ≤ {d.attrs['train_end'][:10]} | valid ≤ {d.attrs['valid_end'][:10]} | test ≤ {d.attrs['mature_end'][:10]}")
    print(d.groupby("split")["is_fraud"].agg(payments="size", frauds="sum").reindex(["train", "valid", "test"]).to_string())
    tr, va, te = (d[d["split"] == s] for s in ("train", "valid", "test"))

    # current production model = champion
    champ_cfg = storage.read_json(config.MODEL_CONFIG_KEY)
    champ_enc, champ_fill = storage.read_pickle(config.ENCODERS_KEY), storage.read_json(config.IMPUTE_KEY)
    try:
        champ_model = storage.read_pickle(config.MODEL_KEY)
    except Exception as e:      # e.g. model trained in Colab with other library versions, or an XGBoost/RF model
        print(f"⚠️ Current model cannot be loaded here ({type(e).__name__}: {e}) — the new model will replace it.")
        champ_model = None
    features = champ_cfg["features"]               # feature SELECTION is fixed (Notebook 6); retraining re-learns weights

    # learn fill values + encoders on TRAIN only
    fill = fit_fill_values(tr)
    tr_basic = add_basic_features(fill_missing(tr, fill), champ_enc)
    all_basic = add_basic_features(fill_missing(d, fill), champ_enc)
    enc = fit_encoders(tr_basic, all_basic)
    for k in ("onehot_categories",):                   # never lose a category the API already knows
        for col, vals in champ_enc[k].items():
            enc[k][col] = sorted(set(enc[k][col]) | set(vals))

    Xtr, ytr, _ = make_xy(tr, fill, enc, features)
    Xva, yva, ava = make_xy(va, fill, enc, features)
    Xte, yte, ate = make_xy(te, fill, enc, features)

    model = LGBMClassifier(scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1), **LGBM_PARAMS).fit(Xtr, ytr)
    threshold = choose_threshold(model.predict_proba(Xva)[:, 1], yva, ava)
    challenger = metrics(model.predict_proba(Xte)[:, 1], yte, ate, threshold)

    if champ_model is not None:
        Xte_c, _, _ = make_xy(te, champ_fill, champ_enc, features)      # champion with ITS OWN recipe
        champion = metrics(champ_model.predict_proba(Xte_c)[:, 1], yte, ate, float(champ_cfg["threshold"]))
        wins = (challenger["pr_auc"] >= champion["pr_auc"] and challenger["money_caught_at3"] >= champion["money_caught_at3"] - 0.005
                and challenger["alert_rate"] <= config.ALERT_CAP + 0.01 and challenger["frauds"] >= 20)
    else:
        champion = {k: None for k in challenger}
        wins = True
    comp = pd.DataFrame({"champion (current)": champion, "challenger (new)": challenger})
    print(f"\nLatest {test_months} mature months — champion {champ_cfg.get('model_version')} vs challenger:")
    print(comp.to_string())
    print(f"\n{'🏆 Challenger WINS' if wins else '🛡️ Champion stays'}")

    new_version = next_version(str(champ_cfg.get("model_version", "v1.0")))
    report = {"run_id": run_id, "reason": reason, "decision": "promote" if (wins and promote) else "keep",
              "challenger_wins": bool(wins), "champion_version": champ_cfg.get("model_version"),
              "new_version": new_version if wins else None, "split": d.attrs, "champion": champion, "challenger": challenger,
              "threshold": threshold, "rows": int(len(hist))}
    storage.write_json(report, f"reports/retraining/{run_id}/report.json")

    if force and promote and not wins:
        print("⚠️ --force: promoting the challenger even though it did not win (e.g. first deployment / library upgrade)")
        wins = True
        report.update(decision="promote", challenger_wins=False, new_version=new_version, forced=True)
    if wins and promote:
        promote_model(model, enc, fill, features, threshold, challenger, new_version, d, hist, reason, run_id)
    return report


def promote_model(model, enc, fill, features, threshold, test_metrics, version, d, hist, reason, run_id):
    bands = {"APPROVE": [0, round(threshold, 3)], "CONFIRM_WITH_CUSTOMER": [round(threshold, 3), 0.70],
             "REVIEW": [0.70, 0.90], "BLOCK": [0.90, 1.01]}
    old = storage.read_json(config.MODEL_CONFIG_KEY)
    cfg = {**old, "model_name": "LightGBM", "model_version": version, "features": features, "threshold": round(threshold, 3),
           "bands": bands, "trained_until": d.attrs["train_end"], "test": test_metrics,
           "retrained": {"run_id": run_id, "reason": reason, "at": datetime.now(timezone.utc).isoformat(), "previous": old.get("model_version")}}
    old_prefix = f"{config.MODEL_HISTORY_PREFIX}{old.get('model_version', 'v1.0')}/"
    if not storage.exists(old_prefix + "model_config.json"):        # back up the CURRENT model before replacing it
        for src_key, name in [(config.MODEL_KEY, "best_model.pkl"), (config.ENCODERS_KEY, "encoders.pkl"),
                              (config.IMPUTE_KEY, "impute_values.json"), (config.MODEL_CONFIG_KEY, "model_config.json"),
                              (config.FEATURE_LIST_KEY, "feature_list.json")]:
            storage.copy(src_key, old_prefix + name)
    hist_prefix = f"{config.MODEL_HISTORY_PREFIX}{version}/"
    storage.write_pickle(model, hist_prefix + "best_model.pkl")       # keep every version (easy rollback)
    storage.write_pickle(enc, hist_prefix + "encoders.pkl")
    storage.write_json(fill, hist_prefix + "impute_values.json")
    storage.write_json(cfg, hist_prefix + "model_config.json")
    # refresh what the API reads — the CONFIG goes last, because a new version number triggers the reload
    storage.write_parquet(hist, "data/history_cleaned.parquet")
    save_store_tables(build_store_tables(hist))
    tr = d[d["split"] == "train"]
    ref_rows = tr.sample(min(20000, len(tr)), random_state=7)
    Xref, yref, _ = make_xy(ref_rows, fill, enc, features)
    ref = pd.concat([ref_rows[["txn_id", "txn_timestamp"]].reset_index(drop=True), Xref.reset_index(drop=True)], axis=1)
    ref["fraud_score"], ref["is_fraud"] = model.predict_proba(Xref)[:, 1], yref
    storage.write_parquet(ref, config.REFERENCE_KEY)                   # new "normal" for drift detection
    storage.write_pickle(enc, config.ENCODERS_KEY)
    storage.write_json(fill, config.IMPUTE_KEY)
    storage.write_json({"features": features, "train_end": d.attrs["train_end"], "valid_end": d.attrs["valid_end"]}, config.FEATURE_LIST_KEY)
    storage.write_pickle(model, config.MODEL_KEY)
    storage.write_json(cfg, config.MODEL_CONFIG_KEY)
    print(f"✅ Promoted {version} → s3://{config.S3_BUCKET}/{config.MODEL_KEY} (API pods reload within {config.MODEL_RELOAD_SECONDS}s)")


def notify_github(version: str):
    """Optional: tell GitHub Actions to restart the Kubernetes pods now (instead of waiting for the reload timer)."""
    if not config.GITHUB_REPO:
        return "skipped (GITHUB_REPO not set)"
    import boto3
    import requests
    token = boto3.client("secretsmanager", region_name=config.AWS_REGION).get_secret_value(SecretId=config.GITHUB_TOKEN_SECRET)["SecretString"]
    r = requests.post(f"https://api.github.com/repos/{config.GITHUB_REPO}/dispatches",
                      headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                      json={"event_type": "new-model", "client_payload": {"version": version}}, timeout=20)
    return f"GitHub dispatch → HTTP {r.status_code}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true", help="save the new model if it wins")
    ap.add_argument("--valid-months", type=int, default=config.VALID_MONTHS)
    ap.add_argument("--test-months", type=int, default=config.TEST_MONTHS)
    ap.add_argument("--force", action="store_true", help="promote even if it does not beat the current model")
    a = ap.parse_args()
    print(json.dumps(retrain(a.promote, a.valid_months, a.test_months, force=a.force), indent=2, default=str)[:2000])
