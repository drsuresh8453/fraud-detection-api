"""
prepare_artifacts.py — run ONCE after the 7 notebooks (and again by retraining).

Reads from the bucket:  data/02_cleaned.csv, data/03_imputed.csv, data/06_features.csv, models/…
Writes to the bucket:
  feature_store/*.parquet               ← customer profiles, device registry, payee store, snapshots, city distances
  monitoring/reference_features.parquet ← "normal" features + scores for drift detection (train period, genuine + fraud)
  artifacts/impute_values.json          ← adds 'login_overall' (fill value for card-at-shop payments)

Usage:  python -m scripts.prepare_artifacts
"""
import numpy as np
import pandas as pd

from src import config, storage
from src.feature_store import build_store_tables, save_store_tables


def build_reference(model=None, sample=20000, seed=7):
    feats = storage.read_csv("data/06_features.csv", low_memory=False)
    flist = storage.read_json(config.FEATURE_LIST_KEY)["features"]
    ref = feats[feats["split"] == "train"]
    ref = ref.sample(min(sample, len(ref)), random_state=seed)
    out = ref[["txn_id", "txn_timestamp"] + flist].copy()
    if model is None:
        model = storage.read_pickle(config.MODEL_KEY)
    out["fraud_score"] = model.predict_proba(ref[flist].astype(float))[:, 1]
    out["is_fraud"] = ref["is_fraud"].values
    return out


def main():
    print(f"Storage: {config.STORAGE_MODE} | bucket/folder: {config.S3_BUCKET if config.STORAGE_MODE == 's3' else config.LOCAL_STORAGE_DIR}")
    clean = storage.read_csv("data/02_cleaned.csv", low_memory=False, parse_dates=["txn_timestamp", "account_open_date"])
    tables = build_store_tables(clean)
    save_store_tables(tables)
    for k, v in tables.items():
        print(f"  feature_store/{k}.parquet  {len(v):,} rows")

    fill = storage.read_json(config.IMPUTE_KEY)
    if "login_overall" not in fill:
        imp = storage.read_csv("data/03_imputed.csv", low_memory=False, usecols=["channel", "login_to_pay_seconds"])
        pos = imp.loc[imp["channel"] == "CARD_POS", "login_to_pay_seconds"]
        fill["login_overall"] = float(pos.iloc[0]) if len(pos) else float(imp["login_to_pay_seconds"].median())
        fill["login_by_channel"] = {k: v for k, v in fill["login_by_channel"].items() if v == v}   # drop NaN
        storage.write_json(fill, config.IMPUTE_KEY)
        print(f"  impute_values.json: login_overall = {fill['login_overall']}")

    ref = build_reference()
    storage.write_parquet(ref, config.REFERENCE_KEY)
    print(f"  {config.REFERENCE_KEY}: {len(ref):,} rows × {ref.shape[1]} columns")

    cfg = storage.read_json(config.MODEL_CONFIG_KEY)
    if "model_version" not in cfg or cfg["model_version"] in (None, ""):
        cfg["model_version"] = "v1.0"
        storage.write_json(cfg, config.MODEL_CONFIG_KEY)
    print("✅ Artifacts ready — the API can start now.")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    np.seterr(all="ignore")
    main()
