"""
features.py — builds the model's feature row (Notebook 6). THE most important production file.

The same function is used:
  • in training  (train.py, on a whole table)
  • in the API   (on one payment)
so the model always sees features calculated in exactly the same way ("no training/serving skew").
"""
import numpy as np
import pandas as pd

ONEHOT = ["channel", "txn_type", "device_os", "merchant_category", "occupation"]
SCAM_WORDS = ["kyc", "refund", "urgent", "verification", "task", "deposit", "level", "recharge", "olx", "payment received"]
SMOOTH = 200


def add_basic_features(d: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Notebook 6, Step 3 — new features from the payment + fetched values."""
    ts = pd.to_datetime(d["txn_timestamp"])
    scam = enc.get("scam_words", SCAM_WORDS)
    amount = d["amount"].to_numpy(dtype=float)
    hour = ts.dt.hour.to_numpy()
    is_night = np.isin(hour, enc.get("night_hours", [0, 1, 2, 3, 4])).astype(int)
    vs_usual = np.round(amount / np.clip(d["avg_txn_amount_90d"].to_numpy(dtype=float), 1, None), 3)
    payee_age = d["payee_account_age_days"].to_numpy(dtype=float)
    merchant_age = d["merchant_age_days"].to_numpy(dtype=float)
    new = {
        "hour": hour,
        "is_night": is_night,
        "is_weekend": (ts.dt.dayofweek.to_numpy() >= 5).astype(int),
        "is_festival_month": np.isin(ts.dt.month.to_numpy(), enc.get("festival_months", [10, 11])).astype(int),
        "log_amount": np.log1p(amount),
        "is_round_amount": ((amount % 1000 == 0) | (amount % 1000 == 999)).astype(int),
        "amount_vs_usual": vs_usual,
        "new_device_and_new_payee": d["is_new_device"].to_numpy() * d["is_new_payee"].to_numpy(),
        "night_and_high_amount": is_night * (vs_usual > 3).astype(int),
        "velocity_ratio": np.round(d["txn_count_1h"].to_numpy(dtype=float) / (d["txn_count_90d"].to_numpy(dtype=float) / 2160 + 0.01), 3),
        "remarks_scam_word": [int(any(w in str(t).lower() for w in scam)) for t in d["remarks"].fillna("")],
        "is_merchant_payment": (d["merchant_category"].to_numpy() != "NOT_MERCHANT").astype(int),
        "payee_handle": [p[p.rfind("@"):] if "@" in p and p[p.rfind("@") + 1:].replace("_", "").isalnum() else "MERCHANT"
                         for p in d["payee_id"].astype(str)],
        "is_foreign_ip": (~d["ip_country"].isin(["IN", "UNKNOWN"])).to_numpy().astype(int),
        # Stable bucketed ages (Notebook 6, Step 10 — raw ages drift naturally)
        "payee_is_new_account_30d": (payee_age < 30).astype(int),
        "payee_account_lt_1y": (payee_age < 365).astype(int),
        "account_is_new_90d": (d["account_age_days"].to_numpy(dtype=float) < 90).astype(int),
        "device_linked_lt_7d": (d["device_age_days"].to_numpy(dtype=float) < 7).astype(int),
        "merchant_is_new_90d": ((merchant_age >= 0) & (merchant_age <= 89)).astype(int),
    }
    return pd.concat([d.drop(columns=[c for c in new if c in d.columns]), pd.DataFrame(new, index=d.index)], axis=1)


def fit_encoders(train_labelled: pd.DataFrame, all_rows: pd.DataFrame = None) -> dict:
    """Notebook 6, Step 5 — learn the encoders from TRAIN rows only (no leakage)."""
    all_rows = train_labelled if all_rows is None else all_rows
    global_rate = float(train_labelled["is_fraud"].mean())
    st = train_labelled.groupby("payee_handle")["is_fraud"].agg(["sum", "count"])
    handle_risk = ((st["sum"] + SMOOTH * global_rate) / (st["count"] + SMOOTH)).round(5).to_dict()
    categories = {c: sorted(all_rows[c].astype(str).unique().tolist()) for c in ONEHOT}
    return {"handle_risk": handle_risk, "handle_default": global_rate, "onehot_categories": categories,
            "kyc_map": {"MIN": 0, "FULL": 1}, "scam_words": SCAM_WORDS, "festival_months": [10, 11],
            "night_hours": [0, 1, 2, 3, 4]}


def encode(d: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Notebook 6, Step 5 — apply the SAVED encoders."""
    d = d.copy()
    d["kyc_level_code"] = d["kyc_level"].map(enc["kyc_map"]).fillna(-1)
    d["payee_handle_risk"] = d["payee_handle"].map(enc["handle_risk"]).fillna(enc["handle_default"])
    onehot = {}
    for col, values in enc["onehot_categories"].items():
        s = d[col].astype(str).to_numpy()
        for v in values:
            onehot[f"{col}_{v}"] = (s == v).astype(int)   # a value never seen in training → all zeros (safe)
    return pd.concat([d, pd.DataFrame(onehot, index=d.index)], axis=1)   # one concat instead of 35 inserts (faster)


def build_features(d: pd.DataFrame, enc: dict, feature_list: list) -> pd.DataFrame:
    """Full recipe: basic features → encoding → final columns in the EXACT training order."""
    d = encode(add_basic_features(d, enc), enc)
    missing = [f for f in feature_list if f not in d.columns]
    if missing:
        raise ValueError(f"features could not be built: {missing}")
    return d[feature_list].astype(float)


def unseen_categories(d: pd.DataFrame, enc: dict) -> list:
    """Warn when a live payment has a category the model never saw (e.g. a new channel)."""
    out = []
    for col, values in enc["onehot_categories"].items():
        for v in d[col].astype(str).unique():
            if v not in values:
                out.append(f"{col}={v}")
    return out
