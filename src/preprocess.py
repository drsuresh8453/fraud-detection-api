"""
preprocess.py — cleaning (Notebook 2) and filling missing values (Notebook 3).

Two ways to use it, SAME rules:
  • clean_batch()  → a whole monthly export (training / retraining in SageMaker)
  • fill_missing() → one live payment in the API, or a whole training table

The fill values (age by occupation, medians …) are LEARNED once in training (fit_fill_values) and
saved as artifacts/impute_values.json. The API only USES them — it never recalculates a median from one payment.
"""
import numpy as np
import pandas as pd

VALID_CHANNELS = ["UPI", "CARD_POS", "CARD_ONLINE", "NETBANKING"]
VALID_TYPES = ["P2P_TRANSFER", "P2M_PAYMENT", "BILL_PAYMENT", "COLLECT_REQUEST", "CARD_PURCHASE"]
CITY_FIX = {"Bombay": "Mumbai", "Bangalore": "Bengaluru", "Calcutta": "Kolkata", "Madras": "Chennai"}
YES_NO = {"1": 1, "1.0": 1, "Y": 1, "YES": 1, "TRUE": 1, "0": 0, "0.0": 0, "N": 0, "NO": 0, "FALSE": 0}
AGE_WORDS = {"sixty two": 62, "forty": 40, "thirty five": 35, "45 yrs": 45, "na": np.nan}
TS_FORMATS = ["%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S+05:30", "%d-%b-%Y %I:%M %p"]
WINDOW_START = pd.Timestamp("2023-11-01")
FIRST_PAYMENT_GAP_SECONDS = 90 * 24 * 3600


# ════════════════════════════════════════════════════════════════════════════════════════
# 1. CLEANING A MONTHLY EXPORT  (Notebook 2, Steps 2–8)
# ════════════════════════════════════════════════════════════════════════════════════════
def clean_batch(df: pd.DataFrame, window_end=None, verbose=True) -> pd.DataFrame:
    """Clean a raw export exactly as Notebook 2 did. Returns a new DataFrame sorted by time."""
    log = []
    d = df.copy()
    n0 = len(d)

    # Step 2 — duplicates
    d = d.drop_duplicates()
    log.append(("duplicates removed", n0 - len(d)))

    # Step 3 — one spelling per category
    d["channel"] = (d["channel"].astype(str).str.strip().str.upper()
                    .replace({"U.P.I": "UPI", "CARD POS": "CARD_POS", "CARD ONLINE": "CARD_ONLINE"}))
    d["txn_type"] = d["txn_type"].astype(str).str.strip().str.upper().str.replace(" ", "_")
    d["kyc_level"] = d["kyc_level"].astype("string").str.strip().str.upper().replace({"F": "FULL", "MINIMUM": "MIN"})
    d["occupation"] = d["occupation"].astype("string").str.strip().str.title()
    d["device_os"] = (d["device_os"].astype("string").str.strip().str.lower()
                      .replace({"android 14": "android", "emulator-x86": "emulator"})
                      .map({"android": "Android", "ios": "iOS", "emulator": "Emulator"}))
    for col in ["txn_city", "home_city"]:
        d[col] = d[col].astype(str).str.strip().str.title().replace(CITY_FIX)
    d["is_international"] = d["is_international"].astype(str).str.strip().str.upper().map(YES_NO)

    # Step 4 — amount
    d["amount"] = pd.to_numeric(d["amount"].astype(str).str.replace("₹", "").str.replace(",", "").str.strip(), errors="coerce")
    bad = d["amount"].isna() | (d["amount"] <= 0)
    log.append(("invalid amount removed", int(bad.sum())))
    d = d[~bad]

    # Step 5 — timestamps (4 formats) + impossible dates
    raw_ts = d["txn_timestamp"].astype(str).str.strip()
    ts = pd.to_datetime(raw_ts, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    for fmt in TS_FORMATS:
        ts = ts.fillna(pd.to_datetime(raw_ts, format=fmt, errors="coerce"))
    d["txn_timestamp"] = ts
    window_end = pd.Timestamp(window_end) if window_end is not None else pd.Timestamp.now() + pd.Timedelta(days=1)
    bad = ts.isna() | (ts < WINDOW_START) | (ts > window_end)
    log.append(("invalid timestamp removed", int(bad.sum())))
    d = d[~bad].sort_values("txn_timestamp").reset_index(drop=True)

    # Step 6 — customer age
    age = d["customer_age"].astype(str).str.strip().str.lower()
    age = pd.to_numeric(age.replace(AGE_WORDS), errors="coerce")
    d["customer_age"] = age.where(~((age < 18) | (age > 100)))

    # Step 7 — other impossible values
    d["account_open_date"] = pd.to_datetime(d["account_open_date"].astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
    d.loc[d["account_open_date"] > d["txn_timestamp"], "account_open_date"] = pd.NaT
    d["account_age_days"] = (d["txn_timestamp"] - d["account_open_date"]).dt.days
    for col in ["device_age_days", "txn_count_24h", "merchant_chargeback_rate", "txn_count_1h"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d.loc[d["device_age_days"] < 0, "device_age_days"] = np.nan
    d.loc[d["txn_count_24h"] > 1000, "txn_count_24h"] = np.nan
    d.loc[d["merchant_chargeback_rate"] > 1, "merchant_chargeback_rate"] = np.nan

    # Step 8 — consistency repairs
    d = apply_consistency_repairs(d)

    if verbose:
        print(f"clean_batch: {n0:,} → {len(d):,} rows | " + ", ".join(f"{k}: {v:,}" for k, v in log))
    return d


def apply_consistency_repairs(d: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2 Step 8 repairs — used for training AND for live payments."""
    is_merchant_payee = d["payee_id"].astype(str).str.startswith("MER")
    wrong_p2p = (d["txn_type"] == "P2P_TRANSFER") & (d["merchant_category"].notna() | is_merchant_payee)
    d.loc[wrong_p2p, "txn_type"] = "P2M_PAYMENT"
    if "txn_count_24h" in d and "txn_count_1h" in d:
        fix = d["txn_count_1h"] > d["txn_count_24h"]
        d.loc[fix, "txn_count_24h"] = d.loc[fix, "txn_count_1h"]
    return d


def data_quality_gate(d: pd.DataFrame) -> pd.DataFrame:
    """Notebook 2 Step 8 rules. Training STOPS if a critical rule fails (broken labels teach wrong lessons)."""
    label_date = pd.to_datetime(d.get("label_date"), errors="coerce") if "label_date" in d else pd.Series(pd.NaT, index=d.index)
    rules = {
        "label CONFIRMED_FRAUD but is_fraud is not 1": ((d["label_status"] == "CONFIRMED_FRAUD") & (d["is_fraud"] != 1), True),
        "label NOT_MATURE / PENDING but is_fraud filled": (d["label_status"].isin(["NOT_MATURE", "PENDING"]) & d["is_fraud"].notna(), True),
        "fraud confirmed BEFORE the payment happened": (label_date < d["txn_timestamp"].dt.normalize(), True),
        "international flag on a card swiped in a shop": ((d["is_international"] == 1) & (d["channel"] == "CARD_POS"), False),
        "unknown channel": (~d["channel"].isin(VALID_CHANNELS), True),
    }
    rep = pd.DataFrame([{"rule": k, "violations": int(m.sum()), "critical": c} for k, (m, c) in rules.items()])
    rep["status"] = np.where(rep["violations"] == 0, "passed", np.where(rep["critical"], "FAILED", "warning"))
    return rep


# ════════════════════════════════════════════════════════════════════════════════════════
# 2. FILLING MISSING VALUES  (Notebook 3, Step 6)
# ════════════════════════════════════════════════════════════════════════════════════════
def fit_fill_values(d: pd.DataFrame) -> dict:
    """Learn the fill values from TRAINING data (saved to artifacts/impute_values.json)."""
    occ = d["occupation"].fillna("Unknown")
    fill = {"age_by_occupation": d.groupby(occ)["customer_age"].median().round(0).dropna().to_dict(),
            "age_overall": float(d["customer_age"].median()),
            "login_by_channel": {k: float(v) for k, v in d.groupby("channel")["login_to_pay_seconds"].median().dropna().items()},
            "login_overall": float(d["login_to_pay_seconds"].median()),
            "distance_other_city": float(d.loc[d["distance_from_home_km"] > 0, "distance_from_home_km"].median())}
    pay_age = d["payee_account_age_days"].fillna(d["merchant_age_days"].where(d["merchant_age_days"] >= 0))
    fill["payee_account_age_days"] = float(pay_age.median())
    fill["device_age_days"] = float(d["device_age_days"].median())
    fill["account_age_days"] = float(d["account_age_days"].median())
    cb = d[d["merchant_chargeback_rate"] >= 0]
    fill["chargeback_by_category"] = cb.groupby("merchant_category")["merchant_chargeback_rate"].median().to_dict()
    return fill


def fill_missing(d: pd.DataFrame, fill: dict) -> pd.DataFrame:
    """Apply the Notebook 3 rules with SAVED fill values. Works for 1 row (API) or 1 lakh rows (training)."""
    d = d.copy()
    # 1. Structural: not a merchant / no device
    d["merchant_category"] = d["merchant_category"].fillna("NOT_MERCHANT")
    not_merchant = d["merchant_category"] == "NOT_MERCHANT"
    for col in ["merchant_age_days", "merchant_chargeback_rate"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype(float)
        d.loc[not_merchant, col] = -1
    d["device_id"] = d["device_id"].fillna("NO_DEVICE")
    d["device_os"] = d["device_os"].fillna("NONE")
    for col in ["is_new_device", "is_rooted_or_emulator", "vpn_or_proxy", "failed_pin_attempts_24h", "password_reset_last_24h"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
    d["accounts_on_device"] = pd.to_numeric(d["accounts_on_device"], errors="coerce").fillna(1)

    # 2. First payments
    d["is_first_txn_90d"] = d["avg_txn_amount_90d"].isna().astype(int)
    d["avg_txn_amount_90d"] = d["avg_txn_amount_90d"].fillna(d["amount"])
    d["seconds_since_last_txn"] = d["seconds_since_last_txn"].fillna(FIRST_PAYMENT_GAP_SECONDS)

    # 3. Customer age by occupation
    d["occupation"] = d["occupation"].fillna("Unknown")
    d["customer_age"] = (d["customer_age"].fillna(d["occupation"].map(fill["age_by_occupation"]))
                         .fillna(fill["age_overall"]))

    # 4. Unknown categories
    d["kyc_level"] = d["kyc_level"].fillna("UNKNOWN")
    d["ip_country_missing"] = (d["ip_country"].isna() & (d["channel"] != "CARD_POS")).astype(int)
    d["ip_country"] = d["ip_country"].fillna("UNKNOWN")
    d["remarks"] = d["remarks"].fillna("")

    # 5. Numeric gaps
    login_overall = fill.get("login_overall", np.nanmedian([v for v in fill["login_by_channel"].values() if v == v]))
    d["login_to_pay_seconds"] = (d["login_to_pay_seconds"].fillna(d["channel"].map(fill["login_by_channel"]))
                                 .fillna(login_overall))
    same_city = d["txn_city"] == d["home_city"]
    d.loc[d["distance_from_home_km"].isna() & same_city, "distance_from_home_km"] = 0
    d["distance_from_home_km"] = d["distance_from_home_km"].fillna(fill["distance_other_city"])
    d["payee_account_age_days"] = d["payee_account_age_days"].fillna(d["merchant_age_days"].where(d["merchant_age_days"] >= 0))
    for col in ["payee_account_age_days", "device_age_days", "account_age_days"]:
        d[col] = d[col].fillna(fill[col])
    d["txn_count_24h"] = d["txn_count_24h"].fillna(d["txn_count_1h"])
    d["merchant_chargeback_rate"] = d["merchant_chargeback_rate"].fillna(d["merchant_category"].map(fill["chargeback_by_category"]))
    d["merchant_chargeback_rate"] = d["merchant_chargeback_rate"].fillna(0.0)   # unseen category → safe default
    return d
