"""
simulate_new_data.py — creates a realistic "new month" situation to TEST drift detection and retraining.

Two kinds of change are injected into Jun–Oct 2026:
  1. DATA DRIFT    — the bank's app adds a biometric step from 1 Sep 2026 → login_to_pay_seconds roughly doubles.
                     (The world changed; the model's inputs look different.)
  2. CONCEPT DRIFT — a NEW scam type appears from June: "investment / trading-profit" scams. Victims use their OWN
                     phone, in the evening, paying aged rented mule accounts on low-risk banks — the current model was
                     never trained on this pattern, so it misses many. (The meaning of the inputs changed.)

Writes:
  new_data/2026-11_bank_export.csv  ← labelled export the bank sends (used by retraining)
  feature_store/snapshots.parquet   ← adds the new payments (as the bank's real-time stores would hold them)
  client/drift_traffic.csv          ← Sep–Oct payments to replay through the API (the "live traffic")
In this simulation we pretend the 90-day label wait for Jun–Oct is over (the bank sent final labels).
Usage: python -m scripts.simulate_new_data
"""
import numpy as np
import pandas as pd

from src import config, storage
from src.feature_store import SNAPSHOT_COLS
from src.predict import SENT_FIELDS

rng = np.random.default_rng(2026)
WAVE_PER_MONTH = {6: 30, 7: 45, 8: 60, 9: 75, 10: 90}          # the scam wave grows month by month
SAFE_HANDLES = ["@oksbi", "@okicici", "@okhdfcbank"]
NOTES = ["profit share", "trading returns", "stock tips fee", "ipo allotment", "crypto profit", ""]


def make_wave(base: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = 0
    for month, k in WAVE_PER_MONTH.items():
        pool = base[(base["txn_timestamp"].dt.month == month) & (base["channel"] == "UPI") & (base["is_fraud"] != 1)]
        victims = pool.sample(k, random_state=month)
        for _, r in victims.iterrows():
            n += 1
            r = r.copy()
            usual = max(float(r["avg_txn_amount_90d"]) if pd.notna(r["avg_txn_amount_90d"]) else 2000.0, 300.0)
            ts = r["txn_timestamp"].normalize() + pd.Timedelta(hours=int(rng.integers(18, 22)), minutes=int(rng.integers(0, 60)))
            r.update({"txn_id": f"TXN4{n:07d}", "txn_timestamp": ts, "txn_type": "P2P_TRANSFER",
                      "amount": float(np.round(usual * rng.uniform(2.5, 6.0), -1) + rng.integers(1, 9)),
                      "payee_id": f"inv{rng.integers(10000, 99999)}{rng.choice(SAFE_HANDLES)}", "merchant_category": np.nan,
                      "remarks": rng.choice(NOTES), "is_new_payee": 1, "payee_account_age_days": float(rng.integers(200, 900)),
                      "payee_reports_count": 0, "payee_incoming_count_24h": int(rng.integers(8, 25)),
                      "merchant_age_days": np.nan, "merchant_chargeback_rate": np.nan,
                      "is_fraud": 1.0, "fraud_type": "INVESTMENT_SCAM", "label_status": "CONFIRMED_FRAUD",
                      "label_source": "CUSTOMER_COMPLAINT", "label_date": (ts + pd.Timedelta(days=int(rng.integers(3, 20)))).strftime("%Y-%m-%d"),
                      "attack_id": f"INV{month:02d}"})
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    clean = storage.read_csv("data/02_cleaned.csv", low_memory=False, parse_dates=["txn_timestamp", "account_open_date"])
    period = clean[(clean["txn_timestamp"] >= "2026-06-01") & (clean["txn_timestamp"] <= "2026-10-31 23:59:59")].copy()

    # labels for Jun–Oct are now final (simulation)
    open_lbl = period["label_status"].isin(["NOT_MATURE", "PENDING"]) & period["is_fraud"].isna()
    period.loc[open_lbl, "label_status"] = "NO_COMPLAINT_90D"
    period.loc[open_lbl, "is_fraud"] = 0.0
    pend_fraud = period["label_status"].isin(["NOT_MATURE", "PENDING"]) & (period["is_fraud"] == 1)
    period.loc[pend_fraud, "label_status"] = "CONFIRMED_FRAUD"

    # 1. DATA DRIFT — biometric step in the app from 1 Sep
    after = (period["txn_timestamp"] >= "2026-09-01") & (period["channel"] != "CARD_POS")
    period.loc[after, "login_to_pay_seconds"] = (period.loc[after, "login_to_pay_seconds"] * rng.uniform(1.8, 2.6, after.sum())).round(1)

    # 2. CONCEPT DRIFT — new scam wave
    wave = make_wave(period)
    wave_after = wave["txn_timestamp"] >= "2026-09-01"
    wave.loc[wave_after, "login_to_pay_seconds"] = (wave.loc[wave_after, "login_to_pay_seconds"] * 2.2).round(1)
    export = pd.concat([period, wave], ignore_index=True).sort_values("txn_timestamp")
    export["txn_timestamp"] = export["txn_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export["account_open_date"] = pd.to_datetime(export["account_open_date"]).dt.strftime("%Y-%m-%d")
    storage.write_csv(export, f"{config.NEW_DATA_PREFIX}2026-11_bank_export.csv")

    # the real-time stores hold these payments too
    snaps = storage.read_parquet(f"{config.FEATURE_STORE_PREFIX}snapshots.parquet")
    new_snap = pd.concat([period, wave])[["txn_id", "txn_timestamp"] + SNAPSHOT_COLS]
    snaps = pd.concat([snaps[~snaps["txn_id"].isin(new_snap["txn_id"])], new_snap], ignore_index=True)
    storage.write_parquet(snaps, f"{config.FEATURE_STORE_PREFIX}snapshots.parquet")

    live = pd.concat([period, wave])
    live = live[live["txn_timestamp"] >= "2026-09-01"].sort_values("txn_timestamp")
    live[SENT_FIELDS + ["is_fraud"]].to_csv("client/drift_traffic.csv", index=False)
    print(f"✅ new_data export: {len(export):,} rows (incl. {len(wave)} INVESTMENT_SCAM frauds) | "
          f"live traffic Sep–Oct: {len(live):,} payments → client/drift_traffic.csv")


if __name__ == "__main__":
    main()
