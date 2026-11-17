"""
feature_store.py — fetches everything the bank does NOT send (project document, Section 11).

  Request (sent by bank)  →  + customer profile   (database)
                             + device registry    (database)
                             + payee & mule store (database)
                             + live counters      (payments in last 1 h / 24 h …)
                             + calculated values  (distance from home)

In a real bank these live in Redis / DynamoDB. Here they are tables in S3 (feature_store/…), loaded
into memory at start-up — the interface (lookup / update) stays the same if you later swap in Redis.

Point-in-time lookup: for a payment that is already in our history (replaying a past payment for testing),
the store returns the values AS THEY WERE at that moment (snapshots table). For a brand-new payment it uses
the latest profile values + live counters.
"""
import threading
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src import config, storage

PROFILE_COLS = ["customer_age", "occupation", "kyc_level", "is_salary_account", "past_fraud_reports", "home_city",
                "avg_txn_amount_90d", "txn_count_90d", "num_payees_90d"]
SNAPSHOT_COLS = PROFILE_COLS + ["account_age_days", "txn_count_1h", "txn_count_24h", "amount_sum_24h",
                                "seconds_since_last_txn", "failed_pin_attempts_24h", "password_reset_last_24h",
                                "sim_changed_last_7d", "is_new_device", "device_age_days", "accounts_on_device",
                                "is_new_payee", "payee_account_age_days", "payee_reports_count",
                                "payee_incoming_count_24h", "merchant_age_days", "merchant_chargeback_rate",
                                "distance_from_home_km"]
SECURITY_COLS = ["failed_pin_attempts_24h", "password_reset_last_24h", "sim_changed_last_7d"]


# ════════════════════════════════════════════════════════════════════════════════════════
# Build the store tables from cleaned history (run once by scripts/prepare_artifacts.py)
# ════════════════════════════════════════════════════════════════════════════════════════
def build_store_tables(clean: pd.DataFrame) -> dict:
    c = clean.sort_values("txn_timestamp")
    last = c.groupby("customer_id").tail(1)
    customers = last[["customer_id", "txn_timestamp", "account_open_date"] + PROFILE_COLS].rename(columns={"txn_timestamp": "as_of"})
    dev = c[c["device_id"].notna()]
    devices = (dev.groupby("device_id").agg(as_of=("txn_timestamp", "last"), device_age_days=("device_age_days", "last"),
                                            accounts_on_device=("customer_id", "nunique"), device_os=("device_os", "last"))
               .reset_index())
    payees = (c.groupby("payee_id").agg(as_of=("txn_timestamp", "last"), payee_account_age_days=("payee_account_age_days", "last"),
                                        payee_reports_count=("payee_reports_count", "max"), merchant_age_days=("merchant_age_days", "last"),
                                        merchant_chargeback_rate=("merchant_chargeback_rate", "last"),
                                        merchant_category=("merchant_category", "last"))
              .reset_index())
    pairs = c[["customer_id", "payee_id"]].drop_duplicates()
    other = c[(c["txn_city"] != c["home_city"]) & c["distance_from_home_km"].notna()]
    city = other.groupby(["home_city", "txn_city"])["distance_from_home_km"].median().reset_index()
    snapshots = c[["txn_id", "txn_timestamp"] + SNAPSHOT_COLS]
    return {"customers": customers, "devices": devices, "payees": payees, "customer_payees": pairs,
            "city_distance": city, "snapshots": snapshots}


def save_store_tables(tables: dict):
    for name, df in tables.items():
        storage.write_parquet(df, f"{config.FEATURE_STORE_PREFIX}{name}.parquet")


# ════════════════════════════════════════════════════════════════════════════════════════
# Live counters (in Redis in a real bank; in memory here, one per API pod)
# ════════════════════════════════════════════════════════════════════════════════════════
class LiveCounters:
    def __init__(self):
        self.cust = defaultdict(deque)      # customer_id → (timestamp, amount)
        self.payee = defaultdict(deque)     # payee_id → timestamp
        self.lock = threading.Lock()

    @staticmethod
    def _trim(q, now, hours=24):
        while q and (now - q[0][0]).total_seconds() > hours * 3600:
            q.popleft()

    def read(self, customer_id, payee_id, now):
        with self.lock:
            q = self.cust[customer_id]
            self._trim(q, now)
            last_1h = [a for t, a in q if (now - t).total_seconds() <= 3600]
            p = self.payee[payee_id]
            self._trim(p, now)
            return {"txn_count_1h": len(last_1h) + 1, "txn_count_24h": len(q) + 1,
                    "amount_sum_24h": float(sum(a for _, a in q)),
                    "seconds_since_last_txn": (now - q[-1][0]).total_seconds() if q else np.nan,
                    "payee_incoming_count_24h": len(p)}

    def add(self, customer_id, payee_id, now, amount):
        with self.lock:
            self.cust[customer_id].append((now, amount))
            self.payee[payee_id].append((now, 1))


class FeatureStore:
    def __init__(self):
        self.counters = LiveCounters()
        self.loaded = False

    def load(self):
        p = config.FEATURE_STORE_PREFIX
        self.customers = storage.read_parquet(f"{p}customers.parquet").set_index("customer_id")
        self.devices = storage.read_parquet(f"{p}devices.parquet").set_index("device_id")
        self.payees = storage.read_parquet(f"{p}payees.parquet").set_index("payee_id")
        pairs = storage.read_parquet(f"{p}customer_payees.parquet")
        self.known_pairs = set(zip(pairs["customer_id"], pairs["payee_id"]))
        city = storage.read_parquet(f"{p}city_distance.parquet")
        self.city = {(a, b): d for a, b, d in city[["home_city", "txn_city", "distance_from_home_km"]].itertuples(index=False)}
        snaps = storage.read_parquet(f"{p}snapshots.parquet")
        self.snapshots = snaps.set_index("txn_id")
        self.loaded = True
        return self

    # ── one payment ────────────────────────────────────────────────────────────────────
    def lookup(self, req: dict) -> dict:
        """Return all fetched fields for one payment (point-in-time if the payment is already in history)."""
        if req["txn_id"] in self.snapshots.index:
            out = self.snapshots.loc[req["txn_id"], SNAPSHOT_COLS].to_dict()
            out["_source"] = "snapshot"
        else:
            out = self._live(req)
            out["_source"] = "live"
        for col in SECURITY_COLS:                       # values sent by the bank win over fetched ones
            if req.get(col) is not None:
                out[col] = req[col]
        return out

    def _live(self, req: dict) -> dict:
        now = pd.Timestamp(req["txn_timestamp"]).tz_localize(None)
        out = {}
        # customer profile
        if req["customer_id"] in self.customers.index:
            c = self.customers.loc[req["customer_id"]]
            out.update({k: c[k] for k in PROFILE_COLS})
            opened = c["account_open_date"]
            out["account_age_days"] = (now - pd.Timestamp(opened)).days if pd.notna(opened) else np.nan
        else:                                           # unknown customer → missing values, filled by preprocess
            out.update({k: np.nan for k in PROFILE_COLS})
            out.update({"txn_count_90d": 0, "num_payees_90d": 0, "is_salary_account": 0, "past_fraud_reports": 0})  # no history yet
            out["account_age_days"] = np.nan
            out["home_city"] = req.get("txn_city")
        # device registry
        dev = req.get("device_id")
        if req["channel"] == "CARD_POS" or not dev:
            out.update({"is_new_device": np.nan, "device_age_days": np.nan, "accounts_on_device": np.nan})
        elif dev in self.devices.index:
            dv = self.devices.loc[dev]
            gap = max((now - pd.Timestamp(dv["as_of"])).days, 0)
            age = float(dv["device_age_days"]) + gap if pd.notna(dv["device_age_days"]) else np.nan
            out.update({"is_new_device": int(age < 1) if age == age else 0, "device_age_days": age,
                        "accounts_on_device": dv["accounts_on_device"]})
        else:                                           # never seen this phone before
            out.update({"is_new_device": 1, "device_age_days": 0.0, "accounts_on_device": 1})
        # payee & mule store
        pid = req["payee_id"]
        if pid in self.payees.index:
            pv = self.payees.loc[pid]
            gap = max((now - pd.Timestamp(pv["as_of"])).days, 0)
            pay_age, m_age = pv["payee_account_age_days"], pv["merchant_age_days"]
            out.update({"payee_account_age_days": pay_age + gap if pd.notna(pay_age) else np.nan,
                        "payee_reports_count": pv["payee_reports_count"],
                        "merchant_age_days": m_age + gap if pd.notna(m_age) and m_age >= 0 else m_age,
                        "merchant_chargeback_rate": pv["merchant_chargeback_rate"]})
            if not req.get("merchant_category") and pd.notna(pv["merchant_category"]):
                req["merchant_category"] = pv["merchant_category"]
        else:                                           # brand-new payee account
            out.update({"payee_account_age_days": 0.0 if not pid.startswith("MER") else np.nan, "payee_reports_count": 0,
                        "merchant_age_days": np.nan, "merchant_chargeback_rate": np.nan})
        out["is_new_payee"] = int((req["customer_id"], pid) not in self.known_pairs)
        # live counters
        out.update(self.counters.read(req["customer_id"], pid, now))
        out.update({"failed_pin_attempts_24h": 0, "password_reset_last_24h": 0, "sim_changed_last_7d": 0})
        # calculated: distance between home city and payment city
        home, city = out.get("home_city"), req.get("txn_city")
        out["distance_from_home_km"] = 0.0 if home == city else self.city.get((home, city), np.nan)
        return out

    def update_after_payment(self, req: dict):
        """After scoring: update live counters and the customer's payee list."""
        now = pd.Timestamp(req["txn_timestamp"]).tz_localize(None)
        self.counters.add(req["customer_id"], req["payee_id"], now, float(req["amount"]))
        self.known_pairs.add((req["customer_id"], req["payee_id"]))
