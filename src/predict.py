"""
predict.py — the INFERENCE PIPELINE for one payment (Notebook 7 settings).

  ① request (validated by pydantic)            validate.py
  ② fetch customer / device / payee / counters  feature_store.py
  ③ same repairs + fill values as training       preprocess.py
  ④ 71 features in the training order            features.py
  ⑤ LightGBM score → threshold → risk band       model_config.json
  ⑥ top reasons (SHAP) in plain words
"""
import threading
import time
import warnings

import numpy as np
import pandas as pd

from src import config, storage
from src.feature_store import FeatureStore
from src.features import build_features, unseen_categories
from src.preprocess import apply_consistency_repairs, fill_missing
from src.validate import ACTIONS

warnings.filterwarnings("ignore", message=".*LightGBM binary classifier.*")

SENT_FIELDS = ["txn_id", "txn_timestamp", "customer_id", "amount", "channel", "txn_type", "payee_id", "merchant_category",
               "remarks", "device_id", "device_os", "ip_country", "txn_city", "is_international", "is_rooted_or_emulator",
               "vpn_or_proxy", "login_to_pay_seconds"]

# Plain-language reasons (what the bank's app / analyst sees)
REASON_TEXT = {
    "payee_reports_count": lambda v: f"Receiver was reported for fraud by {v:.0f} customer(s)",
    "payee_handle_risk": lambda v: f"Receiver's UPI app/handle has a high fraud history ({v:.1%})",
    "payee_is_new_account_30d": lambda v: "Receiver's account is less than 30 days old" if v else "",
    "payee_account_lt_1y": lambda v: "Receiver's account is less than 1 year old" if v else "",
    "payee_incoming_count_24h": lambda v: f"Receiver got {v:.0f} payments in the last 24 h (possible mule account)",
    "is_new_payee": lambda v: "First payment to this receiver" if v else "",
    "is_new_device": lambda v: "Payment from a new device" if v else "",
    "device_linked_lt_7d": lambda v: "Device linked less than 7 days ago" if v else "",
    "device_age_days": lambda v: f"Device linked only {v:.0f} day(s) ago",
    "accounts_on_device": lambda v: f"{v:.0f} different customers used this device",
    "sim_changed_last_7d": lambda v: "SIM card changed in the last 7 days" if v else "",
    "password_reset_last_24h": lambda v: "Password reset in the last 24 hours" if v else "",
    "failed_pin_attempts_24h": lambda v: f"{v:.0f} wrong PIN attempt(s) in the last 24 h",
    "is_rooted_or_emulator": lambda v: "Rooted phone / emulator detected" if v else "",
    "vpn_or_proxy": lambda v: "VPN / proxy in use" if v else "",
    "is_foreign_ip": lambda v: "Internet connection from outside India" if v else "",
    "is_international": lambda v: "International payment" if v else "",
    "distance_from_home_km": lambda v: f"Payment {v:,.0f} km from the customer's home city",
    "amount_vs_usual": lambda v: f"Amount is {v:.1f}× the customer's usual payment",
    "log_amount": lambda v: f"Amount ₹{np.expm1(v):,.0f} is in a high-risk range",
    "amount_sum_24h": lambda v: f"₹{v:,.0f} already sent in the last 24 h",
    "txn_count_1h": lambda v: f"{v:.0f} payments in the last hour",
    "txn_count_24h": lambda v: f"{v:.0f} payments in the last 24 h",
    "seconds_since_last_txn": lambda v: f"Only {v:.0f} s since the previous payment",
    "login_to_pay_seconds": lambda v: f"Paid {v:.0f} s after login (unusual speed)",
    "is_night": lambda v: "Night-time payment (00:00–05:00)" if v else "",
    "night_and_high_amount": lambda v: "Night-time AND much higher than usual" if v else "",
    "new_device_and_new_payee": lambda v: "New device AND new receiver together" if v else "",
    "remarks_scam_word": lambda v: "Payment note contains a typical scam word" if v else "",
    "is_round_amount": lambda v: "Round amount (typical of scams)" if v else "",
    "merchant_chargeback_rate": lambda v: f"Merchant has a high dispute rate ({v:.1%})",
    "merchant_age_days": lambda v: f"Merchant onboarded only {v:.0f} days ago",
    "account_is_new_90d": lambda v: "Customer account opened less than 90 days ago" if v else "",
    "account_age_days": lambda v: f"Customer account is {v:.0f} days old",
}


ONEHOT_TEXT = {"channel_": "Channel", "txn_type_": "Payment type", "device_os_": "Device", "merchant_category_": "Merchant category",
               "occupation_": "Customer occupation"}


def reason_text(feature: str, value: float) -> str:
    """Plain words for one feature; '' when the feature is a 0/1 flag that is OFF (not a reason)."""
    for prefix, label in ONEHOT_TEXT.items():
        if feature.startswith(prefix):
            return f"{label}: {feature[len(prefix):]}" if value == 1 else ""
    if feature in REASON_TEXT:
        return REASON_TEXT[feature](value)
    return f"{feature} = {value:g}"


class FraudModel:
    """Loads everything ONCE at start-up; scores many payments; reloads when a newer model appears in S3."""

    def __init__(self):
        self.lock = threading.Lock()
        self.store = FeatureStore()
        self.version = None

    # ── loading ──────────────────────────────────────────────────────────────────────────
    def load(self):
        cfg = storage.read_json(config.MODEL_CONFIG_KEY)
        model = storage.read_pickle(config.MODEL_KEY)
        enc = storage.read_pickle(config.ENCODERS_KEY)
        fill = storage.read_json(config.IMPUTE_KEY)
        features = cfg.get("features") or storage.read_json(config.FEATURE_LIST_KEY)["features"]
        explainer = None
        try:
            import shap
            explainer = shap.TreeExplainer(model[-1] if hasattr(model, "steps") else model)
        except Exception as e:  # model without tree explainer → reasons fall back to feature values
            print("SHAP explainer not available:", e)
        with self.lock:
            self.model, self.cfg, self.enc, self.fill, self.features, self.explainer = model, cfg, enc, fill, features, explainer
            self.threshold = float(cfg["threshold"])
            self.bands = cfg["bands"]
            self.version = str(cfg.get("model_version", "v1.0"))
        self.store.load()            # new model → also fresh feature-store tables (live counters are kept)
        print(f"✅ Model {cfg.get('model_name')} {self.version} loaded | threshold {self.threshold} | {len(features)} features")
        return self

    def check_for_new_model(self, force: bool = False) -> bool:
        """Called every few minutes: if model_config.json in S3 has a new version, reload (force=True: always reload)."""
        try:
            v = str(storage.read_json(config.MODEL_CONFIG_KEY).get("model_version"))
            if force or v != self.version:
                print(f"🔄 New model version found in storage: {self.version} → {v}")
                self.load()
                return True
        except Exception as e:
            print("model check failed:", e)
        return False

    # ── pipeline ─────────────────────────────────────────────────────────────────────────
    def band(self, score: float) -> str:
        for name, (lo, hi) in self.bands.items():
            if lo <= score < hi:
                return name
        return "BLOCK"

    def prepare(self, reqs: list):
        """Steps ②–④ for a list of request dicts → (feature table, raw table, warnings per payment)."""
        rows, warns = [], []
        for r in reqs:
            r = dict(r)
            fetched = self.store.lookup(r)
            source = fetched.pop("_source")
            row = {k: r.get(k) for k in SENT_FIELDS}
            row.update(fetched)
            rows.append(row)
            warns.append([] if source == "snapshot" else ["live lookup (new payment)"])
        raw = pd.DataFrame(rows)
        raw["txn_timestamp"] = pd.to_datetime(raw["txn_timestamp"]).dt.tz_localize(None)
        num = [c for c in raw.columns if c not in ("txn_id", "customer_id", "channel", "txn_type", "payee_id", "merchant_category",
                                                    "remarks", "device_id", "device_os", "ip_country", "txn_city", "home_city",
                                                    "occupation", "kyc_level", "txn_timestamp")]
        raw[num] = raw[num].apply(pd.to_numeric, errors="coerce")
        raw = apply_consistency_repairs(raw)
        clean = fill_missing(raw, self.fill)
        with self.lock:
            enc, feats = self.enc, self.features
        X = build_features(clean, enc, feats)
        unseen = unseen_categories(clean, enc)
        if unseen:
            warns = [w + [f"value never seen in training: {', '.join(unseen)}"] for w in warns]
        return X, clean, warns

    def score(self, reqs: list, explain=True, update_store=True):
        t0 = time.perf_counter()
        X, clean, warns = self.prepare(reqs)
        with self.lock:
            model, explainer, version = self.model, self.explainer, self.version
        p = model.predict_proba(X)[:, 1]
        reasons = [[] for _ in range(len(X))]
        if explain and explainer is not None:
            sv = explainer.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            for i in range(len(X)):
                for j in np.argsort(-sv[i]):
                    if sv[i, j] <= 0 or len(reasons[i]) == config.TOP_REASONS:
                        break
                    txt = reason_text(self.features[j], X.iat[i, j])
                    if txt and txt not in reasons[i]:
                        reasons[i].append(txt)
                if p[i] < self.threshold:
                    reasons[i] = ["No strong risk signals — looks like the customer's normal behaviour"]
        ms = (time.perf_counter() - t0) * 1000 / max(len(X), 1)
        out = []
        for i, r in enumerate(reqs):
            b = self.band(float(p[i]))
            out.append({"txn_id": r["txn_id"], "fraud_score": round(float(p[i]), 5), "risk_band": b, "action": ACTIONS[b],
                        "reasons": reasons[i], "model_version": version, "latency_ms": round(ms, 2), "warnings": warns[i]})
            if update_store:
                self.store.update_after_payment(r)
        return out, X
