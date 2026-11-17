"""The most important test: the API builds EXACTLY the same features as Notebook 6 (no training/serving skew)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.predict import SENT_FIELDS, FraudModel

DATA = Path(config.LOCAL_STORAGE_DIR) / "data"


@pytest.fixture(scope="module")
def model():
    return FraudModel().load()


@pytest.mark.skipif(not (DATA / "06_features.csv").exists(), reason="training data not downloaded")
def test_features_match_notebook6(model):
    if model.cfg.get("retrained"):
        pytest.skip("model was retrained — features now differ from Notebook 6 on purpose (new fill values / encoders)")
    clean = pd.read_csv(DATA / "02_cleaned.csv", low_memory=False).sample(2000, random_state=0)
    nb6 = pd.read_csv(DATA / "06_features.csv", low_memory=False).set_index("txn_id")
    X, _, _ = model.prepare(clean[SENT_FIELDS].replace({np.nan: None}).to_dict("records"))
    X.index = clean["txn_id"].values
    diff = (X - nb6.loc[X.index, model.features].astype(float)).abs()
    assert (diff > 1e-6).sum().sum() == 0, diff.max()[diff.max() > 1e-6]


def test_feature_order_and_count(model):
    assert model.features == model.cfg["features"]
    assert len(model.features) == 71


def test_unknown_customer_and_device_still_scored(model):
    req = {"txn_id": "TXN99999999", "txn_timestamp": "2026-11-10T03:00:00", "customer_id": "C999999", "amount": 25000.0,
           "channel": "UPI", "txn_type": "P2P_TRANSFER", "payee_id": "brandnew123@ybl", "merchant_category": None,
           "remarks": "", "device_id": "D-UNKNOWN", "device_os": "Android", "ip_country": "IN", "txn_city": "Pune",
           "is_international": 0, "is_rooted_or_emulator": 0, "vpn_or_proxy": 0, "login_to_pay_seconds": 20.0}
    out, X = model.score([req], update_store=False)
    assert 0 <= out[0]["fraud_score"] <= 1
    assert not X.isna().any().any()
