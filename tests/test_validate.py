import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.validate import PaymentRequest, business_rules

SAMPLE = json.loads((Path(__file__).parent.parent / "client" / "sample_requests.json").read_text())["1_genuine_upi_payment"]


def test_valid_request_passes():
    req = PaymentRequest(**SAMPLE)
    assert business_rules(req)[0] == []


@pytest.mark.parametrize("field,value", [("amount", 0), ("amount", -50), ("channel", "BITCOIN"), ("customer_id", "X1"),
                                         ("payee_id", "not a upi id"), ("txn_type", "LOAN")])
def test_bad_values_rejected(field, value):
    with pytest.raises(ValidationError):
        PaymentRequest(**{**SAMPLE, field: value})


def test_card_in_shop_cannot_be_international():
    req = PaymentRequest(**{**SAMPLE, "channel": "CARD_POS", "is_international": 1})
    errors, _ = business_rules(req)
    assert errors


def test_future_timestamp_rejected():
    req = PaymentRequest(**{**SAMPLE, "txn_timestamp": "2099-01-01T10:00:00"})
    assert business_rules(req)[0]
