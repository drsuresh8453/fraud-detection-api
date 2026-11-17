"""
validate.py — checks ONE live payment request before scoring (Notebook 2, Step 11).

The bank sends only what it knows at payment time (~17 fields). Pydantic checks types and formats
automatically (wrong format → HTTP 422 with a clear message). business_rules() adds the cross-field
rules from Notebook 2, Step 8.
"""
from datetime import datetime, timedelta
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Channel = Literal["UPI", "CARD_POS", "CARD_ONLINE", "NETBANKING"]
TxnType = Literal["P2P_TRANSFER", "P2M_PAYMENT", "BILL_PAYMENT", "COLLECT_REQUEST", "CARD_PURCHASE"]


class PaymentRequest(BaseModel):
    """What the bank sends for every payment."""
    txn_id: str = Field(..., pattern=r"^TXN\d{5,12}$", examples=["TXN30085307"])
    txn_timestamp: datetime = Field(..., examples=["2026-07-14T21:05:00"])
    customer_id: str = Field(..., pattern=r"^C\d{6}$", examples=["C100161"])
    amount: float = Field(..., gt=0, le=10_000_000, description="₹, must be > 0")
    channel: Channel
    txn_type: TxnType
    payee_id: str = Field(..., pattern=r"^([\w.\-]+@\w+|MER\d{5})$", examples=["quickcash642@paytm", "MER00412"])
    merchant_category: Optional[str] = Field(None, description="only for merchant payments")
    remarks: Optional[str] = Field("", max_length=200)
    device_id: Optional[str] = Field(None, description="empty for card-at-shop payments")
    device_os: Optional[Literal["Android", "iOS", "Emulator"]] = None
    ip_country: Optional[str] = Field(None, pattern=r"^[A-Z]{2}$")
    txn_city: str
    is_international: Optional[int] = Field(0, ge=0, le=1)
    is_rooted_or_emulator: Optional[int] = Field(0, ge=0, le=1)
    vpn_or_proxy: Optional[int] = Field(0, ge=0, le=1)
    login_to_pay_seconds: Optional[float] = Field(None, ge=0)
    # Optional security events — if the bank's login system already knows them it can send them;
    # otherwise the feature store provides them.
    failed_pin_attempts_24h: Optional[int] = Field(None, ge=0)
    password_reset_last_24h: Optional[int] = Field(None, ge=0, le=1)
    sim_changed_last_7d: Optional[int] = Field(None, ge=0, le=1)

    @field_validator("txn_city")
    @classmethod
    def city_title(cls, v):
        return v.strip().title()


class BatchRequest(BaseModel):
    payments: List[PaymentRequest] = Field(..., max_length=1000)


class ScoreResponse(BaseModel):
    txn_id: str
    fraud_score: float
    risk_band: str
    action: str
    reasons: List[str]
    model_version: str
    latency_ms: float
    warnings: List[str] = []


ACTIONS = {"APPROVE": "Approve silently",
           "CONFIRM_WITH_CUSTOMER": "Ask the customer to confirm in the app before paying",
           "REVIEW": "Hold the payment and send to the fraud team",
           "BLOCK": "Block the payment and alert the customer"}


def business_rules(req: PaymentRequest):
    """Cross-field rules (Notebook 2, Step 8). Returns (errors, warnings)."""
    errors, warnings = [], []
    if req.channel == "CARD_POS" and req.is_international == 1:
        errors.append("international flag on a card swiped in an Indian shop")
    if req.txn_timestamp.replace(tzinfo=None) > datetime.now() + timedelta(minutes=10):
        errors.append("payment timestamp is in the future")
    if req.channel == "CARD_POS" and req.device_id:
        warnings.append("device_id ignored for a card-at-shop payment")
    if req.txn_type in ("P2M_PAYMENT", "CARD_PURCHASE") and not req.merchant_category:
        warnings.append("merchant payment without merchant_category — fetched from the merchant store")
    if req.txn_type == "P2P_TRANSFER" and req.payee_id.startswith("MER"):
        warnings.append("P2P_TRANSFER to a merchant ID — treated as P2M_PAYMENT (Notebook 2 rule)")
    return errors, warnings
