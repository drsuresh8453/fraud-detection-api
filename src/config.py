"""
config.py — every setting of the project in ONE place.

Values come from environment variables, so the same code runs on your laptop, in Docker, in Kubernetes
and in SageMaker without changes. NO secrets are written here: AWS keys come from the environment
(your terminal, a Kubernetes Secret or an IAM role), never from code.
"""
import os


def _env(name, default):
    return os.getenv(name, default)


# ── Storage ────────────────────────────────────────────────────────────────────────────────
# "s3"    → read/write the real bucket (needs AWS keys or an IAM role)
# "local" → read/write a folder that has the SAME layout as the bucket (for testing without AWS)
STORAGE_MODE = _env("STORAGE_MODE", "s3")
S3_BUCKET = _env("S3_BUCKET", "fraud-detection-project-2026-suresh")
AWS_REGION = _env("AWS_REGION", "ap-south-1")
LOCAL_STORAGE_DIR = _env("LOCAL_STORAGE_DIR", "./local_s3")

# ── Files inside the bucket (all created by the notebooks or by scripts/prepare_artifacts.py) ──
MODEL_KEY = "models/best_model.pkl"
MODEL_CONFIG_KEY = "models/model_config.json"
FEATURE_LIST_KEY = "artifacts/feature_list.json"
ENCODERS_KEY = "artifacts/encoders.pkl"
IMPUTE_KEY = "artifacts/impute_values.json"
FEATURE_STORE_PREFIX = "feature_store/"                 # customers, devices, payees, snapshots, city distances
REFERENCE_KEY = "monitoring/reference_features.parquet"  # "what normal looks like" for drift detection
PREDICTION_LOG_PREFIX = "logs/predictions/"              # every scored payment (features + score)
DRIFT_REPORT_PREFIX = "reports/drift/"
NEW_DATA_PREFIX = "new_data/"                            # new labelled monthly exports from the bank
RAW_DATA_KEY = "raw/fraud_transactions_raw_100k.xlsx"
MODEL_HISTORY_PREFIX = "models/history/"

# ── API behaviour ──────────────────────────────────────────────────────────────────────────
MODEL_RELOAD_SECONDS = int(_env("MODEL_RELOAD_SECONDS", "300"))   # check S3 for a newer model every 5 min
LOG_PREDICTIONS = _env("LOG_PREDICTIONS", "true").lower() == "true"
LOG_FLUSH_EVERY = int(_env("LOG_FLUSH_EVERY", "200"))             # write logs to S3 every 200 payments …
LOG_FLUSH_SECONDS = int(_env("LOG_FLUSH_SECONDS", "60"))          # … or every 60 seconds
TOP_REASONS = 3
LATENCY_LIMIT_MS = float(_env("LATENCY_LIMIT_MS", "50"))           # CI fails if scoring is slower than this

# ── Drift rules (Phase 9 of the plan) ──────────────────────────────────────────────────────
DRIFT_WINDOW_DAYS = int(_env("DRIFT_WINDOW_DAYS", "7"))
PSI_LIMIT = float(_env("PSI_LIMIT", "0.25"))                       # big change in one feature
DRIFT_SHARE_LIMIT = float(_env("DRIFT_SHARE_LIMIT", "0.30"))       # > 30 % of features drifted
ALERT_RATE_LIMIT = float(_env("ALERT_RATE_LIMIT", "0.05"))         # normal ≈ 3 %
RECALL_LIMIT = float(_env("RECALL_LIMIT", "0.90"))                 # on mature labels
MIN_ROWS_FOR_DRIFT = int(_env("MIN_ROWS_FOR_DRIFT", "500"))
RETRAIN_COOLDOWN_DAYS = int(_env("RETRAIN_COOLDOWN_DAYS", "14"))   # don't start a new SageMaker job if one ran recently
TOP_FEATURES_FOR_DRIFT = 10

# ── Training (same settings as Notebook 7) ─────────────────────────────────────────────────
LABEL_DELAY_DAYS = 90                  # a genuine label is final only after 90 days without complaint
# Notebook 7 used 3 + 3 months for a careful FIRST evaluation. Monthly retraining uses the newest 1 month for
# validation and 1 month for testing, so a new scam pattern reaches the TRAINING data as quickly as possible.
VALID_MONTHS = int(_env("VALID_MONTHS", "1"))
TEST_MONTHS = int(_env("TEST_MONTHS", "1"))
FALSE_ALARM_COST = 150
ALERT_CAP = 0.03
LEAKAGE_COLUMNS = ["fraud_type", "label_source", "label_date", "attack_id"]

# ── Automatic retraining (SageMaker) and alerts ────────────────────────────────────────────
SAGEMAKER_ROLE_ARN = _env("SAGEMAKER_ROLE_ARN", "")
SAGEMAKER_IMAGE_URI = _env("SAGEMAKER_IMAGE_URI", "")              # the same Docker image, pushed to ECR
SAGEMAKER_INSTANCE = _env("SAGEMAKER_INSTANCE", "ml.m5.xlarge")
GIT_REPO_URL = _env("GIT_REPO_URL", "")                            # https://github.com/<you>/fraud-detection-api.git
SNS_TOPIC_ARN = _env("SNS_TOPIC_ARN", "")                          # email alerts
GITHUB_REPO = _env("GITHUB_REPO", "")                              # <you>/fraud-detection-api (for repository_dispatch)
GITHUB_TOKEN_SECRET = _env("GITHUB_TOKEN_SECRET", "fraud-api/github-token")   # name in AWS Secrets Manager
