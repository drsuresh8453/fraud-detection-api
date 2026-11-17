"""Tests run against a LOCAL copy of the bucket (scripts/download_artifacts.py fills ./local_s3 from S3)."""
import os

os.environ.setdefault("STORAGE_MODE", "local")
os.environ.setdefault("LOCAL_STORAGE_DIR", "./local_s3")
os.environ.setdefault("LOG_PREDICTIONS", "false")
