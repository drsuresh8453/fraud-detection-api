"""
storage.py — read and write files in the S3 bucket (or a local folder with the same layout).

Every other module calls these functions, so switching between S3 and local testing is ONE setting
(STORAGE_MODE) — the code itself never changes.
"""
import io
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from src import config

_s3 = None


def _client():
    global _s3
    if _s3 is None:
        import boto3  # keys come from env vars / IAM role, never from code
        _s3 = boto3.client("s3", region_name=config.AWS_REGION)
    return _s3


def _local(key):
    p = Path(config.LOCAL_STORAGE_DIR) / key
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── bytes ──────────────────────────────────────────────────────────────────────────────────
def read_bytes(key: str) -> bytes:
    if config.STORAGE_MODE == "local":
        return _local(key).read_bytes()
    return _client().get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()


def write_bytes(key: str, data: bytes) -> None:
    if config.STORAGE_MODE == "local":
        _local(key).write_bytes(data)
    else:
        _client().put_object(Bucket=config.S3_BUCKET, Key=key, Body=data)


def exists(key: str) -> bool:
    if config.STORAGE_MODE == "local":
        return _local(key).exists()
    try:
        _client().head_object(Bucket=config.S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def list_keys(prefix: str) -> list:
    if config.STORAGE_MODE == "local":
        root = Path(config.LOCAL_STORAGE_DIR)
        base = root / prefix
        if not base.exists() and not base.parent.exists():
            return []
        search = base if base.is_dir() else base.parent
        return sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in search.rglob("*")
                      if p.is_file() and str(p.relative_to(root)).replace(os.sep, "/").startswith(prefix))
    keys, token = [], None
    while True:
        kw = {"Bucket": config.S3_BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = _client().list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if not resp.get("IsTruncated"):
            return sorted(keys)
        token = resp["NextContinuationToken"]


def copy(src_key: str, dst_key: str) -> None:
    write_bytes(dst_key, read_bytes(src_key))


# ── typed helpers ──────────────────────────────────────────────────────────────────────────
def read_json(key):
    return json.loads(read_bytes(key))


def write_json(obj, key):
    write_bytes(key, json.dumps(obj, indent=2, default=str).encode())


def read_pickle(key):
    return joblib.load(io.BytesIO(read_bytes(key)))


def write_pickle(obj, key):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    write_bytes(key, buf.getvalue())


def read_csv(key, **kw):
    return pd.read_csv(io.BytesIO(read_bytes(key)), **kw)


def write_csv(df, key):
    write_bytes(key, df.to_csv(index=False).encode())


def read_parquet(key):
    return pd.read_parquet(io.BytesIO(read_bytes(key)))


def write_parquet(df, key):
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    write_bytes(key, buf.getvalue())


def read_table(key, **kw):
    """Read csv / parquet / xlsx by file extension."""
    if key.endswith(".parquet"):
        return read_parquet(key)
    if key.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(read_bytes(key)))
    return read_csv(key, low_memory=False, **kw)
