"""
download_artifacts.py — copies what the API and the tests need from S3 into a local folder (same layout).
Used by CI (GitHub Actions) and for offline testing:   python -m scripts.download_artifacts --dest local_s3
"""
import argparse
from pathlib import Path

import boto3

from src import config

PREFIXES = ["models/best_model.pkl", "models/model_config.json", "artifacts/", "feature_store/", "monitoring/"]
DATA = ["data/02_cleaned.csv", "data/06_features.csv"]          # only for the Notebook-6 parity test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="local_s3")
    ap.add_argument("--no-data", action="store_true", help="skip the training CSVs (faster; parity test is skipped)")
    a = ap.parse_args()
    s3 = boto3.client("s3", region_name=config.AWS_REGION)
    keys = []
    for p in PREFIXES + ([] if a.no_data else DATA):
        resp = s3.list_objects_v2(Bucket=config.S3_BUCKET, Prefix=p)
        keys += [o["Key"] for o in resp.get("Contents", [])]
    for k in keys:
        out = Path(a.dest) / k
        out.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(config.S3_BUCKET, k, str(out))
        print(f"  ⬇ s3://{config.S3_BUCKET}/{k}")
    print(f"✅ {len(keys)} files → {a.dest}/")


if __name__ == "__main__":
    main()
