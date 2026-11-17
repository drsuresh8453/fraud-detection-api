"""
trigger_retrain.py — what happens when drift is detected.

  notify()            → e-mail alert through AWS SNS (topic SNS_TOPIC_ARN)
  start_retraining()  → starts a SageMaker Processing job that RUNS THE RETRAINING NOTEBOOK
                        (sagemaker/retrain_pipeline.ipynb, executed with papermill):
                          • pulls the LATEST code from GitHub (GIT_REPO_URL)
                          • reads history + new data from S3
                          • trains, compares with the current model, promotes if better
                          • saves the executed notebook (with all outputs) to s3://<bucket>/reports/retraining/<job>/
                        SageMaker uses its IAM ROLE — no access keys are passed anywhere.

Usage:  python -m monitoring.trigger_retrain --dry-run      # print the SageMaker request, start nothing
        python -m monitoring.trigger_retrain                # start the job now (manual retraining)
"""
import argparse
import json
from datetime import datetime, timezone

from src import config


def notify(summary: dict):
    msg = (f"Fraud model {summary.get('model_version')} — drift check {summary.get('date')}\n"
           f"Reasons: {'; '.join(summary.get('reasons', []))}\n"
           f"Drifted features: {summary.get('drifted_features')}\nPerformance: {summary.get('performance')}\n"
           f"Report: s3://{config.S3_BUCKET}/{summary.get('reports', {}).get('data_drift')}")
    if not config.SNS_TOPIC_ARN:
        print("📧 (SNS_TOPIC_ARN not set — alert printed instead)\n" + msg)
        return
    import boto3
    boto3.client("sns", region_name=config.AWS_REGION).publish(TopicArn=config.SNS_TOPIC_ARN,
                                                                Subject="🚨 Fraud model drift detected", Message=msg)
    print("📧 SNS alert sent")


def build_job_request(drift_report_key: str = "") -> dict:
    job = f"fraud-retrain-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_nb = f"s3://{config.S3_BUCKET}/reports/retraining/{job}/retrain_executed.ipynb"
    code = ('rm -rf /tmp/code && git clone --depth 1 "$GIT_REPO_URL" /tmp/code && cd /tmp/code' if config.GIT_REPO_URL else "cd /app")
    cmd = (f"set -e; {code}; "
           f"papermill sagemaker/retrain_pipeline.ipynb {out_nb} -p drift_report '{drift_report_key}' -p promote True --log-output")
    return {
        "ProcessingJobName": job,
        "RoleArn": config.SAGEMAKER_ROLE_ARN,
        "AppSpecification": {"ImageUri": config.SAGEMAKER_IMAGE_URI, "ContainerEntrypoint": ["bash", "-c", cmd]},
        "ProcessingResources": {"ClusterConfig": {"InstanceCount": 1, "InstanceType": config.SAGEMAKER_INSTANCE, "VolumeSizeInGB": 30}},
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "Environment": {k: v for k, v in {"STORAGE_MODE": "s3", "S3_BUCKET": config.S3_BUCKET, "AWS_REGION": config.AWS_REGION,
                                          "GIT_REPO_URL": config.GIT_REPO_URL, "GITHUB_REPO": config.GITHUB_REPO,
                                          "GITHUB_TOKEN_SECRET": config.GITHUB_TOKEN_SECRET,
                                          "SNS_TOPIC_ARN": config.SNS_TOPIC_ARN}.items() if v},   # SageMaker rejects empty values
        "Tags": [{"Key": "project", "Value": "fraud-detection"}],
    }


def start_retraining(drift_report_key: str = "", dry_run: bool = False) -> dict:
    req = build_job_request(drift_report_key)
    if dry_run or not (config.SAGEMAKER_ROLE_ARN and config.SAGEMAKER_IMAGE_URI):
        print("🧪 SageMaker job request (not started — dry run or SAGEMAKER_ROLE_ARN / SAGEMAKER_IMAGE_URI not set):")
        print(json.dumps(req, indent=2))
        return {"started": False, "job": req["ProcessingJobName"]}
    import boto3
    boto3.client("sagemaker", region_name=config.AWS_REGION).create_processing_job(**req)
    print(f"🚀 SageMaker retraining started: {req['ProcessingJobName']}  "
          f"(AWS Console → SageMaker → Processing jobs; executed notebook → reports/retraining/{req['ProcessingJobName']}/)")
    return {"started": True, "job": req["ProcessingJobName"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--drift-report", default=config.DRIFT_REPORT_PREFIX + "latest.json")
    a = ap.parse_args()
    start_retraining(a.drift_report, dry_run=a.dry_run)
