# 🛡️ Real-Time Fraud Detection API

**Created by: Suresh D R | AI Product Developer & Technology Mentor**

Production code for the fraud-detection project (7 notebooks → API → Docker → Kubernetes → CI/CD → drift monitoring → automatic retraining).

* **API** (FastAPI): one payment in, `fraud_score` + `risk_band` (APPROVE / CONFIRM_WITH_CUSTOMER / REVIEW / BLOCK) + plain-language reasons out, in ≈ 30 ms
* **Same feature code** for training and serving (`src/features.py`) — tested to match Notebook 6 exactly
* **Monitoring**: every prediction logged to S3 → weekly **Evidently AI** drift + performance check
* **Retraining**: drift → SageMaker runs `sagemaker/retrain_pipeline.ipynb` (latest GitHub code + new S3 data) → champion vs challenger → promote only if better → pods reload automatically

👉 Start here: **[HOW_TO_RUN.md](HOW_TO_RUN.md)** — step by step from the virtual environment to AWS.

```
Bank ──JSON──► /score ─► validate ─► feature store ─► preprocess ─► features ─► LightGBM ─► band + SHAP reasons
                                                                                     │
                                                         S3 logs ◄───────────────────┘
                                                            │ weekly (K8s CronJob)
                                                  Evidently drift check ──► SNS e-mail
                                                            │ drift
                                                  SageMaker retraining notebook ──► new model in S3 ──► API reloads
```
