# 🚀 HOW TO RUN — Real-Time Fraud Detection API (end to end)

**Created by: Suresh D R | AI Product Developer & Technology Mentor**

This guide takes you from the 7 notebooks to a production system on AWS. Follow the parts in order:

| Part | What you do | Where |
|---|---|---|
| 0 | Understand the project and the inference pipeline | — |
| 1 | Install the tools (Python, Git, Docker, AWS CLI, kubectl, minikube, eksctl, Postman) | your laptop |
| 2 | Create the virtual environment | your laptop |
| 3 | Connect to AWS (IAM user, keys, bucket check) | AWS Console |
| 4 | Prepare the production artifacts (feature store + drift reference) | laptop → S3 |
| 5 | Run the API locally | laptop |
| 6 | Test with **Postman** | laptop |
| 7 | Test with the **Python client** (sends JSON and reads the results) | laptop |
| 8 | Automated tests | laptop |
| 9 | **Docker** — build and run the image | laptop |
| 10 | Push the code to **GitHub** | GitHub |
| 11 | **Kubernetes locally** with minikube | laptop |
| 12 | **AWS**: ECR + EKS (Kubernetes in the cloud) | AWS |
| 13 | **CI/CD** with GitHub Actions | GitHub → AWS |
| 14 | **Data drift** monitoring with Evidently AI | Kubernetes CronJob |
| 15 | **Automatic retraining** in SageMaker | SageMaker |
| 16 | Full end-to-end test of the drift → retrain → redeploy loop | everything |
| 17 | Clean up (to avoid AWS bills) | AWS |
| 18 | Troubleshooting | — |

> 🐚 **All commands are for Git Bash on Windows** (they also work unchanged in a Mac/Linux terminal).
> Open Git Bash **inside the project folder** (right-click the folder → *Open Git Bash here*, or VS Code → Terminal → ▾ → Git Bash).
>
> Git Bash rules to remember:
>
> * Use `python` (or `py -3.13`), **not** `python3` — on Windows `python3` may open the Microsoft Store.
> * Activate the virtual environment with `source venv/Scripts/activate` (Mac/Linux: `source venv/bin/activate`).
> * Set a variable for the current terminal with `export NAME=value` (it is gone when you close the terminal).
> * If an interactive command hangs (no prompt appears), put `winpty` in front of it, e.g. `winpty aws configure`.
> * Your numbers can differ a little from the examples below — your Colab notebook run produced its own model and threshold.

---

## PART 0 — What you are building

### 0.1 Folder structure
```
fraud-detection-api/
├── src/
│   ├── config.py          all settings (from environment variables — no keys in code)
│   ├── storage.py         read/write S3 (or a local folder with the same layout, for testing)
│   ├── validate.py        NB2: checks every request (types, formats, business rules)
│   ├── preprocess.py      NB2 + NB3: cleaning rules and fill values
│   ├── feature_store.py   fetches what the bank does NOT send (customer, device, payee, live counters)
│   ├── features.py        NB6: builds the 71 features — the SAME code for training and for the API
│   ├── predict.py         NB7: the inference pipeline → score → band → reasons (SHAP)
│   ├── train.py           automatic retraining: champion vs challenger
│   └── api.py             FastAPI app: /health /model-info /score /score/batch /admin/reload
├── monitoring/
│   ├── log_predictions.py every scored payment → S3 logs (needed to detect drift)
│   ├── detect_drift.py    Evidently AI drift + performance check → decision
│   └── trigger_retrain.py SNS e-mail + starts the SageMaker retraining notebook
├── sagemaker/retrain_pipeline.ipynb   the notebook SageMaker runs when drift is found
├── scripts/
│   ├── prepare_artifacts.py      run ONCE after the notebooks (feature store + drift reference)
│   ├── make_client_samples.py    test requests for Postman / Python client
│   ├── download_artifacts.py     copies S3 files to a local folder (CI + offline testing)
│   ├── simulate_new_data.py      creates a drift situation to test the whole loop
│   └── build_retrain_notebook.py regenerates the SageMaker notebook
├── client/   test_api.py · postman_collection.json · sample_requests.json · sample_payments.csv
├── tests/    test_validate.py · test_features.py (parity with Notebook 6) · test_api.py (incl. latency)
├── k8s/      namespace · configmap · secret example · deployment · service · HPA · drift CronJob
├── .github/workflows/  ci-cd.yml · deploy-new-model.yml
├── iam/      least-privilege AWS policies
├── Dockerfile · requirements.txt · requirements-dev.txt · .gitignore · .env.example
```

### 0.2 Where each notebook went
| Notebook | Goes to production? | File |
|---|---|---|
| NB1 First look | ❌ (checks become data-quality alerts) | — |
| NB2 Cleaning | ✅ | `preprocess.clean_batch()` (retraining) + `validate.py` (live requests) |
| NB3 Missing values | ✅ | `preprocess.fill_missing()` + `artifacts/impute_values.json` |
| NB4 EDA | ❌ | — |
| NB5 Statistical tests | ⚠️ only the PSI drift test | `monitoring/detect_drift.py` |
| NB6 Features | ✅ the most important | `features.py` + `encoders.pkl` + `feature_list.json` |
| NB7 Model | ✅ | `predict.py`, `train.py`, `models/best_model.pkl`, `model_config.json` |

### 0.3 The inference pipeline (what happens to ONE payment, ~30 ms)
```
Bank sends ~17 fields (txn_id, time, customer_id, amount, channel, payee, device, IP, city, remarks …)
   │
   ① validate.py       wrong type / format / rule  → HTTP 422 with the reason (bank uses its own rules)
   ② feature_store.py  + customer profile (age, job, KYC, account age, habits)   [database]
                       + device registry (new device? how many customers on this phone?)       [database]
                       + payee & mule store (account age, fraud reports)                        [database]
                       + live counters (payments in last 1 h / 24 h, ₹ in 24 h, seconds since last) [Redis]
                       + calculated (distance from home city)
   ③ preprocess.py     same repairs + SAVED fill values as training (impute_values.json)
   ④ features.py       71 features in the exact training order (encoders.pkl, feature_list.json)
   ⑤ predict.py        LightGBM score → threshold → band: APPROVE / CONFIRM_WITH_CUSTOMER / REVIEW / BLOCK
   ⑥ SHAP              top 3 reasons in plain words ("SIM card changed in the last 7 days" …)
   ⑦ log               features + score → S3 logs/predictions/  (for drift)
   ⑧ JSON response     {fraud_score, risk_band, action, reasons, model_version, latency_ms, warnings}
```
✅ **Tested:** the API builds **exactly** the same 71 features as Notebook 6 for all 99,822 payments (0 differences).
That is what "no training/serving skew" means — and `tests/test_features.py` checks it on every code change.

### 0.4 The monitoring loop
```
API (every payment) ──logs──►  S3: logs/predictions/
      ▲                                 │  every Monday
      │                                 ▼
      │                 Evidently drift check (Kubernetes CronJob)
      │                                 │  drift found?
      │                   ┌─────────────┴──────────────┐
      │                   ▼                            ▼
      │              SNS e-mail          SageMaker runs retrain_pipeline.ipynb
      │                                  (latest code from GitHub + new data from S3)
      │                                  champion vs challenger → promote if better
      │                                                │
      └──── API pods reload the new model from S3 ◄────┘
            (every 5 minutes, or a GitHub Actions restart)
```

---

## PART 1 — Install the tools (once)

| Tool | Why | Windows | Mac | Check |
|---|---|---|---|---|
| **Python 3.11–3.13** | run the code | already have 3.11+? use it. Otherwise python.org → Downloads → 3.13 (tick **"Add python.exe to PATH"**) | `brew install python@3.13` | `python --version` |
| **Git + Git Bash** | version control + the terminal we use | git-scm.com → Download (keep defaults; choose *"Checkout as-is, commit Unix-style line endings"*) | `brew install git` | `git --version` |
| **VS Code** | editor | code.visualstudio.com | same | — |
| **Postman** | test the API by hand | postman.com/downloads | same | — |
| **Docker Desktop** | build/run containers | docker.com/products/docker-desktop (needs WSL 2 — the installer offers it) | same (Apple/Intel chip version) | `docker --version` then `docker run hello-world` |
| **AWS CLI v2** | talk to AWS | aws.amazon.com/cli → Windows installer (.msi) | `brew install awscli` | `aws --version` |
| **kubectl** | control Kubernetes | `winget install Kubernetes.kubectl` | `brew install kubectl` | `kubectl version --client` |
| **minikube** | Kubernetes on your laptop | `winget install Kubernetes.minikube` | `brew install minikube` | `minikube version` |
| **eksctl** | create the AWS Kubernetes cluster | `winget install eksctl` (or choco install eksctl) | `brew tap weaveworks/tap && brew install weaveworks/tap/eksctl` | `eksctl version` |

> Docker Desktop must be **running** (whale icon in the taskbar) before any `docker` or `minikube` command.

---

## PART 2 — Virtual environment

A virtual environment = a private box of Python packages for THIS project, so versions never clash with other projects.

Unzip the project, open the folder in VS Code, open a terminal (**Terminal → New Terminal**), then:

**Step 1 — check the Python version** (3.11, 3.12 or 3.13 all work; Docker uses 3.13):
```bash
cd fraud-detection-api
python --version                        # Python 3.11.x, 3.12.x or 3.13.x
```

**Step 2 — create and activate the virtual environment**
```bash
python -m venv venv                     # CREATE it — a folder called "venv" (run once)
source venv/Scripts/activate            # ACTIVATE it (Mac/Linux: source venv/bin/activate)
```
> If `python --version` shows 3.10 or older, install Python 3.13 from python.org (tick "Add python.exe to PATH"), reopen Git Bash and create it with `py -3.13 -m venv venv`.

**Step 3 — install the project's libraries**
```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt     # ≈ 3–5 minutes
python -c "import lightgbm, fastapi, evidently; print('✅ environment ready')"
```

| Command | What it does | When |
|---|---|---|
| `python -m venv venv` | creates the environment | **once** |
| `source venv/Scripts/activate` | switches it on — the prompt starts with `(venv)` | **every new Git Bash window** |
| `python …` / `pip …` | runs inside the environment | after activating |
| `deactivate` | switches it off | when you are done |

In VS Code: **Ctrl+Shift+P → Python: Select Interpreter → venv**. The `venv/` folder is already excluded by `.gitignore`
and `.dockerignore`, so it is never pushed to GitHub or copied into the Docker image.

---

## PART 3 — Connect to AWS

### 3.1 ⚠️ Security first
The access keys that were inside the earlier VMT notebooks are **exposed** → AWS Console → IAM → Users → (that user) →
Security credentials → **Deactivate and delete** them. Create new keys only for the user below. **Never** paste keys into
code, notebooks, Git, chats or screenshots.

### 3.2 Create a project IAM user (least privilege)
1. AWS Console → **IAM → Policies → Create policy → JSON** → paste `iam/fraud-app-policy.json` → name `FraudAppPolicy`.
2. **IAM → Users → Create user** → name `fraud-app` → *Attach policies directly* → `FraudAppPolicy`.
   (For Part 12 — creating the EKS cluster with eksctl — use your **admin** user; `fraud-app` is for the app and CI/CD.)
3. User → **Security credentials → Create access key** → *Command Line Interface* → download the .csv.

### 3.3 Configure the AWS CLI on your laptop
```bash
aws configure                      # if it hangs in Git Bash: winpty aws configure
#   AWS Access Key ID:     <from the csv>
#   AWS Secret Access Key: <from the csv>
#   Default region name:   ap-south-1
#   Default output format: json
aws sts get-caller-identity            # shows your account → keys work
aws s3 ls s3://fraud-detection-project-2026-suresh/ --recursive
```
The code finds the keys automatically (boto3 reads `aws configure`). You must see these files from the notebooks:
```
raw/fraud_transactions_raw_100k.xlsx
data/02_cleaned.csv  data/03_imputed.csv  data/05_hypothesis_done.csv  data/06_features.csv
artifacts/impute_values.json  artifacts/encoders.pkl  artifacts/feature_list.json
models/best_model.pkl  models/model_config.json
```
If some are missing → run the notebooks (1 → 7) in Colab first.

---

## PART 4 — Prepare the production artifacts (once)

```bash
python -m scripts.prepare_artifacts
```
What it does (reads the notebook outputs from S3, writes back to S3):
```
feature_store/customers.parquet        1,840 customer profiles
feature_store/devices.parquet          4,230 devices
feature_store/payees.parquet          11,052 payees (UPI IDs + merchants)
feature_store/customer_payees.parquet who has paid whom (→ "new payee")
feature_store/city_distance.parquet   distance between cities
feature_store/snapshots.parquet       point-in-time values for every past payment (for replay tests)
monitoring/reference_features.parquet "normal" for drift detection (training period, with model scores)
artifacts/impute_values.json          + login_overall
✅ Artifacts ready — the API can start now.
```
Then create the test requests:
```bash
python -m scripts.make_client_samples
#   ✅ client/sample_requests.json (6), client/sample_payments.csv (1000 rows), client/postman_collection.json
```
> The project zip already contains these three client files (made from the same data), so this step is optional.

---

## PART 5 — Run the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Expected:
```
✅ Model LightGBM v1.0 loaded | threshold 0.011 | 71 features
INFO:     Uvicorn running on http://0.0.0.0:8000
```
Open **http://localhost:8000/docs** → interactive documentation: click an endpoint → *Try it out* → *Execute*.
Leave this terminal running; open a **second terminal** (activate the venv again) for the next parts.

| Endpoint | Purpose |
|---|---|
| `GET /health` | is the service alive and the model loaded? (Kubernetes uses it) |
| `GET /model-info` | name, version, threshold, bands, test results, model card |
| `POST /score` | one payment → score, band, action, reasons |
| `POST /score/batch` | up to 1,000 payments |
| `POST /admin/reload` | reload model + feature store from S3 now |

---

## PART 6 — Test with Postman

1. Postman → **Import** → select `client/postman_collection.json` → collection **"Fraud Detection API"** appears.
2. Click the collection → **Variables** → `baseUrl` = `http://localhost:8000` → Save.
3. Open each request → **Send**. Example request body (account takeover):
```json
{
  "txn_id": "TXN30082620", "txn_timestamp": "2026-06-12T01:47:00", "customer_id": "C101234",
  "amount": 24300.0, "channel": "CARD_ONLINE", "txn_type": "CARD_PURCHASE", "payee_id": "MER00412",
  "merchant_category": "electronics", "remarks": "", "device_id": "D88812", "device_os": "Android",
  "ip_country": "IN", "txn_city": "Mumbai", "is_international": 0, "is_rooted_or_emulator": 0,
  "vpn_or_proxy": 0, "login_to_pay_seconds": 9.0
}
```
(the real values are in the collection). Expected response:
```json
{
  "txn_id": "…", "fraud_score": 0.9991, "risk_band": "BLOCK",
  "action": "Block the payment and alert the customer",
  "reasons": ["Amount ₹24,300 is in a high-risk range", "7 different customers used this device",
              "Password reset in the last 24 hours"],
  "model_version": "v1.0", "latency_ms": 29.9, "warnings": []
}
```
4. **Collection Runner** (▶ Run collection) runs all 10 requests with automatic checks (status 200 / 422, score present).

| Request | Expected band |
|---|---|
| 1 genuine UPI payment | APPROVE — "No strong risk signals" |
| 2 account takeover | BLOCK |
| 3 vishing / KYC scam | CONFIRM_WITH_CUSTOMER (score ≈ 0.26) |
| 4 SIM swap | BLOCK |
| 5 brand-new payment with takeover signals (live lookup) | CONFIRM_WITH_CUSTOMER / REVIEW |
| 6 invalid (amount 0, channel BITCOIN) | **HTTP 422** with the exact reasons |

---

## PART 7 — Test with the Python client (another program consuming the API)

```bash
python client/test_api.py                     # the 6 examples, printed nicely
python client/test_api.py --replay 1000       # + 1,000 real test-period payments through /score/batch
```
Expected (last lines):
```
REPLAY of 1,000 payments from sample_payments.csv in 4.0s (253 payments/s)
  fraud caught      : 194 of 198  (recall 98.0%)        ← same as Notebook 7 on the test months
  genuine disturbed : 9 of 802
  bands: {'APPROVE': 797, 'BLOCK': 166, 'CONFIRM_WITH_CUSTOMER': 23, 'REVIEW': 14}
```
How the client works: it reads payments from a CSV, turns each row into JSON, `requests.post(url + "/score/batch", json=…)`,
and compares the API's decision with the true label — exactly what the bank's payment switch would do (one JSON per payment).

Every scored payment is now in `s3://…/logs/predictions/date=YYYY-MM-DD/` (flushed every 200 payments or 60 s).

---

## PART 8 — Automated tests

Tests run against a **local copy** of the bucket (no AWS calls during the tests):
```bash
python -m scripts.download_artifacts --dest local_s3      # once (≈ 60 MB)
STORAGE_MODE=local LOCAL_STORAGE_DIR=./local_s3 python -m pytest      # the variables apply to this one command only
ruff check .
```
Expected: `20 passed`. What is tested:
* `test_validate.py` — bad amounts, unknown channel, wrong ID formats, card-in-shop "international", future dates → rejected
* `test_features.py` — **API features = Notebook 6 features** (2,000 payments), 71 features in order, unknown customer/device still scored
* `test_api.py` — every endpoint, genuine approved, takeover/SIM-swap blocked, invalid → 422, batch order, **latency limit**

> The API itself still uses S3 afterwards, because the variables were set only for the pytest command.

---

## PART 9 — Docker

### 9.1 Build
```bash
docker build -t fraud-api:latest .
docker images fraud-api              # ≈ 700 MB compressed
```
The image contains the code and libraries only — **no data, no model, no keys**. The model is loaded from S3 at start-up,
so a new model never needs a new image. The same image is used for the API, the drift CronJob and SageMaker retraining.

### 9.2 Run (keys passed at run time, never baked in)
```bash
cp .env.example .env
# edit .env in VS Code → put your fraud-app keys.  Bottom-right of VS Code must show "LF" (not "CRLF"),
# otherwise an invisible \r is added to the keys and AWS says "SignatureDoesNotMatch".
docker run -d --name fraud-api -p 8000:8000 --env-file .env fraud-api:latest
docker logs -f fraud-api                       # wait for "Model LightGBM … loaded", then Ctrl+C
docker ps                                      # STATUS shows (healthy) after ~1 minute
```
Test exactly as before — Postman (`baseUrl` = `http://localhost:8000`) and `python client/test_api.py --replay 1000`.
(Stop the local uvicorn from Part 5 first, or use `-p 8001:8000` and `--url http://localhost:8001`.)
```bash
docker stop fraud-api && docker rm fraud-api
```

---

## PART 10 — GitHub

1. github.com → **New repository** → `fraud-detection-api` → **Public** (simplest for SageMaker to clone; Private works too — see Part 15.4) → *Create* (no README).
2. In the project folder:
```bash
git init
git add .
git status            # CHECK: no .env, no local_s3/, no .pkl, no data → .gitignore protects them
git commit -m "Fraud detection API: production code, Docker, K8s, CI/CD, drift + retraining"
git branch -M main
git remote add origin https://github.com/<your-user>/fraud-detection-api.git
git push -u origin main
```
3. Repository → **Settings → Secrets and variables → Actions → New repository secret**:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (the fraud-app keys).

> The first push already starts the CI/CD workflow. The test job will pass; build/deploy fail until Part 12 exists — that's expected.

---

## PART 11 — Kubernetes on your laptop (minikube)

```bash
minikube start --driver=docker --cpus 4 --memory 6g      # Docker Desktop must be running
minikube addons enable metrics-server
minikube image load fraud-api:latest                  # copy your local image into minikube (no registry needed)

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-configmap.yaml
kubectl create secret generic aws-credentials -n fraud --from-literal=AWS_ACCESS_KEY_ID=<key-id> --from-literal=AWS_SECRET_ACCESS_KEY=<secret>
```
Deployment with the image name filled in (`sed` replaces IMAGE_PLACEHOLDER — Git Bash has sed built in):
```bash
sed "s|IMAGE_PLACEHOLDER|fraud-api:latest|g" k8s/03-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/04-service.yaml -f k8s/05-hpa.yaml
kubectl get pods -n fraud -w            # wait until 2/2 pods are 1/1 Running (≈ 1 min), Ctrl+C
minikube service fraud-api -n fraud --url        # prints a URL like http://127.0.0.1:53412 — keep this terminal open
```
Test with that URL:
```bash
python client/test_api.py --url http://127.0.0.1:53412 --replay 1000
```
Postman: set `baseUrl` to the same URL. Useful commands:
```bash
kubectl logs -n fraud deploy/fraud-api           # logs of one pod
kubectl describe pod -n fraud <pod-name>         # why is a pod not starting?
kubectl get hpa -n fraud                         # auto-scaling status
kubectl scale deploy/fraud-api -n fraud --replicas 3
```
Drift CronJob locally (same sed trick with `k8s/06-drift-cronjob.yaml`), run it now:
```bash
kubectl create job --from=cronjob/fraud-drift-check drift-now -n fraud
kubectl logs -n fraud job/drift-now -f
```
Stop: `minikube stop` (keep) or `minikube delete` (remove).

---

## PART 12 — AWS: ECR (image registry) + EKS (Kubernetes in the cloud)

> 💰 **Cost:** EKS ≈ $0.10/hour for the control plane + 2 × t3.large nodes ≈ $0.17/hour + load balancer.
> Roughly **₹40–60 per hour** in total. Do Part 17 (clean-up) when you finish testing!

### 12.1 ECR — store the Docker image in AWS
```bash
aws ecr create-repository --repository-name fraud-api --region ap-south-1
aws sts get-caller-identity --query Account --output text          # → your ACCOUNT_ID (12 digits)
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com
docker tag fraud-api:latest <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/fraud-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/fraud-api:latest
```
> Apple-silicon Mac: build for the cloud CPU type: `docker buildx build --platform linux/amd64 -t fraud-api:latest .`

### 12.2 EKS — create the cluster (≈ 15–20 minutes, use your ADMIN AWS profile)
`fraud-app` is not allowed to create clusters (on purpose). Add your admin keys as a second profile once:
`winpty aws configure --profile admin`, then in this terminal:
```bash
export AWS_PROFILE=admin
eksctl create cluster --name fraud-cluster --region ap-south-1 --nodegroup-name workers --node-type t3.large --nodes 2 --nodes-min 2 --nodes-max 4 --managed
kubectl get nodes                       # 2 nodes Ready
```
Allow the `fraud-app` user (used by GitHub Actions) to deploy into the cluster:
```bash
eksctl create iamidentitymapping --cluster fraud-cluster --region ap-south-1 --arn arn:aws:iam::<ACCOUNT_ID>:user/fraud-app --group system:masters --username fraud-app
```
Metrics server for auto-scaling:
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
unset AWS_PROFILE          # back to the fraud-app keys for everything else
```

### 12.3 Deploy
```bash
kubectl apply -f k8s/00-namespace.yaml -f k8s/01-configmap.yaml
kubectl create secret generic aws-credentials -n fraud --from-literal=AWS_ACCESS_KEY_ID=<key-id> --from-literal=AWS_SECRET_ACCESS_KEY=<secret>
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export IMAGE=$ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/fraud-api:latest
sed "s|IMAGE_PLACEHOLDER|$IMAGE|g" k8s/03-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/04-service.yaml -f k8s/05-hpa.yaml
kubectl get pods -n fraud
kubectl get svc fraud-api -n fraud      # EXTERNAL-IP = a long …elb.amazonaws.com name (takes 2–3 minutes)
```

### 12.4 Check the live API properly
```bash
curl http://<ELB-HOSTNAME>/health
curl http://<ELB-HOSTNAME>/model-info
python client/test_api.py --url http://<ELB-HOSTNAME> --replay 1000
```
Postman: `baseUrl` = `http://<ELB-HOSTNAME>` → run the collection. Same results as on your laptop = deployment is correct.
Also check: `kubectl get hpa -n fraud`, `kubectl logs -n fraud deploy/fraud-api`, and in S3 the new files under `logs/predictions/`.

---

## PART 13 — CI/CD with GitHub Actions

`.github/workflows/ci-cd.yml` runs on every push to `main`:
```
test        → install → ruff → download model/artifacts from S3 → pytest (parity + API + latency)
build-push  → docker build → push to ECR with tags <commit-id> and latest
deploy      → kubectl apply (configmap, service, HPA, CronJob) + new image in the Deployment
            → rolling update (no downtime) → smoke test /health and /score on the load balancer
            → if anything fails: kubectl rollout undo (automatic rollback)
```
Try it:
```bash
# change something small, e.g. the description text in src/api.py
git add . && git commit -m "test CI/CD" && git push
```
GitHub → **Actions** tab → watch the 3 jobs turn green → `kubectl get pods -n fraud` shows new pods with the new image.
Pull requests run only the **test** job (nothing is deployed until merged).
`deploy-new-model.yml` = restart pods after SageMaker promotes a new model (Part 15) — or run it by hand from the Actions tab.

---

## PART 14 — Data drift monitoring (Evidently AI)

### 14.1 How drift is tracked
| Step | What | Where |
|---|---|---|
| 1. Reference | features + scores of the training period = "normal" | `monitoring/reference_features.parquet` |
| 2. Collect | every scored payment's 71 features + score + band | `logs/predictions/date=…/` |
| 3. Data drift (weekly) | Evidently **DataDriftPreset**, method **PSI**: continuous features in 10 bins, 0/1 and few-value features as categories | `detect_drift.py` |
| 4. Prediction drift | PSI of the fraud score + **alert rate** (normal ≈ 3 %) | `detect_drift.py` |
| 5. Performance (when labels mature) | join logs with labelled exports in `new_data/` → **recall, precision, fraud money caught** (Evidently **ClassificationPreset**) | `detect_drift.py` |
| 6. Decision | retrain if: a TOP-10 feature PSI > 0.25 · > 30 % of features drifted · alert rate > 5 % · recall < 90 % | `config.py` limits |
| 7. Output | `reports/drift/<date>/data_drift.html`, `performance.html`, `summary.json`, `reports/drift/latest.json` | S3 |
| 8. Action | SNS e-mail + start SageMaker retraining (max once per 14 days) | `trigger_retrain.py` |

Two kinds of drift, both caught:
* **Data drift** — inputs change (e.g. a new app version doubles login time). Seen by PSI **immediately**, without labels.
* **Concept drift** — the meaning changes (a new scam type the model never saw). Inputs can look almost normal; it is
  seen in **performance** once labels arrive. That is why both checks exist.

### 14.2 Run it
In Kubernetes it runs every Monday 03:00 UTC (`k8s/06-drift-cronjob.yaml`). Deploy the CronJob (EKS, image from ECR):
```bash
sed "s|IMAGE_PLACEHOLDER|$IMAGE|g" k8s/06-drift-cronjob.yaml | kubectl apply -f -
kubectl create job --from=cronjob/fraud-drift-check drift-now -n fraud      # run now
kubectl logs -n fraud job/drift-now -f
```
Or on your laptop:
```bash
python -m monitoring.detect_drift --no-trigger        # report only
```
Download and open the HTML report:
```bash
aws s3 cp s3://fraud-detection-project-2026-suresh/reports/drift/<date>/data_drift.html .
```

### 14.3 E-mail alerts (SNS)
```bash
aws sns create-topic --name fraud-model-alerts --region ap-south-1          # → TopicArn
aws sns subscribe --topic-arn <TopicArn> --protocol email --notification-endpoint <your e-mail>
```
Confirm the e-mail you receive, then put the TopicArn into `SNS_TOPIC_ARN` in `k8s/01-configmap.yaml` and `kubectl apply -f k8s/01-configmap.yaml`.

---

## PART 15 — Automatic retraining in SageMaker

### 15.1 What happens
```
detect_drift.py finds drift
   → trigger_retrain.py → SageMaker Processing job (our Docker image from ECR, SageMaker's IAM role — no keys)
        1. git clone the LATEST code from GitHub (GIT_REPO_URL)        ← "new code from GitHub"
        2. papermill runs sagemaker/retrain_pipeline.ipynb:
             history (data/…) + every new labelled export in s3://…/new_data/   ← "new data from S3"
             clean_batch → data-quality gate (STOP if labels are broken)
             time split on mature labels: train | valid (1 month) | test (newest 1 month)
             fill values + encoders on train only → features.py (same code as the API) → LightGBM
             threshold by business cost with ≤ 3 % alerts
             CHAMPION (current model) vs CHALLENGER (new) on the newest month
             promote ONLY if the challenger is better → models/history/vX.Y/ + models/best_model.pkl + config
        3. executed notebook with all outputs → s3://…/reports/retraining/<job>/retrain_executed.ipynb
   → API pods load the new model within 5 minutes (or deploy-new-model.yml restarts them at once)
   → the next drift check compares against the NEW reference
```

### 15.2 Create the SageMaker role (once)
AWS Console → **IAM → Roles → Create role** → *Custom trust policy* → paste `iam/sagemaker-trust.json` → Next →
create an inline policy from `iam/sagemaker-role-policy.json` → role name **`FraudSageMakerRole`** → copy its **ARN**.

### 15.3 Fill the configmap and apply
In `k8s/01-configmap.yaml`:
```yaml
SAGEMAKER_ROLE_ARN:  "arn:aws:iam::<ACCOUNT_ID>:role/FraudSageMakerRole"
SAGEMAKER_IMAGE_URI: "<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/fraud-api:latest"
GIT_REPO_URL:        "https://github.com/<your-user>/fraud-detection-api.git"
SNS_TOPIC_ARN:       "arn:aws:sns:ap-south-1:<ACCOUNT_ID>:fraud-model-alerts"
```
```bash
kubectl apply -f k8s/01-configmap.yaml
git add k8s/01-configmap.yaml && git commit -m "retraining settings" && git push
```
⚠️ **Commit it too** — CI/CD re-applies `k8s/01-configmap.yaml` from GitHub on every deploy, so values only applied by hand
would be overwritten. (These are IDs, not secrets — safe in Git. Keys stay in the Kubernetes secret.)

### 15.4 Private GitHub repository?
Leave `GIT_REPO_URL` empty → SageMaker uses the code inside the image. Because CI/CD builds the image from every push to
`main`, the image **is** the latest GitHub code. (Putting a token inside the URL would expose it — don't.)

### 15.5 Test the trigger
```bash
# on your laptop (Git Bash)
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/FraudSageMakerRole
export SAGEMAKER_IMAGE_URI=<ACCOUNT_ID>.dkr.ecr.ap-south-1.amazonaws.com/fraud-api:latest
export GIT_REPO_URL=https://github.com/<your-user>/fraud-detection-api.git
python -m monitoring.trigger_retrain --dry-run      # prints the exact SageMaker request, starts nothing
python -m monitoring.trigger_retrain                # starts a real retraining job now
```
AWS Console → **SageMaker → Processing → Processing jobs** → status *InProgress → Completed* (≈ 5–10 min);
logs in CloudWatch. Download the executed notebook:
```bash
aws s3 ls s3://fraud-detection-project-2026-suresh/reports/retraining/ --recursive
```
You can also open `sagemaker/retrain_pipeline.ipynb` in **SageMaker Studio** (JupyterLab → upload/clone the repo) and run it by hand.

### 15.6 Optional: restart pods immediately after a new model
1. GitHub → Settings → Developer settings → **Fine-grained token** (repo: fraud-detection-api, permission *Contents: Read and write*).
2. `aws secretsmanager create-secret --name fraud-api/github-token --secret-string <token> --region ap-south-1`
3. Configmap: `GITHUB_REPO: "<your-user>/fraud-detection-api"` → the notebook calls GitHub → `deploy-new-model.yml` runs.

---

## PART 16 — Full end-to-end test: drift → retrain → redeploy

This uses a **simulated** new month so you can watch the whole loop. It changes data in your bucket (`new_data/`,
`feature_store/snapshots`) — back up first if you want: `aws s3 sync s3://fraud-detection-project-2026-suresh ./bucket-backup`.

```bash
# 1. Create the drift: app update (login time ×2 from 1 Sep) + a NEW investment-scam wave (Jun–Oct)
python -m scripts.simulate_new_data
#   ✅ new_data export: 15,526 rows (incl. 300 INVESTMENT_SCAM frauds) | live traffic Sep–Oct: 6,374 payments

# 2. The API must see the new store data (restart it, or):
curl -X POST http://localhost:8000/admin/reload

# 3. Send the "live traffic" through the API (this fills the prediction logs)
python client/test_api.py --skip-examples --file client/drift_traffic.csv --replay 10000
#   fraud caught : 19 of 181 (recall 10.5%)    ← the current model misses the new scam

# 4. Drift check
python -m monitoring.detect_drift --no-trigger
```
Expected:
```
Highest PSI: merchant_age_days 2.10, login_to_pay_seconds 0.57, device_age_days 0.52, account_age_days 0.34 …
Drifted features (PSI > 0.25): merchant_age_days, login_to_pay_seconds, device_age_days, account_age_days
Performance on 6,374 labelled payments: recall 10.5%, money caught 24.0%
DECISION: 🚨 RETRAIN — important feature(s) drifted … | recall on mature labels 10.5% < 90%
```
```bash
# 5a. Retrain in SageMaker (real flow)
python -m monitoring.detect_drift
# 5b. …or retrain right here on your laptop (same code)
python -m monitoring.detect_drift --local-retrain
```
Expected comparison (newest month, October):
```
                  champion (current)  challenger (new)
pr_auc                        0.0746            0.9420
recall                        0.0333            1.0000
money_caught                  0.0899            1.0000
🏆 Challenger WINS   ✅ Promoted v1.1
```
```bash
# 6. Within 5 minutes the running API switches to v1.1 by itself (or POST /admin/reload)
curl http://localhost:8000/health          # "model_version": "v1.1"
python client/test_api.py --skip-examples --file client/drift_traffic.csv --replay 10000
#   fraud caught : 181 of 181 (recall 100.0%)   genuine disturbed: 82 of 6193 (unchanged)
```
On EKS: same steps with `--url http://<ELB-HOSTNAME>`; pods log `🔄 New model version found … v1.0 → v1.1`.

**Rollback a model** — every promotion first backs up the current model to `models/history/<version>/`:
```bash
B=s3://fraud-detection-project-2026-suresh
aws s3 cp $B/models/history/v1.0/encoders.pkl       $B/artifacts/encoders.pkl
aws s3 cp $B/models/history/v1.0/impute_values.json $B/artifacts/impute_values.json
aws s3 cp $B/models/history/v1.0/feature_list.json  $B/artifacts/feature_list.json
aws s3 cp $B/models/history/v1.0/best_model.pkl     $B/models/best_model.pkl
aws s3 cp $B/models/history/v1.0/model_config.json  $B/models/model_config.json     # LAST — triggers the reload
```

---

## PART 17 — Clean up (avoid bills!)

```bash
kubectl delete namespace fraud                                   # pods, service (load balancer), CronJob
eksctl delete cluster --name fraud-cluster --region ap-south-1   # ≈ 10 min — stops the EKS + EC2 cost
aws ecr delete-repository --repository-name fraud-api --force --region ap-south-1
minikube delete
```
Check **EC2 → Load Balancers** and **EC2 → Instances** are empty. S3 storage costs only a few rupees per month.

---

## PART 18 — Troubleshooting

| Problem | Fix |
|---|---|
| `AccessDenied` / `NoCredentialsError` | `aws sts get-caller-identity`; in Docker use `--env-file .env`; in K8s check the `aws-credentials` secret |
| `NoSuchKey: feature_store/customers.parquet` | run `python -m scripts.prepare_artifacts` (Part 4) |
| API fails at start-up: model cannot be loaded (`unpickling` / version error, or Colab picked XGBoost / Random Forest) | rebuild the model with the pinned libraries: `python -m src.train --promote` — when the current model cannot be loaded, the new LightGBM replaces it automatically. Then start the API again |
| `SignatureDoesNotMatch` in Docker | `.env` saved with CRLF — switch VS Code to LF and save again |
| `python3: command not found` / Microsoft Store opens | in Git Bash use `python` (venv active) or `py -3.13` |
| Command hangs without a prompt in Git Bash | prefix with `winpty` (e.g. `winpty aws configure`) |
| `ModuleNotFoundError: src` | run commands from the project folder: `python -m scripts.x`, not `python scripts/x.py` |
| Port 8000 already in use | stop the other process, or `--port 8001` / `-p 8001:8000` |
| Pod `ImagePullBackOff` | wrong image name; minikube: `minikube image load fraud-api:latest`; EKS: image pushed to ECR? |
| Pod `CrashLoopBackOff` | `kubectl logs -n fraud <pod>` — usually keys / bucket / artifacts |
| Pod `OOMKilled` | raise memory limit in `03-deployment.yaml` (2Gi → 3Gi) |
| Service EXTERNAL-IP `<pending>` | EKS: wait 3 min; minikube: use `minikube service fraud-api -n fraud --url` |
| GitHub Actions deploy fails with `Unauthorized` | Part 12.2 `eksctl create iamidentitymapping` for the fraud-app user |
| Drift check says "skipped" | fewer than 500 logged payments in the last 7 days — replay more traffic |
| SageMaker job fails at `git clone` | repo private → leave `GIT_REPO_URL` empty (Part 15.4) |

### What is simulated vs what a real bank would add
* **Feature store** = Parquet tables in S3 loaded into memory. A bank would use **Redis / DynamoDB** with the same `lookup()` interface.
* **Live counters** are per pod (in memory). In production they live in **Redis** so all pods share them.
* **Labels**: the bank sends final labels monthly to `new_data/`; the 90-day wait is real in production.
* **Keys**: on EKS, the next step up is **IRSA** (IAM role for the service account) — then no keys exist anywhere.
