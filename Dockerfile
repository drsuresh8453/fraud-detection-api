# ─────────────────────────────────────────────────────────────────────────────
# Fraud Detection API — one image for: the API, the drift CronJob and SageMaker retraining
# Created by: Suresh D R | AI Product Developer & Technology Mentor
# Build:  docker build -t fraud-api:latest .
# Run:    docker run -p 8000:8000 --env-file .env fraud-api:latest
# ─────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# true  → install git (SageMaker pulls the latest code) + libgomp1 (needed by LightGBM)
ARG INSTALL_OS_PACKAGES=true
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STORAGE_MODE=s3 \
    PORT=8000 \
    WORKERS=1

RUN if [ "$INSTALL_OS_PACKAGES" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# 1. dependencies first (cached layer — rebuilt only when requirements.txt changes)
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# 2. application code (NO data, NO models, NO keys — models are loaded from S3 at start-up)
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY scripts/ ./scripts/
COPY sagemaker/ ./sagemaker/
COPY client/ ./client/

# 3. never run as root
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["sh", "-c", "python3 -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]