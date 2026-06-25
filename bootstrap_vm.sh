#!/usr/bin/env bash
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/mlops-serve}
PYTHON_BIN=${PYTHON_BIN:-python3}
SERVICE_NAME=${SERVICE_NAME:-mlops-serve}
SERVICE_PATH=${SERVICE_PATH:-/etc/systemd/system/${SERVICE_NAME}.service}

sudo mkdir -p "$APP_DIR/src" "$APP_DIR/models"
sudo apt-get update
sudo apt-get install -y python3-pip
sudo "$PYTHON_BIN" -m pip install --upgrade pip
sudo "$PYTHON_BIN" -m pip install fastapi==0.111.0 uvicorn==0.29.0 scikit-learn==1.4.2 joblib==1.4.2 boto3==1.34.131
sudo cp src/serve.py "$APP_DIR/src/serve.py"
sudo cp deploy/mlops-serve.service "$SERVICE_PATH"
sudo touch "$APP_DIR/.env"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
