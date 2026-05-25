#!/usr/bin/env bash
# Lance Flask SecurAI sur Colab (GPU T4)
set -e
SECURAI_BASE="${SECURAI_BASE:-/content/drive/MyDrive/Vision_project/securai_store}"
cd "${SECURAI_BASE}"
export SECURAI_BASE
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
