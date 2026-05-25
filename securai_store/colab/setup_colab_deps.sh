#!/usr/bin/env bash
# Dépendances GPU Colab pour SecurAI
set -e
SECURAI_BASE="${SECURAI_BASE:-/content/drive/MyDrive/Vision_project/securai_store}"
pip install -q -r "${SECURAI_BASE}/requirements.txt"
pip install -q facenet-pytorch
echo "✓ Dépendances installées pour ${SECURAI_BASE}"
