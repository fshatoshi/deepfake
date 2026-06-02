# SecurAI — Biometric Access Control & Deepfake Attack Platform

> Plateforme de simulation pour l'audit de systèmes de reconnaissance faciale face aux attaques adversariales (FGSM, patch lunettes) et aux tentatives de spoofing.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Lancement](#4-lancement)
5. [Modules](#5-modules)
6. [API REST](#6-api-rest)
7. [Tests](#7-tests)
8. [GPU Toggle](#8-gpu-toggle)
9. [Contributions](#9-contributions)

---

## 1. Vue d'ensemble

SecurAI simule un système de contrôle d'accès biométrique dans un environnement de magasin, exposé à des attaques adversariales en temps réel.

| Fonctionnalité | Description |
|---|---|
| 🎥 Flux vidéo live | Détection et reconnaissance faciale en temps réel via webcam |
| ⚔️ Attaque FGSM | Perturbation adversariale I-FGSM déportée sur GPU (HF Space) |
| 🕶️ Patch lunettes live | Attaque par patch physique générée en temps réel (T1) |
| 🛡️ Défenseur | Filtres gaussien/médian pour robustesse aux perturbations |
| 🔍 Anomalie FFT | Détection locale d'images adversariales (~2ms) |
| 🚫 Anti-spoofing | Liveness detection LBP — bloque les photos et écrans (T5) |
| 🖥️ GPU toggle | Bascule CPU ↔ CUDA avec redémarrage automatique (T4) |
| 🧪 Tests pytest | 48 tests unitaires et d'intégration (T3) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Machine locale (CPU/GPU)                      │
│                                                                       │
│  Webcam ──► FaceDetector    ──► LivenessDetector  ──► [SPOOF?]      │
│             (YOLOv8n-face)       (LBP anti-spoof)       │            │
│                  │                                       ▼            │
│                  │                               Bloqué / Continuer  │
│                  ▼                                                    │
│             FaceRecognizer  ──────────────────────────────────────►  │
│             (FaceNet/                                   AnomalyDetector│
│              InceptionResnet)                           (FFT ~2ms)   │
│                  │                                                    │
│         ┌────────┴────────┐                                          │
│         ▼                 ▼                                          │
│    FGSMAttacker      PatchAttacker   ◄── /api/toggle_patch_live      │
│    (I-FGSM ε=0.03)   (PGD ε=0.35)                                   │
│         │                 │                                          │
│         └────────┬────────┘                                          │
│                  ▼                                                    │
│             Defender                                                  │
│             (gaussian / median)                                      │
│                  │                                                    │
│             Flask app.py  ──► /api/status, /api/enroll, ...         │
│                  │                                                    │
└──────────────────┼──────────────────────────────────────────────────┘
                   │  HTTP (hf_session)
                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Hugging Face Space (GPU T4)                              │
│         /infer   /fgsm   /enroll                                     │
│         Inférence FaceNet + FGSM GPU                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Flux d'une frame vidéo

```
Frame webcam
    │
    ▼
[YOLOv8] Détection bbox
    │
    ▼
[LivenessDetector] LBP anti-spoofing ──► SPOOF → affichage rouge + skip
    │ OK
    ▼
[FaceRecognizer] Embedding 512D (FaceNet local)
    │
    ├──► [AnomalyDetector] FFT → score anomalie
    │
    ├──► [FGSMAttacker] si attack_active → envoi HF GPU
    │
    ├──► [PatchAttacker] si patch_active → patch lunettes local
    │
    └──► Overlay + /api/status
```

---

## 3. Installation

### Prérequis

- Python 3.11+
- CUDA 12.1 (optionnel, pour inférence GPU locale)
- Webcam

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-repo>
cd Deepfake/securai_store

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\activate         # Windows

# 3. Installer les dépendances
# CPU uniquement :
pip install -r requirements.txt

# GPU (CUDA 12.1) :
pip install -r requirements_gpu.txt

# Dépendances supplémentaires (T5 anti-spoofing) :
pip install scikit-image

# Downgrade NumPy si conflit avec torch :
pip install "numpy<2"

# 4. Télécharger le modèle YOLOv8n-face
# Placer yolov8n-face.pt dans securai_store/
# Téléchargement : https://github.com/akanametov/yolo-face

# 5. Configurer le .env
cp .env.example .env
# Éditer .env :
#   SECURAI_DEVICE=cpu        # ou cuda
#   HF_BASE=https://...       # URL de votre HF Space
```

### Fichier `.env`

```ini
SECURAI_DEVICE=cpu
SECURAI_PORT=5000
```

---

## 4. Lancement

```bash
# Démarrage standard
python app.py

# Avec GPU (si CUDA disponible)
python toggle_gpu.py --gpu
python app.py

# Bascule CPU/GPU sans redémarrer manuellement
python toggle_gpu.py           # bascule automatique
python toggle_gpu.py --status  # état actuel
```

Ouvrir dans le navigateur : `http://localhost:5000`

---

## 5. Modules

| Fichier | Rôle |
|---|---|
| `app.py` | Backend Flask — routes, thread vidéo, état global |
| `modules/face_detector.py` | Détection YOLOv8n-face, crop 160×160 |
| `modules/face_recognizer.py` | Embedding FaceNet 512D, enrôlement, prédiction |
| `modules/fgsm_attacker.py` | Attaque I-FGSM (ε=0.03, 10 steps, α=0.02) |
| `modules/patch_attacker.py` | Attaque patch lunettes PGD (ε=0.35, 40 steps) |
| `modules/defender.py` | Défense : gaussian blur, median filter |
| `modules/anomaly_detector.py` | Détection d'anomalie FFT locale (~2ms) |
| `modules/liveness_detector.py` | Anti-spoofing LBP — détecte photos/écrans (T5) |
| `rights_manager.py` | Gestion des droits Employee / Manager / DENIED |
| `toggle_gpu.py` | Bascule CPU↔GPU + redémarrage Flask (T4) |
| `paths.py` | Chemins partagés (local Windows / Colab) |

### Templates HTML

| Fichier | Description |
|---|---|
| `templates/EntranceControl.html` | Dashboard principal — flux vidéo live |
| `templates/StaticAnalysis.html` | Analyse d'une image statique |
| `templates/glasses_attack.html` | Interface attaque patch lunettes |

---

## 6. API REST

### `GET /api/status`
État temps réel du système.

```json
{
  "identity": "Manager_Demo",
  "access_level": "GRANTED",
  "confidence": 0.92,
  "anomaly_detected": false,
  "anomaly_score": 0.03,
  "attack_active": false,
  "patch_active": false,
  "patch_target": "Manager_Demo",
  "model_mode": "standard",
  "fps": 28
}
```

### `POST /api/enroll`
Enrôle un nouveau visage.

```bash
curl -X POST http://localhost:5000/api/enroll \
  -F "name=Alice" \
  -F "level=Employee" \
  -F "image=@photo.jpg"
```

### `POST /api/toggle_attack`
Active / désactive l'attaque FGSM live.

```json
{ "active": true }
```

### `POST /api/toggle_patch_live` *(T1)*
Active / désactive le patch lunettes en temps réel.

```json
{ "active": true, "target": "Manager_Demo" }
```

### `POST /api/analyze_static`
Analyse une image statique (avec détection d'anomalie).

```bash
curl -X POST http://localhost:5000/api/analyze_static \
  -F "image=@photo.jpg"
```

### `POST /api/generate_glasses_attack`
Génère un patch lunettes sur une image et compare avant/après.

---

## 7. Tests

```bash
cd securai_store/
pytest
```

**48 tests** couvrant :
- `tests/test_face_detector.py` — détection, crop, frame skip (12 tests)
- `tests/test_face_recognizer.py` — embedding, enrôlement, prédiction (15 tests)
- `tests/test_endpoints.py` — routes Flask, JWT, enrôlement (21 tests)

---

## 8. GPU Toggle

```bash
python toggle_gpu.py              # bascule auto CPU↔CUDA
python toggle_gpu.py --gpu        # force CUDA
python toggle_gpu.py --cpu        # force CPU
python toggle_gpu.py --status     # état actuel
python toggle_gpu.py --no-restart # change .env sans redémarrer
```

Le device est persisté dans `.env` (`SECURAI_DEVICE=cpu|cuda`) et injecté automatiquement au redémarrage du serveur.

---

## 9. Contributions

| Membre | Contributions |
|---|---|
| **Nadahe** | Architecture de base, YOLOv8-face, interface web (HTML/CSS/JS), backend Flask, entraînement `best_model.pt` |
| **Traoré** | Attaque FGSM (client & serveur), FaceNet sur crop YOLO, entraînement defender, mode statique, recherche cloud GPU |
| **François** | Patch lunettes live (T1), tests pytest (T3), toggle_gpu.py (T4), anti-spoofing LBP (T5) |
| **Yassin** | — |

---

## Structure du projet

```
securai_store/
├── app.py                      # Backend Flask principal
├── paths.py                    # Chemins partagés
├── rights_manager.py           # Gestion des droits
├── toggle_gpu.py               # Bascule CPU/GPU (T4)
├── pytest.ini                  # Configuration pytest
├── requirements.txt            # Dépendances CPU
├── requirements_gpu.txt        # Dépendances GPU (CUDA 12.1)
├── .env                        # Configuration locale (non versionné)
├── modules/
│   ├── face_detector.py        # YOLOv8n-face
│   ├── face_recognizer.py      # FaceNet / InceptionResnetV1
│   ├── fgsm_attacker.py        # Attaque I-FGSM
│   ├── patch_attacker.py       # Attaque patch PGD
│   ├── defender.py             # Défenses adversariales
│   ├── anomaly_detector.py     # Détection FFT
│   └── liveness_detector.py    # Anti-spoofing LBP (T5)
├── templates/
│   ├── EntranceControl.html    # Dashboard live
│   ├── StaticAnalysis.html     # Analyse statique
│   └── glasses_attack.html     # Interface patch
├── tests/
│   ├── conftest.py
│   ├── test_face_detector.py
│   ├── test_face_recognizer.py
│   └── test_endpoints.py
├── data/
│   └── enrolled/               # Embeddings enrôlés
├── models/
│   └── class_names.json
└── security_audit.log          # Logs d'anomalies
```