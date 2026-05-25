# SecurAI Store - Biometric Access Control & Audit

SecurAI Store est une plateforme de simulation professionnelle pour l'audit de systèmes de reconnaissance faciale en milieu critique (magasin, zone sensible). Elle permet de tester la résilience des accès biométriques face aux attaques adverses (FGSM) et de valider les mécanismes de défense.

## 1. Installation

1. **Environnement virtuel** :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .\.venv\Scripts\activate   # Windows
   ```

2. **Dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Modèles** :
   Placez vos modèles YOLOv8 (`yolov8n.pt`) et FaceCNN dans le dossier `models/`.

## 2. Lancement Démo (Vulnerable Mode)

Lancez la simulation en mode "normal" pour tester la reconnaissance faciale standard :
```bash
python simulate.py --mode normal
```
Ou via l'interface web :
```bash
python app.py
```

## 3. Lancement Mode Hardened (Secure Mode)

Pour tester le système avec les protections actives (Adversarial Training + Anomaly Detection) :
```bash
python simulate.py --mode hardened --defense all
```

## 4. Google Colab (GPU T4)

Sur Drive, copier ce dossier vers : `MyDrive/Vision_project/securai_store`

| Fichier | Usage |
|---------|--------|
| `init_server.ipynb` | SSH + tunnel Cloudflare (`ssh vision-colab`) |
| `webcam_demo.ipynb` | Démo sans SSH — flux webcam navigateur |
| `train.py` | Entraînement + checkpoints (30 min) |
| `colab/setup_colab_deps.sh` | Install dépendances GPU |

---
*Projet développé dans le cadre de la recherche sur la robustesse des modèles CNN face aux attaques de type FGSM.*
