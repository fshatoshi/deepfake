# Cahier des Charges — GPU Déporté : Colab T4 ⇄ Cloudflare Tunnel

> **Contexte** : Machine de dev Windows 11 sans GPU suffisant. On déporte tous les calculs d'entraînement sur Google Colab (GPU T4 gratuit). L'accès se fait via un tunnel Cloudflare Zero Trust — pas de port forwarding, pas d'IP fixe.
>
> **Domaine SSH** : `vision.api.near-u-api.org` (sous-domaine neutre sur Cloudflare)  
> **Domaine de base** : `api.near-u-api.org` (géré sur Cloudflare)

---

## Vue d'ensemble du flux réseau

```
[PC Windows 11]  ⇄  [Cloudflare Edge]  ⇄  [cloudflared daemon]  ⇄  [Google Colab T4]
  VS Code SSH           Zero Trust          (tourne dans Colab)       Serveur SSH :22
  FileZilla SFTP     vision.api.near-u-api.org
```

---

## Étape 1 — Préparer Google Drive

> 🙋 **QUI : TOI**

Crée manuellement cette arborescence sur ton Google Drive :

```
MyDrive/
└── Vision_project/
    └── securai_store/          ← code SecurAI (sync depuis ce dépôt)
        ├── data/enrolled/      ← images d'enrôlement / entraînement
        ├── models/
        ├── checkpoints/
        ├── output/
        ├── logs/
        ├── init_server.ipynb
        ├── train.py
        └── webcam_demo.ipynb   ← alternative sans SSH
```

---

## Étape 2 — Créer le Tunnel Cloudflare

> 🙋 **QUI : TOI** (actions manuelles sur l'interface web Cloudflare)

1. Va sur [one.dash.cloudflare.com](https://one.dash.cloudflare.com) → **Networks → Tunnels → Create a tunnel**
2. Nomme le tunnel : `colab-vision-tunnel`
3. Sélectionne **Linux AMD64** → copie le **token** généré (longue chaîne alphanumérique)
4. Dans **Public Hostname**, configure :
   - Subdomain : `vision`
   - Domain : `api.near-u-api.org`
   - Service : `SSH`
   - URL : `localhost:22`
5. Sauvegarde.

> Le token ressemble à : `eyJhIjoiMWY4...` — garde-le, il sera collé dans le notebook.

---

## Étape 3 — Installer cloudflared sur Windows 11

> 🙋 **QUI : TOI**

Dans PowerShell (admin) :

```powershell
winget install Cloudflare.cloudflared
cloudflared --version  # vérification
```

---

## Étape 4 — Fichier de config SSH Windows

> 🤖 **QUI : ANTIGRAVITY** — générer le fichier `~/.ssh/config`

Créer ou compléter le fichier `C:\Users\<USERNAME>\.ssh\config` avec ce contenu :

```
Host vision-colab
    HostName vision.api.near-u-api.org
    User root
    Port 22
    ProxyCommand cloudflared access ssh --hostname %h
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

> **Résultat attendu** : `ssh vision-colab` fonctionne depuis le terminal Windows sans connaître l'IP de Colab.

---

## Étape 5 — Notebook Colab d'initialisation

> 🤖 **QUI : AGENT** — notebook `securai_store/init_server.ipynb`

Le notebook doit s'exécuter en moins de 3 minutes à chaque démarrage de session Colab. Il contient 3 cellules.

### Cellule 1 — Variables & montage Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# ── À personnaliser ──────────────────────────────────────
SECURAI_BASE = '/content/drive/MyDrive/Vision_project/securai_store'
CF_TOKEN     = 'COLLE_TON_TOKEN_CLOUDFLARE_ICI'
SSH_PASSWORD = 'MotDePasseTemp2024!'  # temporaire, par session
# ─────────────────────────────────────────────────────────

import os
os.environ['SECURAI_BASE'] = SECURAI_BASE
for folder in ['checkpoints', 'output', 'logs', 'data/enrolled', 'models']:
    os.makedirs(f'{SECURAI_BASE}/{folder}', exist_ok=True)

print('✓ Drive monté et dossiers vérifiés.')
```

### Cellule 2 — Installation et configuration SSH

```python
import subprocess, os

for cmd in [
    'apt-get update -qq',
    'apt-get install -y -qq openssh-server',
    'mkdir -p /var/run/sshd',
]:
    subprocess.run(cmd, shell=True, check=True)

with open('/etc/ssh/sshd_config', 'a') as f:
    f.write('\nPermitRootLogin yes\nPasswordAuthentication yes\n')

os.system(f'echo "root:{SSH_PASSWORD}" | chpasswd')
print('✓ SSH configuré.')
```

### Cellule 3 — Lancement du tunnel Cloudflare

```python
import subprocess, time

subprocess.run(
    'wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/'
    'cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared',
    shell=True, check=True
)

subprocess.run('service ssh start', shell=True)

log_path = f'{SECURAI_BASE}/logs/tunnel.log'
proc = subprocess.Popen(
    f'cloudflared tunnel --no-autoupdate run --token {CF_TOKEN}',
    shell=True,
    stdout=open(log_path, 'w'),
    stderr=subprocess.STDOUT
)

time.sleep(6)
print(f'✓ Tunnel actif (PID: {proc.pid})')
print(f'→ Connexion depuis Windows : ssh vision-colab')
```

---

## Étape 6 — Script d'entraînement avec checkpoints

> 🤖 **QUI : AGENT** — `securai_store/train.py`

### Contraintes à respecter

- Checkpoint toutes les **30 minutes** sur Drive
- Garder uniquement les **3 derniers** (supprimer les anciens automatiquement)
- **Reprendre** depuis le dernier checkpoint au démarrage
- Logger dans `logs/train.log`
- Compatible `nohup` (résiste aux déconnexions SSH)

### Fonctions obligatoires

```python
import os, glob, time, torch

CKPT_DIR   = f'{SECURAI_BASE}/checkpoints'
MAX_CKPTS  = 3
SAVE_EVERY = 1800  # 30 min

def save_checkpoint(model, optimizer, epoch, loss):
    ts   = int(time.time())
    path = f'{CKPT_DIR}/ckpt_epoch{epoch:04d}_{ts}.pt'
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }, path)
    for old in sorted(glob.glob(f'{CKPT_DIR}/ckpt_*.pt'))[:-MAX_CKPTS]:
        os.remove(old)
    print(f'[CKPT] Sauvé : {os.path.basename(path)}')

def load_latest_checkpoint(model, optimizer):
    ckpts = sorted(glob.glob(f'{CKPT_DIR}/ckpt_*.pt'))
    if not ckpts:
        print('[CKPT] Aucun checkpoint — démarrage à zéro.')
        return 0
    ckpt = torch.load(ckpts[-1])
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f'[CKPT] Reprise depuis : {os.path.basename(ckpts[-1])}')
    return ckpt['epoch']
```

### Lancement depuis le terminal SSH

```bash
cd /content/drive/MyDrive/Vision_project/securai_store
nohup python -u train.py > logs/train.log 2>&1 &
echo "Entraînement lancé — PID: $!"

# Surveiller en live :
tail -f logs/train.log
```

---

## Étape 7 — Test bout en bout

> 🙋 **QUI : TOI** (validation unique)

1. Colab → **Runtime → Change runtime type → GPU (T4)**
2. Exécuter les 3 cellules de `init_server.ipynb`
3. PowerShell Windows : `ssh vision-colab`
4. Dans le terminal SSH : `nvidia-smi` → doit afficher le T4
5. `tail -f /content/drive/MyDrive/Vision_project/securai_store/logs/tunnel.log`

---

## Checklist de démarrage (chaque session Colab)

> 🙋 **QUI : TOI**

```
[ ] Colab → Runtime → GPU activé (T4)
[ ] Exécuter les 3 cellules init_server.ipynb
[ ] Tester : ssh vision-colab
[ ] nvidia-smi → T4 visible
[ ] load_latest_checkpoint() pour reprendre
[ ] nohup python train.py > logs/train.log 2>&1 &
[ ] Mettre une alarme dans 11h pour relancer si besoin
```

---

## Résumé des responsabilités
## 7️⃣ Intégration avec SecurAI Store (GPU distant)

```text
# 1️⃣ Copier le répertoire `securai_store` sur Google Drive
# Copier le dossier securai_store du PC vers MyDrive/Vision_project/securai_store

# 2️⃣ Lancer le serveur Flask sur Colab (GPU disponible)
cd /content/drive/MyDrive/Vision_project/securai_store
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
# 3️⃣ Accéder depuis votre poste Windows via le tunnel Cloudflare
#   Le tunnel redirige le port 5000 du serveur Colab vers votre domaine
#   Exemple d’URL : http://vision.api.near-u-api.org:5000
#   Vous pouvez alors utiliser l’interface web ou les API (`/api/generate_glasses_attack`, `/api/analyze_static`) comme d’habitude.
```
| Étape | Qui | Quoi |
|---|---|---|
| 1 | 🙋 Toi | Créer les dossiers sur Google Drive |
| 2 | 🙋 Toi | Créer le tunnel sur le dashboard Cloudflare |
| 3 | 🙋 Toi | Installer cloudflared sur Windows |
| 4 | 🤖 Antigravity | Fichier `.ssh/config` Windows |
| 5 | 🤖 Antigravity | Notebook `init_server.ipynb` (3 cellules) |
| 6 | 🤖 Antigravity | Script `train.py` avec checkpoints |
| 7 | 🙋 Toi | Test bout en bout + validation |
