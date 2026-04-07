import os
import requests

url = "https://github.com"
output_path = "backend/models/saved_models/best_model.pth"

# Créer le dossier s'il n'existe pas
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print("Téléchargement du modèle (300MB)...")
r = requests.get(url, allow_redirects=True)
with open(output_path, 'wb') as f:
    f.write(r.content)
print("Modèle récupéré avec succès !")
