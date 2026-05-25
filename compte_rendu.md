# Compte Rendu : Projet SecurAI (Vision & Biométrie) 👁️🛡️

## 1. État des Lieux
Le projet **SecurAI** est actuellement dans sa **version 2 (V2)**. Il s'agit d'un Proof of Concept (POC) mature qui démontre la vulnérabilité des systèmes de reconnaissance faciale face aux attaques adversariales.

L'architecture est passée d'un système nécessitant un entraînement spécifique à une approche **Zero-Shot**, ce qui le rend beaucoup plus flexible et rapide à déployer.

## 2. Architecture Technique
Le projet est structuré de manière modulaire dans le dossier `securai_store/` :

*   **Détection (Face Detection)** : Utilise **YOLOv8** pour localiser les visages avec une précision extrême.
*   **Reconnaissance (Face Recognition)** : Utilise **FaceNet (InceptionResnetV1)**. Au lieu de simples étiquettes, le système génère des **embeddings 512D** et compare la **similarité cosinus**.
*   **Attaque (Adversarial Attack)** : Implémentation du moteur **PGD (Projected Gradient Descent)** / **I-FGSM**. Il permet de tromper l'IA pour qu'elle identifie une personne comme étant une autre (ex: Administrateur).
*   **Défense (Anti-Spoofing)** :
    *   **FFT (Transformée de Fourier)** : Détecte les anomalies de haute fréquence générées par le bruit d'attaque.
    *   **Filtrage Spatial** : Applique un flou gaussien pour "nettoyer" l'image avant l'analyse IA.
*   **Interface (Dashboard)** : Un serveur Flask gère un tableau de bord vidéo temps réel et une page d'analyse statique.

## 3. Analyse du Code
Le code est **très bien structuré** et suit les bonnes pratiques :
- **Multithreading** : La capture vidéo et le serveur web tournent sur des threads séparés pour éviter les saccades.
- **Auto-enrôlement** : Le système scanne automatiquement `data/enrolled/` au démarrage pour enregistrer les visages autorisés.
- **Contrôle d'accès (RBAC)** : Le `RightsManager` gère dynamiquement les permissions (Admin, Employé, Inconnu).

## 4. Ce qui a été fait
- ✅ Consolidation de l'architecture V2.
- ✅ Intégration de YOLOv8 pour la détection.
- ✅ Mise en place du pipeline d'attaque iterative (PGD).
- ✅ Création du dashboard de contrôle cyber-sécurité.
- ✅ Documentation complète (Readme et Bilan).

## 5. Prochaines Étapes
Le projet est prêt à l'emploi. La prochaine phase identifiée est l'**Attaque Physique (Adversarial Patch)** : concevoir des motifs (ex: lunettes ou stickers) capables de tromper l'IA dans le monde réel, sans accès au flux numérique.

---
**Statut actuel : OPÉRATIONNEL** 🚀
