# Bilan du Projet SecurAI (Architecture V2)

Ce document résume l'état d'avancement actuel du Proof of Concept (POC) SecurAI, détaillant les accomplissements tant sur le plan théorique/conceptuel que technique.

---

## 1. Bilan Conceptuel (Cybersécurité & IA)

Le projet a atteint son objectif principal : démontrer de manière tangible les vulnérabilités des systèmes biométriques de pointe face au *Adversarial Machine Learning*, et proposer des pistes de remédiation.

### Ce qui est prouvé et fonctionnel :
*   **Vulnérabilité des Modèles Profonds** : Le système démontre que même une IA entraînée sur des millions de visages (FaceNet) peut être trompée si l'on manipule directement les pixels d'entrée (Pixels Perturbation).
*   **Usurpation d'Identité (Spoofing)** : Contrairement à une simple attaque par déni de service (brouillage), le projet montre une attaque *ciblée* : l'attaquant force le système à l'identifier spécifiquement comme "Administrateur" (Poutine) malgré des différences physiques majeures.
*   **Paradigme de l'Attaque (White-Box / Man-in-the-Middle)** : Le simulateur illustre le scénario où un hacker intercepte le flux de la caméra, injecte le bruit en temps réel, et renvoie le flux altéré au serveur.

### Les Mécanismes de Défense validés :
*   **Destruction par Filtrage Spatial** : Le POC valide l'hypothèse selon laquelle un bruit adverse haute-fréquence peut être "cassé" par des transformations d'image simples (flou gaussien/médian) avant que l'image n'atteigne le réseau de neurones.
*   **Détection d'Anomalie (Analyse de Fréquence)** : L'utilisation de la Transformée de Fourier Rapide (FFT) s'est avérée être un outil redoutable pour détecter la "signature" artificielle du bruit I-FGSM, offrant un système d'alerte précoce.

---

## 2. Bilan Technique (Architecture & Implémentation)

L'architecture logicielle a été entièrement refondue pour passer d'une logique d'entraînement laborieuse à un système "Zero-Shot" modulaire, robuste et performant.

### Moteurs d'Intelligence Artificielle
*   **Détection de Visages** : Utilisation de **YOLOv8**, offrant une détection ultrarapide et précise du cadre du visage.
*   **Reconnaissance Faciale (FaceNet)** : Utilisation de **InceptionResnetV1** (VGGFace2). Le système ne classe plus les images mais génère des **Embeddings 512D**. L'identification se fait par calcul de **Similarité Cosinus** (Seuil de 65%) par rapport aux signatures enregistrées dans `/data/enrolled/`.
*   **Moteur d'Attaque (I-FGSM / PGD)** : L'algorithme attaque désormais l'espace latent. Il utilise une descente de gradient projetée sur 10 itérations (`steps=10`) avec une forte perturbation (`epsilon=0.15`) pour traverser la grande distance mathématique séparant deux visages différents.

### Backend (Serveur & Logique)
*   **Framework** : Flask avec gestion multi-thread (un thread pour la capture vidéo, un autre pour servir les flux API).
*   **Gestion des Droits (`RightsManager`)** : Système RBAC (Role-Based Access Control) dynamique attribuant des zones d'accès selon l'identité (ADMIN = Vert, EMPLOYEE = Jaune, INCONNU = Rouge).

### Frontend (Control Room)
*   **Mode "Live Video"** : Interface plein écran "Cyber-Security" affichant le flux vidéo (MJPEG) avec Bounding Boxes temps réel, log des événements, et indicateurs de statut asynchrones (AJAX).
*   **Mode "Analyse Statique"** : Page dédiée aux tests unitaires, permettant l'upload d'images fixes via Drag & Drop et l'affichage d'un diagnostic détaillé (statut, confiance, score FFT).

---

## 3. Perspectives : L'Attaque Physique (Adversarial Patch)

L'état de l'art actuel du projet permet d'envisager la prochaine grande étape : **sortir de l'attaque numérique (Man-in-the-Middle) pour aller vers l'attaque physique**. 

**Prochain Défi Technique** :
Calculer hors-ligne (via GPU Cloud/Colab) un motif spécifique confiné à la géométrie d'une monture de lunettes (*Adversarial Patch*). L'objectif sera de tester si l'application numérique de ces lunettes sur une image statique réussit à déclencher la faille biométrique, validant ainsi la faisabilité de l'usurpation sans accès au réseau.
