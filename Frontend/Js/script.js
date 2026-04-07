// Script pour la plateforme Deepfake

document.addEventListener('DOMContentLoaded', function() {
    // Même origine si l'interface est servie par Flask ; sinon API locale
    const API_BASE =
        window.location.protocol === 'http:' || window.location.protocol === 'https:'
            ? ''
            : 'http://127.0.0.1:5000';

    /**
     * Lit le corps de la réponse sans supposer du JSON (évite JSON.parse sur HTML ou corps vide).
     */
    async function readJsonFromResponse(response) {
        const text = await response.text();
        const trimmed = text.trim();
        if (!trimmed) {
            return {
                ok: false,
                data: null,
                message:
                    'Réponse vide du serveur (HTTP ' +
                    response.status +
                    '). Vérifiez que Flask tourne : py backend/api/app.py depuis la racine du projet.'
            };
        }
        try {
            return { ok: true, data: JSON.parse(trimmed), raw: null };
        } catch (e) {
            return {
                ok: false,
                data: null,
                message:
                    'Le serveur n’a pas renvoyé du JSON (HTTP ' +
                    response.status +
                    '). Souvent une page d’erreur HTML ou le mauvais port. Aperçu : ' +
                    trimmed.slice(0, 100).replace(/</g, '&lt;')
            };
        }
    }

    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.section');
    const actionBtns = document.querySelectorAll('.action-btn');

    // Gestion de la navigation
    navBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const sectionId = this.getAttribute('data-section');

            // Masquer toutes les sections
            sections.forEach(section => {
                section.classList.remove('active');
            });

            // Désactiver tous les boutons
            navBtns.forEach(b => {
                b.classList.remove('active');
            });

            // Activer la section et le bouton sélectionnés
            document.getElementById(sectionId).classList.add('active');
            this.classList.add('active');
        });
    });

    // Gestion des actions (simulation pour l'instant)
    actionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const section = this.closest('.section');
            const resultDiv = section.querySelector('.result');
            const type = section.querySelector('input[type="radio"]:checked').value;

            // Simulation d'une action
            resultDiv.innerHTML = '<p>Traitement en cours avec ' + (type === 'cnn-scratch' ? 'CNN from Scratch' : 'CNN YOLOv8') + '...</p>';

            // Simuler un délai
            setTimeout(() => {
                resultDiv.innerHTML = '<p>Résultat : Action terminée avec succès !</p>';
            }, 2000);
        });
    });

    // Gestion de l'upload de fichier pour la détection et protection
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const resultDiv = this.closest('.section').querySelector('.result');
                resultDiv.innerHTML = '<p>Fichier sélectionné : ' + file.name + '</p>';
            }
        });
    });

    // Gestion de la webcam pour l'attaque FGSM
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const startAttackBtn = document.getElementById('start-attack');
    const stopAttackBtn = document.getElementById('stop-attack');
    const attackEnabledCheckbox = document.getElementById('attack-enabled');
    const epsilonSlider = document.getElementById('epsilon-slider');
    const epsilonValue = document.getElementById('epsilon-value');
    const epsilonPlus = document.getElementById('epsilon-plus');
    const epsilonMinus = document.getElementById('epsilon-minus');
    const attackStatus = document.getElementById('attack-status');
    const detectionCount = document.getElementById('detection-count');
    const fpsCounter = document.getElementById('fps-counter');
    const attackResultDiv = document.querySelector('#attaque .result');

    let stream;
    let intervalId;
    let lastFrameTime = 0;
    let frameCount = 0;

    // Variables d'état
    let attackEnabled = true;
    let epsilon = 0.015;

    // Gestion du slider epsilon
    epsilonSlider.addEventListener('input', (e) => {
        epsilon = parseFloat(e.target.value);
        epsilonValue.textContent = epsilon.toFixed(3);
    });

    // Boutons + et - pour epsilon
    epsilonPlus.addEventListener('click', () => {
        epsilon = Math.min(0.100, epsilon + 0.005);
        epsilonSlider.value = epsilon;
        epsilonValue.textContent = epsilon.toFixed(3);
    });

    epsilonMinus.addEventListener('click', () => {
        epsilon = Math.max(0.001, epsilon - 0.005);
        epsilonSlider.value = epsilon;
        epsilonValue.textContent = epsilon.toFixed(3);
    });

    // Checkbox pour activer/désactiver l'attaque
    attackEnabledCheckbox.addEventListener('change', (e) => {
        attackEnabled = e.target.checked;
        updateAttackStatus();
    });

    startAttackBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;
            attackResultDiv.innerHTML = '<p>Webcam démarrée. Simulation FGSM en cours...</p>';
            attackStatus.textContent = 'Statut: Simulation active';

            // Envoyer des frames toutes les 100ms
            intervalId = setInterval(() => {
                captureAndSendFrame();
            }, 100);

            // Démarrer le compteur FPS
            startFPSCounter();

        } catch (error) {
            attackResultDiv.innerHTML = '<p>Erreur d\'accès à la webcam: ' + error.message + '</p>';
            attackStatus.textContent = 'Statut: Erreur webcam';
        }
    });

    stopAttackBtn.addEventListener('click', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcam.srcObject = null;
        }
        if (intervalId) {
            clearInterval(intervalId);
        }
        stopFPSCounter();
        attackResultDiv.innerHTML = '<p>Simulation arrêtée.</p>';
        attackStatus.textContent = 'Statut: Arrêté';
        detectionCount.textContent = 'Objets détectés: 0';
        fpsCounter.textContent = 'FPS: 0';
    });

    function updateAttackStatus() {
        if (attackEnabled) {
            attackStatus.textContent = 'Statut: Attaque FGSM active';
            attackStatus.style.color = '#ff4444';
        } else {
            attackStatus.textContent = 'Statut: Mode normal';
            attackStatus.style.color = '#44ff44';
        }
    }

    function startFPSCounter() {
        lastFrameTime = performance.now();
        frameCount = 0;
    }

    function stopFPSCounter() {
        fpsCounter.textContent = 'FPS: 0';
    }

    function updateFPS() {
        const now = performance.now();
        const deltaTime = now - lastFrameTime;
        if (deltaTime >= 1000) { // Mise à jour chaque seconde
            const fps = Math.round((frameCount * 1000) / deltaTime);
            fpsCounter.textContent = `FPS: ${fps}`;
            frameCount = 0;
            lastFrameTime = now;
        }
        frameCount++;
    }

    function captureAndSendFrame() {
        if (!webcam.videoWidth || !webcam.videoHeight) return;

        const context = canvas.getContext('2d');
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        context.drawImage(webcam, 0, 0);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.png');
            formData.append('attack_enabled', attackEnabled.toString());
            formData.append('epsilon', epsilon.toString());

            try {
                const response = await fetch(API_BASE + '/attack', {
                    method: 'POST',
                    body: formData
                });

                const parsed = await readJsonFromResponse(response);

                if (!parsed.ok || parsed.data === null) {
                    attackResultDiv.innerHTML = '<p>' + parsed.message + '</p>';
                    attackStatus.textContent = 'Statut: Erreur serveur';
                    return;
                }

                const result = parsed.data;

                if (!response.ok) {
                    attackResultDiv.innerHTML =
                        '<p>Erreur: ' + (result.error || 'HTTP ' + response.status) + '</p>';
                    attackStatus.textContent = 'Statut: Erreur serveur';
                    return;
                }

                if (!result.image_url) {
                    attackResultDiv.innerHTML = '<p>Réponse JSON inattendue (pas d’image_url).</p>';
                    attackStatus.textContent = 'Statut: Erreur serveur';
                    return;
                }

                attackResultDiv.innerHTML =
                    `<img src="${result.image_url}" style="max-width: 100%; border: 2px solid #00aaff; border-radius: 8px;">`;
                detectionCount.textContent = `Objets détectés: ${result.detection_count || 0}`;
                updateFPS();
            } catch (error) {
                attackResultDiv.innerHTML =
                    '<p>Erreur réseau (serveur arrêté ou URL incorrecte) : ' +
                    error.message +
                    '</p>';
                attackStatus.textContent = 'Statut: Erreur connexion';
            }
        }, 'image/png');
    }
});