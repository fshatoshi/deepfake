
"""
Backend Flask pour SecurAI Store.
Expose le flux vidéo MJPEG et les API de contrôle avec multithreading.
"""
import os
import cv2
import time
import numpy as np
import threading
from dataclasses import dataclass
import logging
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template
from werkzeug.utils import secure_filename

# Import des modules d'intelligence
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.fgsm_attacker import FGSMAttacker
from modules.patch_attacker import PatchAttacker
from modules.defender import Defender
from modules.anomaly_detector import AnomalyDetector
from rights_manager import RightsManager

app = Flask(__name__)

# --- CONFIGURATION ---
from paths import BASE_DIR, MODELS_DIR, ENROLLED_DIR, AUDIT_LOG

os.makedirs(ENROLLED_DIR, exist_ok=True)

# --- LOGGING ---
LOG_FILE = AUDIT_LOG
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Système SecurAI démarré.")

# --- ÉTAT GLOBAL (Thread-safe) ---
@dataclass
class SystemState:
    identity: str = "Aucun"
    access_level: str = "DENIED"
    permissions: dict = None
    anomaly_detected: bool = False
    anomaly_score: float = 0.0
    attack_active: bool = False
    model_mode: str = "standard"
    fps: int = 0
    confidence: float = 0.0
    latest_frame: np.ndarray = None

state = SystemState()
state.permissions = {'entrance': False, 'stock': False, 'cashier': False, 'server': False}
state_lock = threading.Lock()

# --- INITIALISATION DES MODULES ---
print("Initialisation des modules IA en cours...")
face_detector = FaceDetector()
rights_manager = RightsManager()

# On ne charge plus class_names.json car on utilise les embeddings
face_recognizer = FaceRecognizer(mode='standard')
fgsm_attacker = FGSMAttacker(face_recognizer.model, epsilon=0.03)

# Auto-enrôlement des images présentes dans data/enrolled/
print("Enrôlement des visages connus...")
for filename in os.listdir(ENROLLED_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Le nom complet sans extension (ex: Manager_Demo)
        # On ignore ce qui suit un tiret '-' pour permettre plusieurs photos (ex: Manager_Demo-1.jpg)
        base_name = os.path.splitext(filename)[0]
        identity_name = base_name.split('-')[0]
        
        filepath = os.path.join(ENROLLED_DIR, filename)
        img = cv2.imread(filepath)
        if img is not None:
            # On détecte le visage dans l'image
            bboxes = face_detector.detect(img)
            if bboxes:
                face_crop = face_detector.crop_face(img, bboxes[0], size=160)
                if face_crop is not None:
                    face_recognizer.enroll_face(identity_name, face_crop)
                    
print(f"{len(face_recognizer.enrolled_embeddings)} identités enrôlées.")

defender = Defender()
anomaly_detector = AnomalyDetector()
patch_attacker = PatchAttacker(face_recognizer.model, epsilon=0.35, steps=40, alpha=0.02)
print("Modules prêts.")

# --- THREAD DE TRAITEMENT VIDEO ---
def video_processing_thread():
    camera = cv2.VideoCapture(0)
    # Résolution 640x480
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.1)
            continue
            
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            with state_lock:
                state.fps = int(frame_count / elapsed)
            frame_count = 0
            start_time = time.time()
            
        bboxes = face_detector.detect(frame)
        
        current_identity = "Inconnu"
        current_conf = 0.0
        is_attacked = False
        anom_score = 0.0
        
        with state_lock:
            attack_active = state.attack_active
            current_mode = state.model_mode
            
        for bbox in bboxes:
            face_crop = face_detector.crop_face(frame, bbox, size=128)
            if face_crop is None or face_recognizer is None:
                continue
                
            # Attaque
            if attack_active and frame_count % 3 == 0:
                # Si on connaît l'embedding du Manager_Demo, on fait une attaque ciblée vers lui
                target_emb = face_recognizer.enrolled_embeddings.get('Manager_Demo')
                face_crop = fgsm_attacker.attack(face_crop, target_emb)
                
            # Anomalie
            is_attacked_flag, anom_score = anomaly_detector.analyze(face_crop)
            is_attacked = is_attacked or is_attacked_flag
            
            # Défense
            if current_mode == 'hardened':
                face_crop = defender.apply_defense(face_crop, defense_type='gaussian')
                
            # Reconnaissance
            identity, conf = face_recognizer.predict(face_crop)
            if conf > current_conf:
                current_identity = identity
                current_conf = conf
                
            # Overlay Bbox
            x1, y1, x2, y2 = bbox
            ui_config = rights_manager.get_ui_config(identity)
            color_hex = ui_config['color'].lstrip('#')
            color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{identity} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Mise à jour de l'état global
        with state_lock:
            state.identity = current_identity
            state.confidence = current_conf
            state.anomaly_detected = is_attacked
            state.anomaly_score = anom_score
            state.access_level = rights_manager.get_access_level(current_identity)
            state.permissions = rights_manager.get_permissions(current_identity)
            
            # Log des alertes
            if is_attacked:
                logging.warning(f"ANOMALIE DÉTECTÉE - Score: {anom_score:.2f} - Identité suspectée: {current_identity}")
            
            # On stocke l'image encodée pour le flux MJPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                state.latest_frame = buffer.tobytes()

# Démarrage du thread
t = threading.Thread(target=video_processing_thread, daemon=True)
t.start()

# --- ROUTES FLASK ---

@app.route('/')
def index():
    return render_template('EntranceControl.html')

@app.route('/static_analysis')
def static_analysis():
    return render_template('StaticAnalysis.html')

@app.route('/api/analyze_static', methods=['POST'])
def analyze_static():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Aucune image envoyée"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400

    try:
        # Lecture de l'image depuis la requête
        in_memory_file = file.read()
        nparr = np.frombuffer(in_memory_file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Format d'image invalide"}), 400
            
        # Détection du visage
        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({
                "success": True, 
                "results": {"identity": "Aucun visage", "access_level": "DENIED", "confidence": 0, "anomaly_detected": False, "anomaly_score": 0.0}
            })
            
        # On prend le premier visage trouvé
        bbox = bboxes[0]
        face_crop = face_detector.crop_face(img, bbox, size=160)
        
        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur recadrage"}), 500
            
        # Détection d'anomalie (Spoofing)
        is_attacked, anom_score = anomaly_detector.analyze(face_crop)
        
        # Reconnaissance
        identity, conf = face_recognizer.predict(face_crop)
        
        # Récupération des droits
        access_level = rights_manager.get_access_level(identity)
        permissions = rights_manager.get_permissions(identity)
        
        # Overlay bbox sur l'image d'origine pour le retour
        x1, y1, x2, y2 = bbox
        ui_config = rights_manager.get_ui_config(identity)
        color_hex = ui_config['color'].lstrip('#')
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 3)
        cv2.putText(img, f"{identity} ({conf:.2f})", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
        
        # Encodage de l'image résultante en base64 pour affichage frontend
        import base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "results": {
                "identity": identity,
                "confidence": conf,
                "access_level": access_level,
                "permissions": permissions,
                "anomaly_detected": is_attacked,
                "anomaly_score": anom_score,
                "image_base64": img_base64
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def generate_mjpeg():
    while True:
        with state_lock:
            frame = state.latest_frame
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # ~30fps max

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status', methods=['GET'])
def get_status():
    with state_lock:
        return jsonify({
            'identity': state.identity,
            'access_level': state.access_level,
            'permissions': state.permissions,
            'anomaly_detected': state.anomaly_detected,
            'anomaly_score': state.anomaly_score,
            'attack_active': state.attack_active,
            'model_mode': state.model_mode,
            'fps': state.fps,
            'confidence': state.confidence
        })

@app.route('/api/toggle_attack', methods=['POST'])
def toggle_attack():
    data = request.json
    if 'active' in data:
        with state_lock:
            state.attack_active = data['active']
            status = "activée" if state.attack_active else "désactivée"
        return jsonify({"success": True, "message": f"Attaque {status}."})
    return jsonify({"success": False, "error": "Paramètre 'active' manquant."}), 400

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    data = request.json
    if 'mode' in data and data['mode'] in ['standard', 'hardened']:
        mode = data['mode']
        try:
            if face_recognizer:
                face_recognizer.switch_mode(mode)
            with state_lock:
                state.model_mode = mode
            return jsonify({"success": True, "message": f"Mode changé vers {mode}."})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    return jsonify({"success": False, "error": "Mode invalide."}), 400

@app.route('/api/enroll', methods=['POST'])
def enroll():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"success": False, "error": "Données manquantes"}), 400
        
    file = request.files['image']
    name = request.form['name']
    level = request.form.get('level', RightsManager.EMPLOYEE)
    
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide"}), 400
        
    filename = secure_filename(f"{name}_{file.filename}")
    save_path = os.path.join(ENROLLED_DIR, filename)
    file.save(save_path)
    
    # Ajout dynamique au RightsManager
    try:
        rights_manager.add_identity(name, level)
        return jsonify({"success": True, "message": f"{name} enrôlé comme {level}"})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/generate_glasses_attack', methods=['POST'])
def generate_glasses_attack():
    """
    Génère une image avec des lunettes adversariales pour tromper FaceNet.
    Paramètres (multipart/form-data):
        image  : Image du visage à attaquer.
        target : Nom de l'identité cible (doit être enrôlée, ex: 'Manager_Demo').
    """
    import base64

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Paramètre 'image' manquant."}), 400

    target_name = request.form.get('target', 'Manager_Demo')
    target_emb  = face_recognizer.enrolled_embeddings.get(target_name)

    if target_emb is None:
        available = list(face_recognizer.enrolled_embeddings.keys())
        return jsonify({
            "success": False,
            "error": f"Identité cible '{target_name}' non enrôlée.",
            "enrolled": available
        }), 404

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Fichier vide."}), 400

    try:
        in_memory = file.read()
        nparr = np.frombuffer(in_memory, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Format d'image invalide."}), 400

        # Détection du visage dans l'image uploadée
        bboxes = face_detector.detect(img)
        if not bboxes:
            return jsonify({"success": False, "error": "Aucun visage détecté dans l'image."}), 422

        bbox = bboxes[0]
        x1, y1, x2, y2 = bbox
        face_crop = face_detector.crop_face(img, bbox, size=160)

        if face_crop is None:
            return jsonify({"success": False, "error": "Erreur lors du recadrage du visage."}), 500

        # --- Analyse AVANT l'attaque ---
        id_before, conf_before = face_recognizer.predict(face_crop)

        # --- Application du patch lunettes adversarial ---
        logging.info(f"Génération lunettes adversariales: cible='{target_name}'")
        attacked_crop = patch_attacker.attack(face_crop, target_emb)

        # --- Analyse APRÈS l'attaque ---
        id_after, conf_after   = face_recognizer.predict(attacked_crop)
        is_anom, anom_score    = anomaly_detector.analyze(attacked_crop)
        access_after           = rights_manager.get_access_level(id_after)
        permissions_after      = rights_manager.get_permissions(id_after)

        # --- Reconstruction de l'image complète avec le crop attaqué ---
        result_img = img.copy()
        h_crop, w_crop = attacked_crop.shape[:2]
        # Recalcule les coords clampées comme dans crop_face
        img_h, img_w = img.shape[:2]
        rx1 = max(0, x1); ry1 = max(0, y1)
        rx2 = min(img_w, x2); ry2 = min(img_h, y2)
        face_patch_resized = cv2.resize(attacked_crop, (rx2 - rx1, ry2 - ry1))
        result_img[ry1:ry2, rx1:rx2] = face_patch_resized

        # Overlay bbox couleur selon accès APRÈS
        ui_cfg = rights_manager.get_ui_config(id_after)
        col_hex = ui_cfg['color'].lstrip('#')
        col_bgr = tuple(int(col_hex[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(result_img, (x1, y1), (x2, y2), col_bgr, 3)
        label = f"{id_after} ({conf_after:.2f}) [PATCH]"
        cv2.putText(result_img, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_bgr, 2)

        _, buf = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode('utf-8')

        logging.info(
            f"Résultat patch: {id_before}({conf_before:.2f}) → {id_after}({conf_after:.2f}) "
            f"| Anomalie={is_anom} score={anom_score:.3f}"
        )

        return jsonify({
            "success": True,
            "target": target_name,
            "before": {"identity": id_before, "confidence": conf_before},
            "after":  {
                "identity":         id_after,
                "confidence":       conf_after,
                "access_level":     access_after,
                "permissions":      permissions_after,
                "anomaly_detected": is_anom,
                "anomaly_score":    anom_score,
                "image_base64":     img_b64
            }
        })

    except Exception as e:
        logging.error(f"Erreur generate_glasses_attack: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
