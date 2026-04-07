from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import io
import cv2
import numpy as np
from ultralytics import YOLO

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from backend.model.cnn_architecture import FaceCNN
    print("FaceCNN import successful")
except ImportError as e:
    print(f"FaceCNN import failed: {e}")
    FaceCNN = None

from backend.simulate import run_attack_pipeline

app = Flask(__name__)
CORS(app)

frontend_dir = os.path.join(project_root, "Frontend")

model_path = os.path.join(project_root, "backend", "models", "saved_models", "best_model.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    num_classes = len(checkpoint["class_to_idx"])
    model = FaceCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Face model loaded successfully")
else:
    model = None
    print("Face model not found")

model_yolo = None
pytorch_model = None
try:
    yolo_path = os.path.join(project_root, "yolov8n.pt")
    if os.path.exists(yolo_path):
        model_yolo = YOLO(yolo_path)
        pytorch_model = model_yolo.model
        pytorch_model.eval()
        print("YOLO model loaded successfully")
    else:
        print(f"YOLO model file not found at {yolo_path}")
except Exception as e:
    print(f"YOLO model not available: {e}")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@app.route("/")
def index():
    return send_from_directory(frontend_dir, "index.html")


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({"message": "Deepfake Detection API", "status": "running"})


@app.route("/Css/<path:filename>")
def css(filename):
    return send_from_directory(os.path.join(frontend_dir, "Css"), filename)


@app.route("/Js/<path:filename>")
def js(filename):
    return send_from_directory(os.path.join(frontend_dir, "Js"), filename)


@app.route("/detect", methods=["POST"])
def detect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = idx_to_class[predicted.item()]
            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()

        return jsonify(
            {
                "prediction": predicted_class,
                "confidence": round(confidence * 100, 2),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(silent=True) or {}
    sim_type = data.get("type", "cnn-scratch")
    return jsonify(
        {
            "message": f"Simulation lancée avec {sim_type}",
            "status": "success",
        }
    )


@app.route("/attack", methods=["POST"])
def attack():
    if model_yolo is None or pytorch_model is None:
        return jsonify(
            {
                "error": "YOLO indisponible. Placez yolov8n.pt à la racine du projet "
                "et installez ultralytics.",
            }
        ), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    attack_enabled = request.form.get("attack_enabled", "true").lower() == "true"
    epsilon = float(request.form.get("epsilon", "0.015"))

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_np = np.array(image)
        frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        annotated_frame, detection_count = run_attack_pipeline(
            frame_bgr, attack_enabled, epsilon, model_yolo, pytorch_model
        )

        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        annotated_pil.save(buf, format="PNG")
        buf.seek(0)

        import base64

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_url = f"data:image/png;base64,{image_base64}"

        return jsonify(
            {
                "image_url": image_url,
                "detection_count": detection_count,
                "attack_enabled": attack_enabled,
                "epsilon": epsilon,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/protect", methods=["POST"])
def protect():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = idx_to_class[predicted.item()]

        return jsonify(
            {
                "message": f"Protection appliquée pour {predicted_class}",
                "status": "protected",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
