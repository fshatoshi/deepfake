#!/usr/bin/env python3
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Test imports
try:
    from backend.model.cnn_architecture import FaceCNN
    print("FaceCNN import successful")
except ImportError as e:
    print(f"FaceCNN import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    from ultralytics import YOLO
    print("YOLO import successful")
except ImportError as e:
    print(f"YOLO import failed: {e}")

# Test model loading
try:
    model_path = os.path.join(project_root, 'backend', 'models', 'saved_models', 'best_model.pth')
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")

    if os.path.exists(model_path):
        import torch
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        print("Model loaded successfully")
        print(f"Classes: {len(checkpoint['class_to_idx'])}")
    else:
        print("Model file not found")

except Exception as e:
    print(f"Model loading failed: {e}")
    import traceback
    traceback.print_exc()