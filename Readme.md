# Face Recognition & Adversarial Attack Framework

A deep learning research project exploring the vulnerabilities of face recognition systems through adversarial attacks. This framework demonstrates how carefully crafted perturbations can deceive CNN-based facial recognition models, and provides tools for both training robust models and simulating real-time adversarial attacks.

## Overview

This project implements an end-to-end pipeline for:

- **Face Recognition** — Training a custom CNN architecture on the LFW (Labeled Faces in the Wild) dataset for identity classification.
- **Adversarial Attack Simulation** — Generating adversarial perturbations (FGSM-based) that can fool the trained model in real-time using a webcam feed.
- **Defense Analysis** — Evaluating model robustness under varying attack strengths (epsilon values).
- **Web Interface** — An interactive dashboard to visualize attack results and test the model via image upload.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch, TorchVision |
| Face Detection | YOLOv8 (Ultralytics) |
| Backend API | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data Processing | NumPy, OpenCV, Pillow |
| Visualization | Matplotlib |
| Dataset | LFW-Deepfunneled |

## Project Structure

```
Deepfake_Project/
├── backend/
│   ├── api/
│   │   ├── app.py              # Flask API server (prediction + attack endpoints)
│   │   └── routes.py           # API route definitions
│   ├── model/
│   │   ├── cnn_architecture.py # FaceCNN model definition (Conv2D layers + FC)
│   │   └── utils.py            # DataManager, transforms, dataset loading
│   ├── models/
│   │   ├── saved_model/        # Trained model checkpoints
│   │   └── saved_models/       # Additional model snapshots
│   ├── simulate.py             # Real-time adversarial attack pipeline (webcam)
│   └── train_model.py          # Alternative training entry point
├── Training/
│   ├── train.py                # Main training script (with progress bars & plots)
│   └── evaluate_model.py       # Model evaluation and accuracy metrics
├── Frontend/
│   ├── index.html              # Main web interface
│   ├── Css/
│   │   └── Style.css           # Dashboard styling
│   ├── Js/
│   │   ├── script.js           # Core UI logic
│   │   ├── main.js             # App initialization
│   │   ├── face_recognition.js # Face recognition module
│   │   ├── attack_simulation.js# Attack visualization
│   │   └── protection.js       # Defense demo module
│   └── Assets/                 # Images and media files
├── data/
│   └── lfw-deepfunneled/       # LFW face dataset (not included in repo)
├── Requirements.txt            # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.10 and 3.12)
- pip
- Git
- A webcam (for real-time attack simulation)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/nadahemed/Face-Recognition-Adversarial-Attacks.git
cd Face-Recognition-Adversarial-Attacks
```

2. **Create and activate a virtual environment**

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r Requirements.txt
```

### Training the Model

```bash
python -m Training.train --data_dir data/lfw-deepfunneled --epochs 30 --batch_size 32
```

Training progress is displayed with `tqdm` progress bars, and a loss/accuracy plot is saved as `training_history_cpu.png` at the end.

### Evaluating the Model

```bash
python -m Training.evaluate_model --model_path backend/models/saved_model
```

### Running the Application

1. **Start the Flask backend**

```bash
python -m backend.api.app
```

The API will be available at `http://localhost:5000`.

2. **Open the frontend**

Open `Frontend/index.html` in your browser, or serve it with any static file server:

```bash
# Quick serve with Python
python -m http.server 8080 --directory Frontend
```

Then navigate to `http://localhost:8080`.

3. **Run real-time attack simulation** (requires webcam)

```bash
python -m backend.simulate
```

This opens a webcam feed where:
- YOLOv8 detects faces in real-time
- The CNN model classifies detected faces
- FGSM adversarial perturbations are applied with adjustable epsilon
- Press `q` to quit

## How It Works

### CNN Architecture

The `FaceCNN` model uses a series of convolutional layers followed by fully connected layers to classify face identities. The architecture is optimized for the LFW dataset resolution.

### FGSM Attack

The **Fast Gradient Sign Method** computes the gradient of the loss with respect to the input image, then adds a small perturbation in the direction that maximizes the loss:

```
adversarial_image = original_image + epsilon * sign(gradient)
```

A higher `epsilon` value produces stronger attacks but more visible distortions.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Classify an uploaded face image |
| POST | `/attack` | Apply adversarial attack and return results |
| GET | `/health` | API health check |

## Configuration

Key parameters can be adjusted:

- **Epsilon** (attack strength): `0.01` to `0.3` — higher values = stronger perturbation
- **Epochs** (training): default `30`
- **Batch size**: default `32`
- **Image size**: `128x128` (resized during preprocessing)

## License

This project is developed for academic and research purposes.
