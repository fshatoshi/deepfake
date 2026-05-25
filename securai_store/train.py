"""
Entraînement FaceNet (fine-tuning) — checkpoints Drive toutes les 30 min.
Colab : cd $SECURAI_BASE && nohup python -u train.py > logs/train.log 2>&1 &
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from paths import BASE_DIR, CHECKPOINTS_DIR, ENROLLED_DIR, TRAIN_LOG

CKPT_DIR = CHECKPOINTS_DIR
DATASET_DIR = ENROLLED_DIR
LOG_PATH = TRAIN_LOG

MAX_CKPTS = 3
SAVE_EVERY = 1800  # 30 min
BATCH_SIZE = 16
LR = 1e-4
IMAGE_SIZE = 160


def setup_logging() -> logging.Logger:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_checkpoint(model, optimizer, classifier, epoch, loss):
    os.makedirs(CKPT_DIR, exist_ok=True)
    ts = int(time.time())
    path = f"{CKPT_DIR}/ckpt_epoch{epoch:04d}_{ts}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "classifier_state": classifier.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    for old in sorted(glob.glob(f"{CKPT_DIR}/ckpt_*.pt"))[:-MAX_CKPTS]:
        os.remove(old)
    print(f"[CKPT] Sauvé : {os.path.basename(path)}", flush=True)


def load_latest_checkpoint(model, optimizer, classifier):
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/ckpt_*.pt"))
    if not ckpts:
        print("[CKPT] Aucun checkpoint — démarrage à zéro.", flush=True)
        return 0
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "classifier_state" in ckpt:
        classifier.load_state_dict(ckpt["classifier_state"])
    print(f"[CKPT] Reprise depuis : {os.path.basename(ckpts[-1])}", flush=True)
    return int(ckpt["epoch"])


class FaceImageDataset(Dataset):
    """Images par identité : data/enrolled/<personne>/*.jpg ou fichiers plats."""

    EXT = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        root_path = Path(root)
        if not root_path.is_dir():
            return

        subdirs = [p for p in root_path.iterdir() if p.is_dir()]
        if subdirs:
            for label, person_dir in enumerate(sorted(subdirs)):
                for img in person_dir.rglob("*"):
                    if img.suffix.lower() in self.EXT:
                        self.samples.append((str(img), label))
        else:
            names: dict[str, int] = {}
            for img in root_path.rglob("*"):
                if img.is_file() and img.suffix.lower() in self.EXT:
                    identity = img.stem.split("-")[0]
                    if identity not in names:
                        names[identity] = len(names)
                    self.samples.append((str(img), names[identity]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_model(device: torch.device):
    from facenet_pytorch import InceptionResnetV1

    backbone = InceptionResnetV1(pretrained="vggface2").eval()
    for p in backbone.parameters():
        p.requires_grad = False
    for p in backbone.last_linear.parameters():
        p.requires_grad = True
    for p in backbone.last_bn.parameters():
        p.requires_grad = True
    return backbone.to(device)


def train_loop():
    log = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | BASE_DIR: %s", device, BASE_DIR)

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = FaceImageDataset(DATASET_DIR, transform)
    if len(dataset) == 0:
        log.error(
            "Aucune image dans %s — ajoute des photos (data/enrolled/<nom>/*.jpg).",
            DATASET_DIR,
        )
        sys.exit(1)

    num_classes = max(label for _, label in dataset.samples) + 1
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    log.info("Dataset: %d images, %d classes", len(dataset), num_classes)

    model = build_model(device)
    classifier = nn.Linear(512, num_classes).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(
        classifier.parameters()
    )
    optimizer = torch.optim.Adam(trainable, lr=LR)

    start_epoch = load_latest_checkpoint(model, optimizer, classifier)
    epoch = start_epoch
    last_ckpt_time = time.time()
    criterion = nn.CrossEntropyLoss()

    log.info("Entraînement démarré (epoch %d). Ctrl+C pour arrêter proprement.", epoch)

    try:
        while True:
            epoch += 1
            model.train()
            classifier.train()
            running_loss = 0.0
            n_batches = 0

            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                emb = model(images)
                emb = F.normalize(emb, p=2, dim=1)
                logits = classifier(emb)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(n_batches, 1)
            log.info("Epoch %d — loss: %.4f", epoch, avg_loss)

            if time.time() - last_ckpt_time >= SAVE_EVERY:
                save_checkpoint(model, optimizer, classifier, epoch, avg_loss)
                last_ckpt_time = time.time()

    except KeyboardInterrupt:
        log.info("Interruption — sauvegarde finale.")
        final_loss = running_loss / max(n_batches, 1) if n_batches else 0.0
        save_checkpoint(model, optimizer, classifier, epoch, final_loss)


if __name__ == "__main__":
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_loop()
