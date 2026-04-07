import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms

TARGET_SIZE = 640

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
])


def fgsm_attack(image_tensor, model, epsilon):
    """
    Applique l'attaque FGSM sur un tenseur image [1,3,H,W] avec requires_grad=True.
    """
    output = model(image_tensor)[0]
    loss = output.abs().sum()
    model.zero_grad()
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
    loss.backward()
    grad_sign = image_tensor.grad.data.sign()
    perturbed_image = image_tensor + epsilon * grad_sign
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def preprocess_frame(frame_bgr):
    """
    Frame OpenCV BGR -> tenseur [1,3,H,W] pour FGSM (même pipeline que l'exécutable local).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(frame_rgb).unsqueeze(0)
    return tensor


def tensor_to_frame(tensor):
    """Tenseur [1,3,H,W] en [0,1] -> frame OpenCV BGR uint8."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def run_attack_pipeline(frame_bgr, attack_enabled, epsilon, yolo_model, pytorch_model):
    """
    Exécute la même chaîne que la section webcam de __main__ :
    FGSM (optionnel) -> YOLO sur la frame résultante -> frame annotée.

    Args:
        frame_bgr: numpy BGR (HxWx3), ex. issue d'OpenCV ou de cv2.cvtColor depuis RGB.
        yolo_model: instance ultralytics.YOLO
        pytorch_model: yolo_model.model (pour les gradients FGSM)

    Returns:
        (annotated_frame_bgr, detection_count)
    """
    if pytorch_model is None or yolo_model is None:
        raise RuntimeError("YOLO models are not loaded")

    if attack_enabled:
        tensor = preprocess_frame(frame_bgr)
        tensor = tensor.detach().clone().requires_grad_(True)
        perturbed_tensor = fgsm_attack(tensor, pytorch_model, epsilon)
        attack_input = perturbed_tensor.detach().requires_grad_(False)
        attacked_frame = tensor_to_frame(attack_input)
        results = yolo_model(attacked_frame)
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"FGSM ATTACK ACTIVE - epsilon={epsilon:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    else:
        results = yolo_model(frame_bgr)
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            "NORMAL MODE - No Attack",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    detection_count = len(results[0].boxes) if results and len(results) > 0 else 0
    return annotated_frame, detection_count


def _main():
    EPSILON = 0.015
    model = YOLO("yolov8n.pt")
    pytorch_model = model.model
    pytorch_model.eval()

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=== ATTAQUE FGSM EN TEMPS RÉEL ===")
    print(f"Paramètres : epsilon = {EPSILON}")
    print("Appuyez sur 'a' pour activer/désactiver l'attaque")
    print("Appuyez sur '+' ou '-' pour augmenter/réduire epsilon")
    print("Appuyez sur 'q' pour quitter")

    attack_enabled = True
    epsilon = EPSILON

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, _ = run_attack_pipeline(
            frame, attack_enabled, epsilon, model, pytorch_model
        )

        cv2.putText(
            annotated_frame,
            "Press 'a': toggle attack | '+/-': epsilon | 'q': quit",
            (10, frame_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("YOLOv8 - FGSM Adversarial Attack", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("a"):
            attack_enabled = not attack_enabled
            print(f"Attaque FGSM: {'ACTIVEE' if attack_enabled else 'DESACTIVEE'}")
        elif key == ord("+") or key == ord("="):
            epsilon = min(0.1, epsilon + 0.005)
            print(f"Epsilon augmenté à : {epsilon:.3f}")
        elif key == ord("-") or key == ord("_"):
            epsilon = max(0.0, epsilon - 0.005)
            print(f"Epsilon diminué à : {epsilon:.3f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _main()
