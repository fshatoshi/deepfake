"""
Adversarial Patch Attacker pour SecurAI.
Génère un patch en forme de lunettes qui trompe FaceNet pour usurper
une identité cible, sans modifier l'image entière (attaque localisée).
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2


class PatchAttacker:
    """
    Implémente une attaque par patch adversarial localisé en forme de lunettes.
    Seuls les pixels sous le masque sont modifiés, simulant une paire de lunettes
    imprimées / portées physiquement.
    """

    def __init__(self, model, epsilon: float = 0.35, steps: int = 40, alpha: float = 0.02):
        """
        Args:
            model:   Le modèle FaceNet (InceptionResnetV1) avec .eval() déjà appelé.
            epsilon: Amplitude maximale de la perturbation (plus grand = lunettes plus visibles).
            steps:   Nombre d'itérations PGD.
            alpha:   Taille du pas par itération.
        """
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = next(model.parameters()).device

        # Normalisation FaceNet
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)

    # ------------------------------------------------------------------
    # Masque Lunettes
    # ------------------------------------------------------------------
    def create_glasses_mask(self, h: int = 160, w: int = 160) -> np.ndarray:
        """
        Génère un masque binaire (h x w) représentant une monture de lunettes.
        La forme est calée sur un visage recadré 160x160 typique de FaceNet.

        Retourne un tableau (h, w) de float32 : 1.0 = zone attaquable, 0.0 = zone protégée.
        """
        mask = np.zeros((h, w), dtype=np.float32)

        # --- Dimensions dynamiques selon la taille ---
        cx = w // 2               # Centre X
        ey = int(h * 0.38)        # Hauteur des yeux (~38% du haut du visage recadré)
        eye_ry = int(h * 0.10)    # Rayon vertical des verres
        eye_rx = int(w * 0.18)    # Rayon horizontal des verres
        eye_sep = int(w * 0.22)   # Demi-distance entre les deux yeux

        # Verre gauche (ellipse)
        lx = cx - eye_sep
        cv2.ellipse(mask, (lx, ey), (eye_rx, eye_ry), 0, 0, 360, 1.0, -1)

        # Verre droit (ellipse)
        rx = cx + eye_sep
        cv2.ellipse(mask, (rx, ey), (eye_rx, eye_ry), 0, 0, 360, 1.0, -1)

        # Pont du nez (fine barre horizontale entre les deux verres)
        bridge_y1 = ey - int(eye_ry * 0.3)
        bridge_y2 = ey + int(eye_ry * 0.3)
        bridge_x1 = lx + eye_rx
        bridge_x2 = rx - eye_rx
        if bridge_x2 > bridge_x1:
            mask[bridge_y1:bridge_y2, bridge_x1:bridge_x2] = 1.0

        # Branches gauche et droite
        branch_thick = int(h * 0.045)
        branch_y1 = ey - branch_thick // 2
        branch_y2 = ey + branch_thick // 2
        # Gauche (vers bord gauche)
        mask[branch_y1:branch_y2, 0: lx - eye_rx] = 1.0
        # Droite (vers bord droit)
        mask[branch_y1:branch_y2, rx + eye_rx: w] = 1.0

        return mask

    # ------------------------------------------------------------------
    # Pré / post traitement
    # ------------------------------------------------------------------
    def _preprocess(self, bgr_crop: np.ndarray) -> torch.Tensor:
        """BGR numpy (H,W,3) → tenseur normalisé (1,3,160,160)."""
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (160, 160))
        t = torch.from_numpy(rgb_resized.transpose(2, 0, 1)).float() / 255.0
        t = (t - self.mean.cpu().squeeze(0)) / self.std.cpu().squeeze(0)
        return t.unsqueeze(0).to(self.device)

    def _postprocess(self, tensor: torch.Tensor, original_size) -> np.ndarray:
        """Tenseur normalisé → BGR numpy redimensionné à original_size (w, h)."""
        out = tensor * self.std + self.mean
        out = torch.clamp(out, 0.0, 1.0)
        np_rgb = out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        bgr = cv2.cvtColor((np_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if original_size != (160, 160):
            bgr = cv2.resize(bgr, original_size)
        return bgr

    # ------------------------------------------------------------------
    # Attaque principale
    # ------------------------------------------------------------------
    def attack(self, face_bgr: np.ndarray, target_embedding: torch.Tensor) -> np.ndarray:
        """
        Applique l'attaque adversariale localisée (masque lunettes) sur un crop BGR.

        Args:
            face_bgr:         Image du visage en BGR (numpy, n'importe quelle taille).
            target_embedding: Embedding 512D normalisé du visage cible (ex: Manager_Demo).

        Retourne l'image BGR avec les lunettes adversariales incrustées.
        """
        if target_embedding is None:
            return face_bgr

        orig_h, orig_w = face_bgr.shape[:2]
        img_tensor_orig = self._preprocess(face_bgr)             # (1,3,160,160)
        mask_np = self.create_glasses_mask(160, 160)             # (160,160)

        # Masque → tenseur (1,3,160,160) pour pouvoir masquer pixel par pixel
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,160,160)
        mask_t = mask_t.expand_as(img_tensor_orig)                    # (1,3,160,160)

        # On optimise uniquement la perturbation, pas l'image complète
        perturbation = torch.zeros_like(img_tensor_orig, requires_grad=False)

        for step in range(self.steps):
            perturbation.requires_grad_(True)

            # Appliquer la perturbation uniquement sous le masque
            adv_tensor = img_tensor_orig + perturbation * mask_t
            adv_tensor = torch.clamp(adv_tensor, -1.0, 1.0)

            embedding = self.model(adv_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
            target_embedding = target_embedding.to(self.device)

            # Perte : minimiser la distance cosinus avec la cible
            loss = 1.0 - F.cosine_similarity(embedding, target_embedding)

            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                # Descente de gradient projetée (PGD) sur la perturbation masquée
                grad_sign = perturbation.grad.sign()
                perturbation = perturbation - self.alpha * grad_sign  # signe - pour minimiser
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)

        # Reconstruction de l'image finale avec le patch
        with torch.no_grad():
            adv_final = img_tensor_orig + perturbation.detach() * mask_t
            adv_final = torch.clamp(adv_final, -1.0, 1.0)

        result_bgr = self._postprocess(adv_final, (orig_w, orig_h))
        return result_bgr

    # ------------------------------------------------------------------
    # Utilitaire : visualisation du masque seul
    # ------------------------------------------------------------------
    def get_glasses_overlay(self, size: int = 160) -> np.ndarray:
        """
        Retourne une image BGRA (avec canal alpha) du masque lunettes,
        utile pour l'affichage dans le dashboard.
        """
        mask = self.create_glasses_mask(size, size)
        overlay = np.zeros((size, size, 4), dtype=np.uint8)
        # Couleur cyan semi-transparente pour visualisation
        overlay[mask == 1.0] = [255, 200, 0, 180]  # BGR + Alpha
        return overlay

    def get_colored_mask_preview(self, size: int = 160) -> np.ndarray:
        """
        Retourne une image BGR colorée du masque pour preview base64.
        """
        mask = self.create_glasses_mask(size, size)
        preview = np.zeros((size, size, 3), dtype=np.uint8)
        preview[mask == 1.0] = [0, 200, 255]  # Cyan
        return preview
