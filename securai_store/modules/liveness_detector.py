"""
modules/liveness_detector.py — T5 SecurAI
────────────────────────────────────────────────────────────────────────────────
Détection de liveness par analyse de texture LBP (Local Binary Patterns).

Principe
────────
Un vrai visage a une texture complexe (pores, relief, variations fines).
Une photo imprimée ou affichée sur écran a une texture plus uniforme
avec des patterns périodiques (trame d'impression, pixels d'écran).

LBP mesure la micro-texture locale → on compare l'histogramme LBP
à des seuils empiriques pour décider "vrai" vs "spoofing".

Score retourné : 0.0 (certain spoof) → 1.0 (certain réel)
Seuil par défaut : 0.5
"""

from __future__ import annotations

import logging
import numpy as np
import cv2

from skimage.feature import local_binary_pattern


# ── Paramètres LBP ────────────────────────────────────────────────────────────
_LBP_RADIUS   = 3       # rayon du voisinage
_LBP_N_POINTS = 24      # 8 * radius — standard "uniform LBP"
_LBP_METHOD   = "uniform"


class LivenessDetector:
    """
    Détecteur de liveness basé sur LBP.

    Usage
    ─────
        detector = LivenessDetector(threshold=0.5)
        is_real, score, reason = detector.analyze(face_crop_bgr)
    """

    def __init__(self, threshold: float = 0.30, upper_threshold: float = 0.46):
        self.threshold = threshold
        self.upper_threshold = upper_threshold
        logging.info(f"[LivenessDetector] Initialisé (seuil={threshold}, upper={upper_threshold})")

    # ── API publique ──────────────────────────────────────────────────────────

    def analyze(self, face_crop: np.ndarray) -> tuple[bool, float, str]:
        """
        Analyse un crop de visage et détecte si c'est un vrai visage.

        Parameters
        ----------
        face_crop : image BGR (H×W×3), typiquement 160×160

        Returns
        -------
        (is_real, score, reason)
          is_real : True si vrai visage, False si photo/écran
          score   : 0.0 (spoof) → 1.0 (réel)
          reason  : explication courte
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0, "Crop invalide"

        gray = self._to_gray(face_crop)

        # Les trois indicateurs LBP
        variance_score = self._texture_variance_score(gray)
        entropy_score  = self._lbp_entropy_score(gray)
        freq_score     = self._frequency_score(gray)

        # Score global : moyenne pondérée
        score = (
            0.4 * variance_score +
            0.4 * entropy_score  +
            0.2 * freq_score
        )
        score = float(np.clip(score, 0.0, 1.0))

        is_real = self.threshold <= score <= self.upper_threshold

        if is_real:
            reason = f"Visage réel (score={score:.2f})"
        else:
            # Déterminer la cause principale
            weakest = min(
                [("texture", variance_score),
                 ("entropie LBP", entropy_score),
                 ("fréquence", freq_score)],
                key=lambda x: x[1]
            )
            reason = f"Spoofing probable — {weakest[0]} faible (score={score:.2f})"

        return is_real, score, reason

    # ── Indicateurs internes ──────────────────────────────────────────────────

    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        """Convertit BGR → niveaux de gris."""
        if bgr.ndim == 2:
            return bgr
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _texture_variance_score(self, gray: np.ndarray) -> float:
        """
        Variance de la texture LBP.
        Un vrai visage a une variance élevée (texture riche).
        Une photo a une variance faible (texture lisse/uniforme).
        """
        lbp = local_binary_pattern(
            gray, _LBP_N_POINTS, _LBP_RADIUS, method=_LBP_METHOD
        )
        # Variance normalisée par rapport à une valeur de référence empirique
        variance = float(np.var(lbp))
        # Empirique : vrai visage ~2500-8000, photo ~500-1500
        score = np.clip((variance - 400) / 5000, 0.0, 1.0)
        return float(score)

    def _lbp_entropy_score(self, gray: np.ndarray) -> float:
        """
        Entropie de l'histogramme LBP.
        Vrai visage → distribution variée → haute entropie.
        Photo       → distribution piquée → faible entropie.
        """
        lbp = local_binary_pattern(
            gray, _LBP_N_POINTS, _LBP_RADIUS, method=_LBP_METHOD
        )
        n_bins = _LBP_N_POINTS + 2   # uniform LBP : n_points + 2 bins
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)
        # Entropie de Shannon
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        # Normaliser : max théorique = log2(n_bins)
        max_entropy = np.log2(n_bins)
        score = np.clip(entropy / max_entropy, 0.0, 1.0)
        return float(score)

    def _frequency_score(self, gray: np.ndarray) -> float:
        """
        Analyse fréquentielle (FFT).
        Un écran ou une photo imprimée génère des patterns périodiques
        (trames, pixels) visibles dans le spectre fréquentiel.
        Vrai visage → énergie distribuée uniformément.
        Photo/écran → pics périodiques dans les hautes fréquences.
        """
        # FFT 2D
        fft    = np.fft.fft2(gray.astype(np.float32))
        fft_sh = np.fft.fftshift(fft)
        mag    = np.log1p(np.abs(fft_sh))

        h, w   = mag.shape
        cx, cy = w // 2, h // 2

        # Ratio énergie basses fréquences / hautes fréquences
        r = min(h, w) // 6
        low_mask  = np.zeros_like(mag, dtype=bool)
        low_mask[cy - r:cy + r, cx - r:cx + r] = True

        low_energy  = float(mag[low_mask].mean())
        high_energy = float(mag[~low_mask].mean())

        if high_energy < 1e-6:
            return 0.5

        # Vrai visage : ratio ~2-4 | Photo : ratio peut être < 1.5 ou > 6
        ratio = low_energy / (high_energy + 1e-6)
        # Score maximal autour du ratio attendu (2.5)
        score = np.exp(-0.3 * (ratio - 2.5) ** 2)
        return float(np.clip(score, 0.0, 1.0))