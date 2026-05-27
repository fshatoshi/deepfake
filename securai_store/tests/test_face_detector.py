"""
tests/test_face_detector.py
────────────────────────────────────────────────────────────────────────────────
Tests unitaires pour modules/face_detector.py (FaceDetector / YOLOv8-face).

Stratégie : on mocke YOLO pour ne pas dépendre du fichier yolov8n-face.pt
ni d'un GPU — tous les tests s'exécutent en environnement CI pur Python.
"""

import sys

# ── Vider les stubs éventuellement injectés par test_endpoints.py ────────────
# pytest collecte test_endpoints.py en premier (ordre alpha) et injecte des
# faux modules dans sys.modules. On les supprime ici pour que ce fichier
# importe le VRAI modules.face_detector.
for _k in list(sys.modules):
    if _k.startswith("modules."):
        sys.modules.pop(_k, None)

import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# ── Stubs légers pour ultralytics (absente en CI) ────────────────────────────
_ultra = types.ModuleType("ultralytics")

class _FakeBox:
    """Simule ultralytics.engine.results.Boxes pour une détection.
    box.xyxy[0].cpu().numpy().astype(int) doit retourner un np.array — on
    utilise un MagicMock chaîné pour éviter tout appel torch→numpy."""
    def __init__(self, x1, y1, x2, y2):
        _arr  = np.array([x1, y1, x2, y2], dtype=np.float32)
        _mock = MagicMock()
        _mock.cpu.return_value.numpy.return_value = _arr
        self.xyxy = [_mock]

class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes

class _FakeYOLO:
    """YOLO bouchon : retourne une détection fixe ou aucune selon le mode."""
    def __init__(self, path, **kw):
        self._mode = "one_face"          # "one_face" | "no_face" | "multi"

    def __call__(self, frame, **kw):
        if self._mode == "no_face":
            return [_FakeResult([])]
        if self._mode == "multi":
            return [_FakeResult([
                _FakeBox(10, 10, 90, 90),
                _FakeBox(100, 10, 180, 90),
            ])]
        # default: one face
        return [_FakeResult([_FakeBox(20, 30, 100, 110)])]

_ultra.YOLO = _FakeYOLO
sys.modules.setdefault("ultralytics", _ultra)

# ── Après avoir injecté le stub, on peut importer FaceDetector ───────────────
# On patch aussi pathlib.Path.is_file et stat pour simuler un fichier présent.
_real_is_file = Path.is_file

def _patched_is_file(self):
    if "yolov8n-face" in str(self):
        return True
    return _real_is_file(self)

_real_stat = Path.stat

class _FakeStat:
    st_size = 5_000_000   # > 1 Mo — passe la vérif de taille

def _patched_stat(self, *a, **kw):
    if "yolov8n-face" in str(self):
        return _FakeStat()
    return _real_stat(self, *a, **kw)

# Patch actif pendant toute la session de test
pytestmark = pytest.mark.usefixtures()

with (
    patch.object(Path, "is_file", _patched_is_file),
    patch.object(Path, "stat",    _patched_stat),
):
    # Ajout du dossier securai_store au sys.path
    _store = Path(__file__).parent.parent
    if str(_store) not in sys.path:
        sys.path.insert(0, str(_store))
    from modules.face_detector import FaceDetector


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """Instance FaceDetector avec YOLO mocké — partagée pour toute la classe."""
    with (
        patch.object(Path, "is_file", _patched_is_file),
        patch.object(Path, "stat",    _patched_stat),
    ):
        d = FaceDetector(model_path="yolov8n-face.pt", frame_skip=1)
    return d

@pytest.fixture
def blank_frame():
    """Frame 200×200 BGR noire."""
    return np.zeros((200, 200, 3), dtype=np.uint8)

@pytest.fixture
def color_frame():
    """Frame 200×200 BGR avec une zone claire simulant un visage."""
    f = np.zeros((200, 200, 3), dtype=np.uint8)
    f[30:110, 20:100] = (180, 160, 140)   # zone « chair »
    return f


# ── Tests detect() ────────────────────────────────────────────────────────────

class TestDetect:

    def test_returns_list(self, detector, blank_frame):
        """detect() doit toujours retourner une liste."""
        result = detector.detect(blank_frame)
        assert isinstance(result, list)

    def test_one_face_detected(self, detector, color_frame):
        """En mode par défaut YOLO retourne 1 bbox."""
        detector.model._mode = "one_face"
        bboxes = detector.detect(color_frame)
        assert len(bboxes) == 1

    def test_bbox_has_four_ints(self, detector, color_frame):
        """Chaque bbox est un tuple / liste de 4 entiers."""
        detector.model._mode = "one_face"
        bboxes = detector.detect(color_frame)
        x1, y1, x2, y2 = bboxes[0]
        for v in (x1, y1, x2, y2):
            assert isinstance(v, (int, np.integer))

    def test_no_face_returns_empty(self, detector, blank_frame):
        """Sans visage, detect() renvoie une liste vide."""
        detector.model._mode = "no_face"
        bboxes = detector.detect(blank_frame)
        assert bboxes == []
        detector.model._mode = "one_face"   # reset

    def test_multi_face_returns_multiple(self, detector, color_frame):
        """Plusieurs visages → plusieurs bboxes."""
        detector.model._mode = "multi"
        bboxes = detector.detect(color_frame)
        assert len(bboxes) == 2
        detector.model._mode = "one_face"   # reset

    def test_frame_skip_caches_result(self, detector, color_frame):
        """Avec frame_skip=3, detect() doit retourner les mêmes bboxes
        en frames intermédiaires (résultat en cache)."""
        d = FaceDetector.__new__(FaceDetector)
        d.model       = detector.model
        d.frame_skip  = 3
        d.frame_count = 0
        d.last_results = []

        first  = d.detect(color_frame)
        second = d.detect(color_frame)   # frame intermédiaire — cache
        assert first == second


# ── Tests crop_face() ─────────────────────────────────────────────────────────

class TestCropFace:

    def test_crop_correct_size(self, detector, color_frame):
        """crop_face() doit retourner un array de taille (size, size, 3)."""
        bbox = (20, 30, 100, 110)
        crop = detector.crop_face(color_frame, bbox, size=128)
        assert crop is not None
        assert crop.shape == (128, 128, 3)

    def test_custom_size(self, detector, color_frame):
        """Le paramètre size doit être respecté."""
        bbox = (20, 30, 100, 110)
        crop = detector.crop_face(color_frame, bbox, size=160)
        assert crop.shape == (160, 160, 3)

    def test_returns_ndarray(self, detector, color_frame):
        """Le résultat est bien un numpy array."""
        bbox = (20, 30, 100, 110)
        crop = detector.crop_face(color_frame, bbox, size=64)
        assert isinstance(crop, np.ndarray)

    def test_out_of_bounds_bbox_clipped(self, detector, color_frame):
        """Bbox partiellement hors-image : doit retourner un crop valide
        (pas None, pas de crash)."""
        bbox = (-10, -10, 50, 50)
        crop = detector.crop_face(color_frame, bbox, size=64)
        assert crop is not None
        assert crop.shape == (64, 64, 3)

    def test_zero_area_returns_none(self, detector, color_frame):
        """Bbox de surface nulle → retourner None plutôt que crasher."""
        bbox = (50, 50, 50, 50)
        crop = detector.crop_face(color_frame, bbox, size=64)
        assert crop is None

    def test_fully_outside_bbox_returns_none(self, detector, color_frame):
        """Bbox entièrement hors de l'image → None."""
        bbox = (300, 300, 400, 400)
        crop = detector.crop_face(color_frame, bbox, size=64)
        assert crop is None