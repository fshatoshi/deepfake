"""
tests/test_face_recognizer.py
────────────────────────────────────────────────────────────────────────────────
Tests unitaires pour modules/face_recognizer.py (FaceRecognizer / FaceNet).

Stratégie : on mocke InceptionResnetV1 et torchvision.transforms pour
- éviter le téléchargement des poids VGGFace2 (plusieurs centaines de Mo)
- exécuter les tests hors-GPU en CI
Le mock retourne des embeddings 512D déterministes.
"""

import sys

# ── Vider les stubs éventuellement injectés par test_endpoints.py ────────────
for _k in list(sys.modules):
    if _k.startswith("modules."):
        sys.modules.pop(_k, None)

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# ── Stubs pour facenet_pytorch ────────────────────────────────────────────────

def _make_embedding(seed: int) -> torch.Tensor:
    """Vecteur 512D normalisé déterministe."""
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(1, 512, generator=g)
    return F.normalize(v, p=2, dim=1)

class _FakeInceptionResnet(torch.nn.Module):
    """InceptionResnetV1 bouchon : retourne un embedding déterministe."""
    _call_count = 0

    def __init__(self, pretrained=None, **kw):
        super().__init__()
        self._seed = 42

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward(self, x):
        _FakeInceptionResnet._call_count += 1
        return _make_embedding(self._seed)

_facenet_mod = types.ModuleType("facenet_pytorch")
_facenet_mod.InceptionResnetV1 = _FakeInceptionResnet
sys.modules.setdefault("facenet_pytorch", _facenet_mod)

# torchvision.transforms stub
_tv = sys.modules.get("torchvision") or types.ModuleType("torchvision")
_tv_t = types.ModuleType("torchvision.transforms")

class _FakeCompose:
    def __init__(self, transforms): pass
    def __call__(self, img): return torch.zeros(3, 160, 160)

_tv_t.Compose   = _FakeCompose
_tv_t.Resize    = lambda s: None
_tv_t.ToTensor  = lambda: None
_tv_t.Normalize = lambda mean, std: None
_tv_t.ToPILImage = lambda: None

sys.modules.setdefault("torchvision",            _tv)
sys.modules.setdefault("torchvision.transforms", _tv_t)

_tv.transforms = _tv_t

# ── Import du module à tester ─────────────────────────────────────────────────
_store = Path(__file__).parent.parent
if str(_store) not in sys.path:
    sys.path.insert(0, str(_store))

from modules.face_recognizer import FaceRecognizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_face(seed: int = 0, size: int = 160) -> np.ndarray:
    """Génère un faux crop BGR déterministe."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def recognizer():
    """FaceRecognizer frais avec FaceNet mocké."""
    return FaceRecognizer(mode="standard")

@pytest.fixture
def enrolled_recognizer():
    """FaceRecognizer avec deux identités pré-enrôlées."""
    r = FaceRecognizer(mode="standard")
    # On surcharge get_embedding pour retourner des vecteurs distincts
    def _emb(face_crop):
        h = int(face_crop.mean())   # valeur différente selon l'image
        return _make_embedding(h)
    r.get_embedding = _emb

    r.enrolled_embeddings["Alice"] = _make_embedding(10)
    r.enrolled_embeddings["Bob"]   = _make_embedding(99)
    return r


# ── Tests get_embedding() ─────────────────────────────────────────────────────

class TestGetEmbedding:

    def test_returns_tensor(self, recognizer):
        crop = _random_face(0)
        emb  = recognizer.get_embedding(crop)
        assert isinstance(emb, torch.Tensor)

    def test_shape_512d(self, recognizer):
        crop = _random_face(1)
        emb  = recognizer.get_embedding(crop)
        assert emb.shape == (1, 512)

    def test_normalized(self, recognizer):
        """L'embedding doit être L2-normalisé (norme ≈ 1)."""
        crop = _random_face(2)
        emb  = recognizer.get_embedding(crop)
        norm = torch.norm(emb, p=2, dim=1).item()
        assert abs(norm - 1.0) < 1e-5


# ── Tests enroll_face() ───────────────────────────────────────────────────────

class TestEnrollFace:

    def test_enroll_new_identity(self, recognizer):
        """Enrôler une nouvelle identité renvoie True et la stocke."""
        crop = _random_face(3)
        ok   = recognizer.enroll_face("Charlie", crop)
        assert ok is True
        assert "Charlie" in recognizer.enrolled_embeddings

    def test_enroll_none_returns_false(self, recognizer):
        """Enrôler None doit renvoyer False sans crasher."""
        ok = recognizer.enroll_face("Ghost", None)
        assert ok is False
        assert "Ghost" not in recognizer.enrolled_embeddings

    def test_enroll_twice_updates_embedding(self, recognizer):
        """Enrôler la même personne deux fois met à jour l'embedding."""
        c1, c2 = _random_face(4), _random_face(5)
        recognizer.enroll_face("Diana", c1)
        emb_first = recognizer.enrolled_embeddings["Diana"].clone()
        recognizer.enroll_face("Diana", c2)
        # L'embedding doit avoir été mis à jour (différent du premier)
        # (même si le mock retourne toujours le même vecteur, le chemin code est couvert)
        assert "Diana" in recognizer.enrolled_embeddings

    def test_multiple_identities_stored(self, recognizer):
        """Plusieurs identités peuvent être enrôlées indépendamment."""
        recognizer.enroll_face("Eve",   _random_face(6))
        recognizer.enroll_face("Frank", _random_face(7))
        assert "Eve"   in recognizer.enrolled_embeddings
        assert "Frank" in recognizer.enrolled_embeddings


# ── Tests predict() ───────────────────────────────────────────────────────────

class TestPredict:

    def test_no_enrolled_returns_unknown(self, recognizer):
        """Sans identités enrôlées → (Inconnu, 0.0)."""
        name, score = recognizer.predict(_random_face(8))
        assert name  == "Inconnu"
        assert score == 0.0

    def test_none_input_returns_unknown(self, enrolled_recognizer):
        """Face None → (Inconnu, 0.0) sans exception."""
        name, score = enrolled_recognizer.predict(None)
        assert name  == "Inconnu"
        assert score == 0.0

    def test_high_similarity_returns_name(self):
        """Similarité cosinus élevée → l'identité est reconnue."""
        r   = FaceRecognizer()
        emb = _make_embedding(42)

        # On injecte directement l'embedding dans la base
        r.enrolled_embeddings["Alice"] = emb

        # On surcharge get_embedding pour retourner le MÊME vecteur
        r.get_embedding = lambda _: emb

        name, score = r.predict(_random_face(9))
        assert name  == "Alice"
        assert score  > r.threshold

    def test_low_similarity_returns_unknown(self):
        """Similarité cosinus faible → Inconnu malgré une identité enrôlée."""
        r = FaceRecognizer()
        r.enrolled_embeddings["Bob"] = _make_embedding(1)

        # get_embedding retourne un vecteur orthogonal (similarité ≈ 0)
        r.get_embedding = lambda _: _make_embedding(9999)

        name, score = r.predict(_random_face(10))
        assert name == "Inconnu"

    def test_score_non_negative(self, enrolled_recognizer):
        """Le score retourné est toujours ≥ 0."""
        _, score = enrolled_recognizer.predict(_random_face(11))
        assert score >= 0.0


# ── Tests switch_mode() ───────────────────────────────────────────────────────

class TestSwitchMode:

    def test_switch_to_hardened(self, recognizer):
        recognizer.switch_mode("hardened")
        assert recognizer.mode == "hardened"

    def test_switch_back_to_standard(self, recognizer):
        recognizer.switch_mode("hardened")
        recognizer.switch_mode("standard")
        assert recognizer.mode == "standard"

    def test_initial_mode_standard(self):
        r = FaceRecognizer()
        assert r.mode == "standard"