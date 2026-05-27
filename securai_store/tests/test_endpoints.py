"""
tests/test_endpoints.py
────────────────────────────────────────────────────────────────────────────────
Tests d'intégration Flask (test client) pour les endpoints :
  • /api/enroll
  • /api/toggle_patch_live   (T1)
  • /api/status
  • /api/toggle_attack

Stratégie : on injecte des faux modules directement dans sys.modules AVANT
tout import de app.py — pas de context manager au niveau module (incompatible
Python 3.11 + StopIteration).
"""

import io
import sys
import types
import threading as _threading
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Stubs injectés dans sys.modules AVANT import de app.py
# ─────────────────────────────────────────────────────────────────────────────

# --- cv2 ---------------------------------------------------------------
_cv2 = types.ModuleType("cv2")
_cv2.COLOR_BGR2RGB        = 4
_cv2.FONT_HERSHEY_SIMPLEX = 0
_cv2.IMREAD_COLOR         = 1
_cv2.IMWRITE_JPEG_QUALITY = 95
_cv2.CAP_DSHOW            = 700
_cv2.CAP_PROP_FRAME_WIDTH = 3
_cv2.CAP_PROP_FRAME_HEIGHT= 4
_cv2.imencode  = lambda ext, img, *a: (True, MagicMock(tobytes=lambda: b"\xff\xd8\xff"))
_cv2.imdecode  = lambda buf, flag: np.zeros((160, 160, 3), dtype=np.uint8)
_cv2.imread    = lambda path, flag=1: np.zeros((160, 160, 3), dtype=np.uint8)
_cv2.VideoCapture = MagicMock(return_value=MagicMock(
    isOpened=lambda: False,
    set=lambda *a: None,
    read=lambda: (False, None),
    release=lambda: None,
))
_cv2.cvtColor  = lambda img, code: img
_cv2.resize    = lambda img, sz, **kw: np.zeros((*reversed(sz), 3), dtype=np.uint8)
_cv2.rectangle = lambda *a, **kw: None
_cv2.putText   = lambda *a, **kw: None
sys.modules["cv2"] = _cv2

# --- requests ----------------------------------------------------------
_req       = types.ModuleType("requests")
_fake_resp = MagicMock(ok=False, json=lambda: {})
_req.post  = MagicMock(return_value=_fake_resp)

_fake_sess                = MagicMock()
_fake_sess.headers        = MagicMock()
_fake_sess.headers.update = MagicMock()
_fake_sess.post           = MagicMock(return_value=_fake_resp)
_req.Session              = MagicMock(return_value=_fake_sess)

_req_exc         = types.ModuleType("requests.exceptions")
class _Timeout(Exception): pass
_req_exc.Timeout = _Timeout
_req.exceptions  = _req_exc

sys.modules["requests"]            = _req
sys.modules["requests.exceptions"] = _req_exc

# --- modules.face_detector (stub complet — évite YOLO + pathlib) -------
class _FakeFaceDetector:
    def __init__(self, **kw):       pass
    def detect(self, frame):        return []
    def crop_face(self, f, b, size=160):
        return np.zeros((size, size, 3), dtype=np.uint8)

_fd_mod = types.ModuleType("modules.face_detector")
_fd_mod.FaceDetector = _FakeFaceDetector
sys.modules["modules.face_detector"] = _fd_mod

# --- modules.face_recognizer (stub complet — évite FaceNet / torch.nn) -
class _FakeFaceRecognizer:
    def __init__(self, mode="standard", **kw):
        self.mode                 = mode
        self.model                = MagicMock()
        self.threshold            = 0.65
        self.enrolled_embeddings  = {}          # dict réel utilisé par les tests

    def get_embedding(self, crop):
        return F.normalize(torch.randn(1, 512), p=2, dim=1)

    def enroll_face(self, name, crop):
        if crop is None:
            return False
        self.enrolled_embeddings[name] = self.get_embedding(crop)
        return True

    def predict(self, crop):
        return ("Inconnu", 0.0)

    def switch_mode(self, mode):
        self.mode = mode

_fr_mod = types.ModuleType("modules.face_recognizer")
_fr_mod.FaceRecognizer = _FakeFaceRecognizer
sys.modules["modules.face_recognizer"] = _fr_mod

# --- modules.patch_attacker (stub) -------------------------------------
class _FakePatchAttacker:
    def __init__(self, *a, **kw): pass
    def attack(self, crop, target_emb):
        return np.zeros_like(crop)

_pa_mod = types.ModuleType("modules.patch_attacker")
_pa_mod.PatchAttacker = _FakePatchAttacker
sys.modules["modules.patch_attacker"] = _pa_mod

# --- modules.fgsm_attacker (stub) --------------------------------------
class _FakeFGSMAttacker:
    def __init__(self, *a, **kw): pass
    def attack(self, *a, **kw):   return None

_fa_mod = types.ModuleType("modules.fgsm_attacker")
_fa_mod.FGSMAttacker = _FakeFGSMAttacker
sys.modules["modules.fgsm_attacker"] = _fa_mod

# --- modules.defender / anomaly_detector (stubs) ----------------------
for _name, _cls_name in [("modules.defender",       "Defender"),
                          ("modules.anomaly_detector","AnomalyDetector")]:
    _m = types.ModuleType(_name)
    _fake_cls = type(_cls_name, (), {
        "__init__":    lambda self, *a, **kw: None,
        "apply_defense": lambda self, img, **kw: img,
        "analyze":       lambda self, img: (False, 0.0),
    })
    setattr(_m, _cls_name, _fake_cls)
    sys.modules[_name] = _m

# --- modules.rights_manager (stub) ------------------------------------
class _FakeRightsManager:
    EMPLOYEE = "Employee"
    MANAGER  = "Manager"
    def __init__(self):            self._ids = {}
    def add_identity(self, n, l):  self._ids[n] = l
    def get_access_level(self, n): return "DENIED"
    def get_permissions(self, n):  return {}
    def get_ui_config(self, n):    return {"color": "#ffffff"}

_rm_mod = types.ModuleType("modules.rights_manager")
_rm_mod.RightsManager = _FakeRightsManager
sys.modules["modules.rights_manager"] = _rm_mod

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bloquer le thread vidéo PUIS importer app.py
#     On patch threading.Thread directement (pas de context manager).
# ─────────────────────────────────────────────────────────────────────────────
_OrigThread = _threading.Thread
_threading.Thread = lambda *a, **kw: MagicMock(start=lambda: None, daemon=True)

_store = Path(__file__).parent.parent
if str(_store) not in sys.path:
    sys.path.insert(0, str(_store))

import app as _app                 # noqa: E402 — import intentionnellement tardif

_threading.Thread = _OrigThread    # restore

flask_app       = _app.app
face_recognizer = _app.face_recognizer
state           = _app.state
state_lock      = _app.state_lock
_patch_cache    = _app._patch_cache


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Remet le SystemState à zéro avant chaque test."""
    with state_lock:
        state.attack_active    = False
        state.patch_active     = False
        state.patch_target     = "Manager_Demo"
        state.identity         = "Aucun"
        state.confidence       = 0.0
        state.anomaly_detected = False
        state.anomaly_score    = 0.0
        state.fps              = 0
        state.model_mode       = "standard"
    face_recognizer.enrolled_embeddings.clear()
    _patch_cache['crop'] = None
    yield


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


def _fake_image_bytes() -> bytes:
    """PNG 1×1 blanc minimal valide."""
    return (
        b'\x89PNG\r\n\x1a\n'
        b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00'
        b'\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18'
        b'\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tests /api/status
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus:

    def test_200_ok(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200

    def test_returns_json(self, client):
        assert client.get("/api/status").get_json() is not None

    def test_required_fields(self, client):
        required = {
            "identity", "access_level", "permissions",
            "anomaly_detected", "anomaly_score",
            "attack_active", "patch_active", "patch_target",
            "model_mode", "fps", "confidence",
        }
        data    = client.get("/api/status").get_json()
        missing = required - data.keys()
        assert not missing, f"Champs manquants dans /api/status : {missing}"

    def test_patch_fields_default_values(self, client):
        data = client.get("/api/status").get_json()
        assert data["patch_active"] is False
        assert data["patch_target"] == "Manager_Demo"

    def test_attack_active_default_false(self, client):
        assert client.get("/api/status").get_json()["attack_active"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. Tests /api/toggle_attack
# ─────────────────────────────────────────────────────────────────────────────

class TestToggleAttack:

    def test_activate(self, client):
        r = client.post("/api/toggle_attack", json={"active": True})
        assert r.status_code == 200
        assert r.get_json()["success"] is True
        assert state.attack_active is True

    def test_deactivate(self, client):
        with state_lock:
            state.attack_active = True
        r = client.post("/api/toggle_attack", json={"active": False})
        assert r.status_code == 200
        assert state.attack_active is False

    def test_missing_active_field(self, client):
        r = client.post("/api/toggle_attack", json={})
        assert r.status_code == 400
        assert r.get_json()["success"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tests /api/toggle_patch_live  (T1)
# ─────────────────────────────────────────────────────────────────────────────

class TestTogglePatchLive:

    def _enroll_target(self, name: str = "Manager_Demo"):
        """Injecte un embedding directement dans face_recognizer."""
        face_recognizer.enrolled_embeddings[name] = F.normalize(
            torch.randn(1, 512), p=2, dim=1
        )

    def test_activate_enrolled_target(self, client):
        self._enroll_target("Manager_Demo")
        r    = client.post("/api/toggle_patch_live",
                           json={"active": True, "target": "Manager_Demo"})
        body = r.get_json()
        assert r.status_code       == 200
        assert body["success"]      is True
        assert body["patch_active"] is True
        assert body["patch_target"] == "Manager_Demo"

    def test_state_updated_after_activate(self, client):
        self._enroll_target("Manager_Demo")
        client.post("/api/toggle_patch_live",
                    json={"active": True, "target": "Manager_Demo"})
        assert state.patch_active is True
        assert state.patch_target == "Manager_Demo"

    def test_deactivate_clears_cache(self, client):
        _patch_cache['crop'] = np.zeros((160, 160, 3), dtype=np.uint8)
        r = client.post("/api/toggle_patch_live", json={"active": False})
        assert r.status_code        == 200
        assert state.patch_active   is False
        assert _patch_cache['crop'] is None

    def test_missing_active_field_400(self, client):
        r = client.post("/api/toggle_patch_live", json={})
        assert r.status_code        == 400
        assert r.get_json()["success"] is False

    def test_unknown_target_404(self, client):
        r    = client.post("/api/toggle_patch_live",
                           json={"active": True, "target": "Fantôme"})
        body = r.get_json()
        assert r.status_code    == 404
        assert body["success"]  is False
        assert "enrolled" in body

    def test_unknown_target_leaves_state_unchanged(self, client):
        client.post("/api/toggle_patch_live",
                    json={"active": True, "target": "Fantôme"})
        assert state.patch_active is False

    def test_default_target_used_when_omitted(self, client):
        self._enroll_target("Manager_Demo")
        r = client.post("/api/toggle_patch_live", json={"active": True})
        assert r.status_code == 200
        assert r.get_json()["patch_target"] == "Manager_Demo"

    def test_status_reflects_patch_active(self, client):
        self._enroll_target("Manager_Demo")
        client.post("/api/toggle_patch_live",
                    json={"active": True, "target": "Manager_Demo"})
        assert client.get("/api/status").get_json()["patch_active"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 7. Tests /api/enroll
# ─────────────────────────────────────────────────────────────────────────────

class TestEnroll:

    def test_missing_image_400(self, client):
        r = client.post("/api/enroll", data={"name": "Alice"})
        assert r.status_code == 400

    def test_missing_name_400(self, client):
        r = client.post("/api/enroll", data={
            "image": (io.BytesIO(_fake_image_bytes()), "face.png"),
        })
        assert r.status_code == 400

    def test_empty_filename_400(self, client):
        r = client.post("/api/enroll", data={
            "image": (io.BytesIO(b""), ""),
            "name":  "Alice",
        })
        assert r.status_code == 400

    def test_valid_enroll_200(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(_app, "ENROLLED_DIR", str(tmp_path))
        _app.rights_manager.add_identity = MagicMock()
        r = client.post("/api/enroll", data={
            "image": (io.BytesIO(_fake_image_bytes()), "face.png"),
            "name":  "Alice",
            "level": "Employee",
        })
        assert r.status_code          == 200
        assert r.get_json()["success"] is True

    def test_enroll_calls_add_identity(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(_app, "ENROLLED_DIR", str(tmp_path))
        mock_add = MagicMock()
        monkeypatch.setattr(_app.rights_manager, "add_identity", mock_add)
        client.post("/api/enroll", data={
            "image": (io.BytesIO(_fake_image_bytes()), "face.png"),
            "name":  "Bob",
            "level": "Manager",
        })
        mock_add.assert_called_once_with("Bob", "Manager")