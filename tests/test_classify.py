"""
Tests de criterios de aceptación:
  HU-04 Happy : POST /ml/classify → devuelve clase + confianza + model_version
  HU-04 Unhappy: campos faltantes → 422
  HU-04 Unhappy: modelo no cargado → 503
  HU-05 Happy : sensor dorsal inclinado adelante → forward_slouch
  HU-05 Happy : sensor dorsal inclinado atrás → excessive_recline
  HU-05 Unhappy: datos ambiguos → indeterminate (confianza < 0.5)
"""
import pytest
from fastapi.testclient import TestClient

from src.api.routes import get_classifier, set_classifier
from src.main import app


class _MockClassifier:
    """Clasifica basándose en el eje Z del sensor dorsal (features[2])."""
    MODEL_VERSION = "rf_v1"

    def predict(self, features: list[float]) -> tuple[str, float]:
        az = features[2]
        if az > 8.0:   return ("adequate", 0.95)
        if az > 2.0:   return ("forward_slouch", 0.88)
        if az < -2.0:  return ("excessive_recline", 0.80)
        return ("indeterminate", 0.30)  # confianza < threshold → indeterminate


VALID_BODY = {
    "cervical": [0.1, 0.0, 0.3],
    "dorsal":   [0.0, 0.1, 9.8],   # az > 8 → adequate
    "lumbar":   [0.0, 0.1, 0.3],
}


@pytest.fixture(scope="module")
def client():
    """TestClient sin context manager para evitar el lifespan (carga de .pkl)."""
    set_classifier(_MockClassifier())
    return TestClient(app)


# ── HU-04 ─────────────────────────────────────────────────────────────────────

# Happy: clasificación retorna estructura correcta
def test_hu04_classify_retorna_estructura_completa(client):
    response = client.post("/ml/classify", json=VALID_BODY)
    assert response.status_code == 200
    body = response.json()
    assert "class" in body
    assert "confidence" in body
    assert "model_version" in body
    assert 0.0 <= body["confidence"] <= 1.0


# Unhappy: falta campo dorsal → 422
def test_hu04_campo_faltante_retorna_422(client):
    body = {"cervical": [0.1, 0.0, 0.3], "lumbar": [0.0, 0.1, 0.3]}
    response = client.post("/ml/classify", json=body)
    assert response.status_code == 422


# Unhappy: modelo no cargado → 503
def test_hu04_modelo_no_cargado_retorna_503():
    from src.api import routes
    original = routes._classifier
    routes._classifier = None
    try:
        c = TestClient(app)
        response = c.post("/ml/classify", json=VALID_BODY)
        assert response.status_code == 503
    finally:
        routes._classifier = original


# ── HU-05 ─────────────────────────────────────────────────────────────────────

# Happy: sensor dorsal inclinado hacia adelante → forward_slouch
def test_hu05_identifica_encorvamiento_frontal(client):
    body = {**VALID_BODY, "dorsal": [0.0, 0.1, 3.0]}  # az 2-8 → forward_slouch
    response = client.post("/ml/classify", json=body)
    assert response.status_code == 200
    assert response.json()["class"] == "forward_slouch"


# Happy: sensor dorsal inclinado hacia atrás → excessive_recline
def test_hu05_identifica_reclinacion_excesiva(client):
    body = {**VALID_BODY, "dorsal": [0.0, 0.1, -3.0]}  # az < -2 → excessive_recline
    response = client.post("/ml/classify", json=body)
    assert response.status_code == 200
    assert response.json()["class"] == "excessive_recline"


# Unhappy: datos ambiguos → indeterminate (confianza < 0.5 en el mock)
def test_hu05_datos_ambiguos_devuelven_indeterminate(client):
    body = {**VALID_BODY, "dorsal": [0.0, 0.0, 0.5]}  # az ≈ 0 → indeterminate (conf 0.30)
    response = client.post("/ml/classify", json=body)
    assert response.status_code == 200
    assert response.json()["class"] == "indeterminate"


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_retorna_ok(client):
    response = client.get("/ml/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
