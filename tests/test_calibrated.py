import math

from src.inference.classifier import CalibratedClassifier
from src.inference.feature_extraction import deviation_angle_deg

REF = {"cervical": [0, 0, 1], "dorsal": [0, 0, 1], "lumbar": [0, 0, 1]}
SQRT3_2 = math.sqrt(3) / 2  # cos(30°)


def test_deviation_cero_cuando_igual():
    assert deviation_angle_deg([0, 0, 1], [0, 0, 1]) == 0.0


def test_deviation_90_ortogonal():
    assert round(deviation_angle_deg([1, 0, 0], [0, 0, 1])) == 90


def test_deviation_30_grados():
    assert round(deviation_angle_deg([0.5, 0, SQRT3_2], [0, 0, 1])) == 30


def test_adecuada_cerca_de_la_neutra():
    current = {"cervical": [0.05, 0, 0.998], "dorsal": [0, 0.05, 0.998], "lumbar": [0, 0, 1]}
    cls, conf = CalibratedClassifier().predict(current, REF)
    assert cls == "adequate"
    assert conf > 0.7


def test_forward_slouch_si_el_dorsal_se_inclina_adelante():
    current = {"cervical": [0, 0, 1], "dorsal": [0.5, 0, SQRT3_2], "lumbar": [0, 0, 1]}
    cls, _ = CalibratedClassifier().predict(current, REF)
    assert cls == "forward_slouch"


def test_excessive_recline_si_el_dorsal_se_inclina_atras():
    current = {"cervical": [0, 0, 1], "dorsal": [-0.5, 0, SQRT3_2], "lumbar": [0, 0, 1]}
    cls, _ = CalibratedClassifier().predict(current, REF)
    assert cls == "excessive_recline"


def test_detecta_desviacion_en_cualquier_zona():
    # solo el cervical se desvía -> ya no es adecuada (usa los 3 sensores)
    current = {"cervical": [0.5, 0, SQRT3_2], "dorsal": [0, 0, 1], "lumbar": [0, 0, 1]}
    cls, _ = CalibratedClassifier().predict(current, REF)
    assert cls != "adequate"
