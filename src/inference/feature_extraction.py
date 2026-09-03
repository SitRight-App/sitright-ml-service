import math

Triple = list[float]


def extract_dorsal_features(dorsal: Triple) -> Triple:
    """Devuelve [ax, ay, az] del sensor dorsal — features del modelo rf_v1."""
    return list(dorsal)


def _norm(v: Triple) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def deviation_angle_deg(current: Triple, reference: Triple) -> float:
    """Ángulo (grados) entre el vector de gravedad actual y el de la calibración.

    Mide cuánto se inclinó el sensor respecto de su postura neutra. Al comparar
    contra la referencia del propio usuario, es invariante a cómo esté montado
    el sensor y a la anatomía de cada persona.
    """
    na, nb = _norm(current), _norm(reference)
    if na == 0 or nb == 0:
        return 0.0
    dot = sum(c * r for c, r in zip(current, reference))
    cos = max(-1.0, min(1.0, dot / (na * nb)))
    return math.degrees(math.acos(cos))


# Ejes del sensor usados para el pitch (inclinación sagital). Dependen del
# montaje físico del chaleco; si el signo sale invertido en la prueba real,
# ajustar FORWARD_SIGN.
SAGITTAL_AXIS = 0  # eje adelante/atrás
VERTICAL_AXIS = 2  # eje de la gravedad en neutra
FORWARD_SIGN = 1.0


def signed_sagittal_tilt_deg(current: Triple, reference: Triple) -> float:
    """Inclinación adelante(+) / atrás(-) del sensor respecto de la neutra."""

    def pitch(v: Triple) -> float:
        return math.degrees(math.atan2(v[SAGITTAL_AXIS], v[VERTICAL_AXIS]))

    return FORWARD_SIGN * (pitch(current) - pitch(reference))
