import joblib
from pathlib import Path

from .feature_extraction import deviation_angle_deg, signed_sagittal_tilt_deg

Triple = list[float]
Reference = dict[str, Triple]  # {"cervical": [...], "dorsal": [...], "lumbar": [...]}
ZONES = ("cervical", "dorsal", "lumbar")


class PostureClassifier:
    """Modelo rf_v1: Random Forest sobre el sensor dorsal. Sin calibración."""

    MODEL_VERSION = "rf_v1"
    CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, model_path: Path) -> None:
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._label_map: dict[int, str] = bundle["label_map"]

    def predict(self, dorsal_features: Triple) -> tuple[str, float]:
        proba = self._model.predict_proba([dorsal_features])[0]
        class_idx = int(proba.argmax())
        confidence = float(proba[class_idx])
        if confidence < self.CONFIDENCE_THRESHOLD:
            return "indeterminate", confidence
        return self._label_map[class_idx], confidence


class CalibratedClassifier:
    """Clasificador por desviación respecto de la calibración del usuario.

    Usa los tres sensores: mide cuánto se desvió cada zona (cervical, dorsal,
    lumbar) de la postura neutra registrada en la calibración. Si la mayor
    desviación es pequeña, la postura es adecuada; si no, distingue inclinación
    hacia adelante de reclinación por la inclinación sagital del dorsal.
    """

    MODEL_VERSION = "calibrated_v2"
    # Desviación (grados) desde la neutra por debajo de la cual la postura se
    # considera adecuada (referencia clínica ~20°, con margen de tolerancia).
    ADEQUATE_MAX_DEV_DEG = 12.0
    # Desviación a partir de la cual damos confianza plena.
    FULL_CONFIDENCE_DEV_DEG = 25.0

    def predict(self, current: Reference, reference: Reference) -> tuple[str, float]:
        deviations = {
            zone: deviation_angle_deg(current[zone], reference[zone]) for zone in ZONES
        }
        max_dev = max(deviations.values())

        if max_dev < self.ADEQUATE_MAX_DEV_DEG:
            confidence = 1.0 - 0.3 * (max_dev / self.ADEQUATE_MAX_DEV_DEG)
            return "adequate", round(min(1.0, confidence), 3)

        tilt = signed_sagittal_tilt_deg(current["dorsal"], reference["dorsal"])
        posture = "forward_slouch" if tilt >= 0 else "excessive_recline"
        span = self.FULL_CONFIDENCE_DEV_DEG - self.ADEQUATE_MAX_DEV_DEG
        confidence = 0.6 + 0.4 * min(1.0, (max_dev - self.ADEQUATE_MAX_DEV_DEG) / span)
        return posture, round(confidence, 3)
