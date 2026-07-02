import joblib
from pathlib import Path


class PostureClassifier:
    MODEL_VERSION = "rf_v1"
    CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, model_path: Path) -> None:
        bundle = joblib.load(model_path)
        self._model = bundle["model"]
        self._label_map: dict[int, str] = bundle["label_map"]

    def predict(self, dorsal_features: list[float]) -> tuple[str, float]:
        proba = self._model.predict_proba([dorsal_features])[0]
        class_idx = int(proba.argmax())
        confidence = float(proba[class_idx])
        if confidence < self.CONFIDENCE_THRESHOLD:
            return "indeterminate", confidence
        return self._label_map[class_idx], confidence
