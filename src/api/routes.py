from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..inference.classifier import CalibratedClassifier, PostureClassifier
from ..inference.feature_extraction import extract_dorsal_features

router = APIRouter(prefix="/ml", tags=["ml"])

_classifier: PostureClassifier | None = None
_calibrated: CalibratedClassifier | None = None

ZONES = ("cervical", "dorsal", "lumbar")


def set_classifier(classifier: PostureClassifier) -> None:
    global _classifier
    _classifier = classifier


def set_calibrated_classifier(classifier: CalibratedClassifier) -> None:
    global _calibrated
    _calibrated = classifier


def get_classifier() -> PostureClassifier:
    if _classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _classifier


class ClassifyRequest(BaseModel):
    cervical: list[float] = Field(..., min_length=3, max_length=3)
    dorsal: list[float] = Field(..., min_length=3, max_length=3)
    lumbar: list[float] = Field(..., min_length=3, max_length=3)
    reference: dict | None = None


class ClassifyResponse(BaseModel):
    class_: str = Field(..., alias="class")
    confidence: float
    model_version: str

    model_config = {"populate_by_name": True}


@router.post("/classify", response_model=ClassifyResponse, response_model_by_alias=True)
async def classify_posture(
    request: ClassifyRequest,
    classifier: Annotated[PostureClassifier, Depends(get_classifier)],
) -> ClassifyResponse:
    ref = request.reference
    if _calibrated is not None and ref and all(z in ref for z in ZONES):
        current = {"cervical": request.cervical, "dorsal": request.dorsal, "lumbar": request.lumbar}
        posture_class, confidence = _calibrated.predict(current, ref)
        version = CalibratedClassifier.MODEL_VERSION
    else:
        features = extract_dorsal_features(request.dorsal)
        posture_class, confidence = classifier.predict(features)
        version = PostureClassifier.MODEL_VERSION
    return ClassifyResponse(
        class_=posture_class,
        confidence=confidence,
        model_version=version,
    )


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/model-info")
async def model_info(
    classifier: Annotated[PostureClassifier, Depends(get_classifier)],
) -> dict:
    return {
        "version": PostureClassifier.MODEL_VERSION,
        "classes": list(classifier._label_map.values()),
        "features": ["dorsal_ax", "dorsal_ay", "dorsal_az"],
    }
