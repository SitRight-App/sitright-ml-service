from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..inference.classifier import PostureClassifier
from ..inference.feature_extraction import extract_dorsal_features

router = APIRouter(prefix="/ml", tags=["ml"])

_classifier: PostureClassifier | None = None


def set_classifier(classifier: PostureClassifier) -> None:
    global _classifier
    _classifier = classifier


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
    features = extract_dorsal_features(request.dorsal)
    posture_class, confidence = classifier.predict(features)
    return ClassifyResponse(
        class_=posture_class,
        confidence=confidence,
        model_version=PostureClassifier.MODEL_VERSION,
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
