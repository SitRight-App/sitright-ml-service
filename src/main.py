import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .api.routes import router, set_calibrated_classifier, set_classifier
from .inference.classifier import CalibratedClassifier, PostureClassifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", "./src/models/rf_v1.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run ml-training/notebooks/04-export.ipynb to generate it."
        )
    set_classifier(PostureClassifier(model_path))
    set_calibrated_classifier(CalibratedClassifier())
    yield


app = FastAPI(
    title="SitRight ML Service",
    description="Clasificación postural: adequate | forward_slouch | excessive_recline",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
