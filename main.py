"""Serve the trained root-cause classifier through a small FastAPI application.

Run this file from the project virtual environment, or use Uvicorn directly:
    .\\.venv\\Scripts\\python.exe main.py
    .\\.venv\\Scripts\\python.exe -m uvicorn main:app --reload
"""

# Core Python libraries
import json                                # For working with JSON data
from contextlib import asynccontextmanager # For managing async context (e.g., DB connections, lifecycle events)
from pathlib import Path                   # For handling filesystem paths
from typing import Annotated               # For type hints with metadata

# ML and data handling
import joblib                              # For saving/loading ML models
import pandas as pd                        # For working with tabular datasets

# FastAPI and server
import uvicorn                             # ASGI server to run FastAPI apps
from fastapi import FastAPI, HTTPException, Request  # FastAPI framework + error handling + request object

# Data validation
from pydantic import BaseModel, ConfigDict, Field    # For defining request/response schemas with validation


# Resolve artifact paths relative to this file rather than the shell's current
# directory. This lets the API find its model when launched from another folder.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "root_cause_xgboost_model.pkl"
METADATA_PATH = BASE_DIR / "root_cause_model_metadata.json"
FEATURES_PATH = BASE_DIR / "feature_columns.json"


class PredictionInput(BaseModel):
    """One interaction with the same predictor fields used during training.

    FastAPI uses these types and constraints to validate incoming JSON and to
    document the request schema in /docs. Extra fields are rejected so that
    misspelled or unsupported inputs do not silently pass unnoticed.
    """

    model_config = ConfigDict(extra="forbid")

    # The skill name is accepted by the API, then converted to the exact
    # integer code used in training via skill_mapping in the metadata file.
    skill_id: str

    # Outcome and timing measurements for this interaction. `correct` is
    # binary; response duration must be positive.
    correct: Annotated[int, Field(ge=0, le=1)]
    elapsed_time_ms: Annotated[float, Field(gt=0)]

    # Item difficulty and the student's prior attempts/interaction position.
    # Accuracy features are proportions, while accuracy_decay is a signed
    # difference and is therefore allowed to be negative.
    question_difficulty: Annotated[float, Field(ge=0, le=1)]
    question_attempt_number: Annotated[int, Field(ge=1)]
    interaction_sequence: Annotated[int, Field(ge=1)]
    recent_accuracy: Annotated[float, Field(ge=0, le=1)]
    historical_accuracy: Annotated[float, Field(ge=0, le=1)]
    accuracy_decay: float

    # Response-time comparisons and error-rate indicators used as signals for
    # changes in performance. Error rates are proportions between zero and one.
    recent_response_time: Annotated[float, Field(ge=0)]
    response_time_change: Annotated[float, Field(ge=0)]
    same_skill_error_rate: Annotated[float, Field(ge=0, le=1)]
    cross_skill_error_rate: Annotated[float, Field(ge=0, le=1)]
    prerequisite_failure_rate: Annotated[float, Field(ge=0, le=1)]

    # Session context. session_length counts interactions; the two elapsed
    # time fields are in seconds, matching the training data.
    session_length: Annotated[int, Field(ge=1)]
    time_since_last_interaction: Annotated[float, Field(ge=0)]
    test_elapsed_time: Annotated[float, Field(ge=0)]

    # Calendar features derived from the interaction timestamp. Weekday uses
    # Python's convention: Monday=0 through Sunday=6.
    interaction_hour: Annotated[int, Field(ge=0, le=23)]
    interaction_day_of_week: Annotated[int, Field(ge=0, le=6)]
    interaction_month: Annotated[int, Field(ge=1, le=12)]
    is_weekend: Annotated[int, Field(ge=0, le=1)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and check shared model artifacts once when the API starts.

    Loading during application startup avoids re-reading a multi-megabyte
    pickle for every request. Fail fast when artifacts are absent or their
    feature/class definitions do not match, rather than serving bad results.
    """
    missing_files = [
        str(path)
        for path in (MODEL_PATH, METADATA_PATH, FEATURES_PATH)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Required model artifact files are missing: "
            + ", ".join(missing_files)
        )

    # Metadata contains human-readable class names and the trained skill-code
    # mapping; the feature file preserves the exact training feature order.
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    with FEATURES_PATH.open(encoding="utf-8") as features_file:
        feature_columns = json.load(features_file)

    # Pickle deserialization must only be used with a trusted model file.
    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != feature_columns:
        raise ValueError(
            "Model feature order does not match feature_columns.json."
        )

    # XGBoost returns numeric class IDs. The metadata must provide one display
    # name for each ID so the API can return meaningful labels to callers.
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or len(class_names) != len(model.classes_):
        raise ValueError(
            "Model classes do not match class_names in model metadata."
        )

    # Keep loaded objects on the app instance so endpoint handlers can reuse
    # them without module-level mutable globals.
    app.state.model = model
    app.state.metadata = metadata
    app.state.feature_columns = feature_columns
    yield


app = FastAPI(
    title="Root Cause Diagnostic API",
    description="Predict a learning-interaction root cause with the trained XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Report whether the model is available to handle prediction requests."""
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {"status": "ok", "model_type": request.app.state.metadata["model_type"]}


@app.post("/predict")
def predict(payload: PredictionInput, request: Request) -> dict[str, object]:
    """Classify one interaction and return its label and per-class scores.

    The request uses the original string skill ID. Before inference, it is
    transformed to the integer encoding used at training time and assembled
    in the persisted feature order expected by XGBoost.
    """
    if not hasattr(request.app.state, "model"):
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    metadata = request.app.state.metadata
    skill_mapping = metadata["skill_mapping"]
    # Never assign an arbitrary/new skill a code: the model was not trained
    # with that mapping, so accepting it could produce misleading predictions.
    if payload.skill_id not in skill_mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown skill_id '{payload.skill_id}'.",
        )

    values = payload.model_dump()
    values["skill_id"] = skill_mapping[payload.skill_id]
    # Reindex by the saved training feature list to guarantee the model sees
    # the same columns in the same order as it did during fitting.
    values = {
        column: values[column]
        for column in request.app.state.feature_columns
    }
    features = pd.DataFrame([values], columns=request.app.state.feature_columns)

    model = request.app.state.model
    # `predict` selects the winning class; `predict_proba` supplies a score for
    # every class so clients can assess how decisive that prediction is.
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    class_names = metadata["class_names"]
    # Pair each model class ID with the corresponding saved human-readable
    # label. The probability values are converted to native floats for JSON.
    class_probabilities = {
        class_names[int(class_id)]: float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }

    return {
        "root_cause": class_names[predicted_class],
        "confidence": class_probabilities[class_names[predicted_class]],
        "probabilities": class_probabilities,
    }


if __name__ == "__main__":
    # Support `python main.py` as well as the usual `uvicorn main:app` command.
    uvicorn.run(app, host="127.0.0.1", port=8000)
