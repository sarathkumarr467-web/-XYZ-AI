import os
import json
import joblib
import pandas as pd


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(
    BACKEND_DIR,
    "models",
    "student_model.pkl"
)

METADATA_PATH = os.path.join(
    BACKEND_DIR,
    "models",
    "model_metadata.json"
)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ML model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def predict_student(data: dict):

    model = load_model()
    metadata = load_metadata()

    features = metadata.get("features", [])

    if not features:
        raise ValueError(
            "Model features are not available."
        )

    input_data = pd.DataFrame([data])

    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    result = {
        "prediction": (
            prediction.item()
            if hasattr(prediction, "item")
            else prediction
        )
    }

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        confidence = max(probabilities) * 100

        result["confidence"] = round(
            float(confidence),
            2
        )

    return result