import os
import joblib
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "student_risk_model.pkl"
)

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


def predict_student_risk(features):

    if model is None:
        return {
            "success": False,
            "message": "ML model is not trained yet."
        }

    values = np.array(features).reshape(1, -1)

    prediction = model.predict(values)[0]

    return {
        "success": True,
        "prediction": float(prediction)
    }