from fastapi import APIRouter
from pydantic import BaseModel
from .ml_service import predict_student_risk

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


class PredictionRequest(BaseModel):

    features: list[float]


@router.post("/predict")
def predict(request: PredictionRequest):

    result = predict_student_risk(request.features)

    return result