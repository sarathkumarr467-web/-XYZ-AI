from fastapi import APIRouter, HTTPException

from .ml_predict import predict_student


router = APIRouter(
    prefix="/ml",
    tags=["Machine Learning"]
)


@router.get("/status")
def ml_status():

    return {
        "status": "online",
        "service": "Student ML Prediction",
        "message": "ML API is ready"
    }


@router.post("/predict")
def predict(data: dict):

    try:

        result = predict_student(
            data
        )

        return {
            "success": True,
            "message": "Prediction generated successfully",
            "result": result
        }

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )