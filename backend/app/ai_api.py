from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .database import get_db
from .ai_assistant import assistant_response


router = APIRouter(
    prefix="/ai",
    tags=["XYZ AI Assistant"]
)


@router.get("/status")
def ai_status():

    return {
        "status": "online",
        "assistant": "XYZ AI",
        "version": "1.0",
        "message": "XYZ AI Assistant is ready"
    }


@router.post("/chat")
def chat(
    data: dict,
    db: Session = Depends(get_db)
):

    message = data.get("message", "")

    if not isinstance(message, str):

        return {
            "success": False,
            "message": "Message must be text."
        }

    message = message.strip()

    if not message:

        return {
            "success": False,
            "message": "Please provide a message."
        }

    try:

        result = assistant_response(
            message,
            db
        )

        return {
            "success": True,
            "user_message": message,
            "assistant": result
        }

    except Exception as error:

        return {
            "success": False,
            "message": f"AI processing failed: {error}"
        }