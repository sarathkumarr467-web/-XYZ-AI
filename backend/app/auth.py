from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .schemas import LoginRequest

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):

    user = db.query(User).filter(
        User.username == data.username,
        User.role == data.role
    ).first()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or role"
        )

    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "role": user.role
        }
    }