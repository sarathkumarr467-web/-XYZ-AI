from pydantic import BaseModel


# =========================================================
# LOGIN
# =========================================================

class LoginRequest(BaseModel):
    username: str
    role: str


class UserResponse(BaseModel):
    id: int
    username: str
    name: str
    role: str

    class Config:
        from_attributes = True


# =========================================================
# STUDENT
# =========================================================

class StudentCreate(BaseModel):
    name: str
    class_name: str
    section: str
    attendance_percentage: float = 0.0


class StudentUpdate(BaseModel):
    name: str
    class_name: str
    section: str
    attendance_percentage: float = 0.0


# =========================================================
# TEACHER
# =========================================================

class TeacherCreate(BaseModel):
    name: str
    username: str


class TeacherUpdate(BaseModel):
    name: str
    username: str


# =========================================================
# ATTENDANCE
# =========================================================

class AttendanceCreate(BaseModel):
    student_id: int
    date: str
    status: str