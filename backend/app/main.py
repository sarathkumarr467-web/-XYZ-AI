from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import Base, engine, get_db
from .models import User, Student, Attendance

from .schemas import (
    StudentCreate,
    StudentUpdate,
    TeacherCreate,
    TeacherUpdate,
    AttendanceCreate,
)

from .auth import router as auth_router

# =========================================================
# NEW: ML + AI ROUTERS
# =========================================================

from .ml_api import router as ml_router
from .ai_api import router as ai_router


# =========================================================
# DATABASE
# =========================================================

Base.metadata.create_all(bind=engine)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="XYZ AI School Assistant",
    description=(
        "School Management, AI Assistant "
        "and Student ML Prediction Backend"
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# AUTH ROUTES
# =========================================================

app.include_router(auth_router)


# =========================================================
# ML + AI ROUTES
# =========================================================

app.include_router(ml_router)
app.include_router(ai_router)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def root():

    return {
        "message": "XYZ AI School Assistant is running",
        "status": "success",
        "version": "1.0.0",
        "services": [
            "School Management",
            "Attendance",
            "Reports",
            "XYZ AI Assistant",
            "ML Prediction",
        ],
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Backend is working",
    }


# =========================================================
# DASHBOARD
# =========================================================

@app.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
):

    total_users = (
        db.query(User).count()
    )

    total_students = (
        db.query(Student).count()
    )

    total_teachers = (
        db.query(User)
        .filter(User.role == "teacher")
        .count()
    )

    total_attendance_records = (
        db.query(Attendance).count()
    )

    students = (
        db.query(Student).all()
    )

    if students:

        total_percentage = sum(
            student.attendance_percentage or 0
            for student in students
        )

        average_attendance = round(
            total_percentage / len(students),
            2,
        )

    else:

        average_attendance = 0

    return {
        "message": "Dashboard data",
        "total_users": total_users,
        "total_students": total_students,
        "total_teachers": total_teachers,
        "total_attendance_records": (
            total_attendance_records
        ),
        "average_attendance": (
            average_attendance
        ),
    }


# =========================================================
# USERS
# =========================================================

@app.get("/users")
def get_users(
    db: Session = Depends(get_db),
):

    users = (
        db.query(User)
        .all()
    )

    return [
        {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "role": user.role,
        }
        for user in users
    ]


# =========================================================
# STUDENTS - GET ALL
# =========================================================

@app.get("/students")
def get_students(
    db: Session = Depends(get_db),
):

    students = (
        db.query(Student)
        .all()
    )

    return [
        {
            "id": student.id,
            "name": student.name,
            "class_name": student.class_name,
            "section": student.section,
            "attendance_percentage": (
                student.attendance_percentage or 0
            ),
            "parent_id": student.parent_id,
            "teacher_id": student.teacher_id,
        }
        for student in students
    ]


# =========================================================
# STUDENTS - GET SINGLE
# =========================================================

@app.get("/students/{student_id}")
def get_student(
    student_id: int,
    db: Session = Depends(get_db),
):

    student = (
        db.query(Student)
        .filter(
            Student.id == student_id
        )
        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found",
        )

    return {
        "id": student.id,
        "name": student.name,
        "class_name": student.class_name,
        "section": student.section,
        "attendance_percentage": (
            student.attendance_percentage or 0
        ),
        "parent_id": student.parent_id,
        "teacher_id": student.teacher_id,
    }


# =========================================================
# CREATE STUDENT
# =========================================================

@app.post("/students")
def create_student(
    student_data: StudentCreate,
    db: Session = Depends(get_db),
):

    if not 0 <= student_data.attendance_percentage <= 100:

        raise HTTPException(
            status_code=400,
            detail=(
                "Attendance must be between 0 and 100"
            ),
        )

    student = Student(
        name=student_data.name,
        class_name=student_data.class_name,
        section=student_data.section,
        attendance_percentage=(
            student_data.attendance_percentage
        ),
    )

    db.add(student)
    db.commit()
    db.refresh(student)

    return {
        "message": "Student created successfully",
        "student": {
            "id": student.id,
            "name": student.name,
            "class_name": student.class_name,
            "section": student.section,
            "attendance_percentage": (
                student.attendance_percentage
            ),
        },
    }


# =========================================================
# UPDATE STUDENT
# =========================================================

@app.put("/students/{student_id}")
def update_student(
    student_id: int,
    student_data: StudentUpdate,
    db: Session = Depends(get_db),
):

    student = (
        db.query(Student)
        .filter(
            Student.id == student_id
        )
        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found",
        )

    if not 0 <= student_data.attendance_percentage <= 100:

        raise HTTPException(
            status_code=400,
            detail=(
                "Attendance must be between 0 and 100"
            ),
        )

    student.name = student_data.name

    student.class_name = (
        student_data.class_name
    )

    student.section = (
        student_data.section
    )

    student.attendance_percentage = (
        student_data.attendance_percentage
    )

    db.commit()
    db.refresh(student)

    return {
        "message": "Student updated successfully",
        "student": {
            "id": student.id,
            "name": student.name,
            "class_name": student.class_name,
            "section": student.section,
            "attendance_percentage": (
                student.attendance_percentage
            ),
        },
    }


# =========================================================
# DELETE STUDENT
# =========================================================

@app.delete("/students/{student_id}")
def delete_student(
    student_id: int,
    db: Session = Depends(get_db),
):

    student = (
        db.query(Student)
        .filter(
            Student.id == student_id
        )
        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found",
        )

    # Delete attendance records
    db.query(Attendance).filter(
        Attendance.student_id == student_id
    ).delete(
        synchronize_session=False
    )

    db.delete(student)
    db.commit()

    return {
        "message": "Student deleted successfully",
    }


# =========================================================
# TEACHERS - GET ALL
# =========================================================

@app.get("/teachers")
def get_teachers(
    db: Session = Depends(get_db),
):

    teachers = (
        db.query(User)
        .filter(
            User.role == "teacher"
        )
        .all()
    )

    return [
        {
            "id": teacher.id,
            "username": teacher.username,
            "name": teacher.name,
            "role": teacher.role,
        }
        for teacher in teachers
    ]


# =========================================================
# GET SINGLE TEACHER
# =========================================================

@app.get("/teachers/{teacher_id}")
def get_teacher(
    teacher_id: int,
    db: Session = Depends(get_db),
):

    teacher = (
        db.query(User)
        .filter(
            User.id == teacher_id,
            User.role == "teacher",
        )
        .first()
    )

    if not teacher:

        raise HTTPException(
            status_code=404,
            detail="Teacher not found",
        )

    return {
        "id": teacher.id,
        "username": teacher.username,
        "name": teacher.name,
        "role": teacher.role,
    }


# =========================================================
# CREATE TEACHER
# =========================================================

@app.post("/teachers")
def create_teacher(
    teacher_data: TeacherCreate,
    db: Session = Depends(get_db),
):

    existing_user = (
        db.query(User)
        .filter(
            User.username
            == teacher_data.username
        )
        .first()
    )

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Username already exists",
        )

    teacher = User(
        username=teacher_data.username,
        name=teacher_data.name,
        role="teacher",
    )

    db.add(teacher)

    try:

        db.commit()
        db.refresh(teacher)

    except Exception:

        db.rollback()

        raise HTTPException(
            status_code=400,
            detail="Unable to create teacher",
        )

    return {
        "message": "Teacher created successfully",
        "teacher": {
            "id": teacher.id,
            "username": teacher.username,
            "name": teacher.name,
            "role": teacher.role,
        },
    }


# =========================================================
# UPDATE TEACHER
# =========================================================

@app.put("/teachers/{teacher_id}")
def update_teacher(
    teacher_id: int,
    teacher_data: TeacherUpdate,
    db: Session = Depends(get_db),
):

    teacher = (
        db.query(User)
        .filter(
            User.id == teacher_id,
            User.role == "teacher",
        )
        .first()
    )

    if not teacher:

        raise HTTPException(
            status_code=404,
            detail="Teacher not found",
        )

    duplicate = (
        db.query(User)
        .filter(
            User.username
            == teacher_data.username,
            User.id != teacher_id,
        )
        .first()
    )

    if duplicate:

        raise HTTPException(
            status_code=400,
            detail="Username already exists",
        )

    teacher.name = (
        teacher_data.name
    )

    teacher.username = (
        teacher_data.username
    )

    db.commit()
    db.refresh(teacher)

    return {
        "message": "Teacher updated successfully",
        "teacher": {
            "id": teacher.id,
            "username": teacher.username,
            "name": teacher.name,
            "role": teacher.role,
        },
    }


# =========================================================
# DELETE TEACHER
# =========================================================

@app.delete("/teachers/{teacher_id}")
def delete_teacher(
    teacher_id: int,
    db: Session = Depends(get_db),
):

    teacher = (
        db.query(User)
        .filter(
            User.id == teacher_id,
            User.role == "teacher",
        )
        .first()
    )

    if not teacher:

        raise HTTPException(
            status_code=404,
            detail="Teacher not found",
        )

    db.delete(teacher)
    db.commit()

    return {
        "message": "Teacher deleted successfully",
    }


# =========================================================
# ATTENDANCE - GET ALL
# =========================================================

@app.get("/attendance")
def get_attendance(
    db: Session = Depends(get_db),
):

    records = (
        db.query(Attendance)
        .order_by(
            Attendance.date.desc()
        )
        .all()
    )

    return [
        {
            "id": record.id,
            "student_id": record.student_id,
            "date": str(record.date),
            "status": record.status,
        }
        for record in records
    ]


# =========================================================
# ATTENDANCE - CREATE
# =========================================================

@app.post("/attendance")
def create_attendance(
    attendance_data: AttendanceCreate,
    db: Session = Depends(get_db),
):

    # -----------------------------------------------------
    # Validate date
    # -----------------------------------------------------

    try:

        attendance_date = (
            datetime.strptime(
                attendance_data.date,
                "%Y-%m-%d",
            ).date()
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Date must be YYYY-MM-DD",
        )


    # -----------------------------------------------------
    # Validate status
    # -----------------------------------------------------

    status = (
        attendance_data.status
        .lower()
        .strip()
    )

    if status not in [
        "present",
        "absent",
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Status must be present or absent"
            ),
        )


    # -----------------------------------------------------
    # Check student
    # -----------------------------------------------------

    student = (
        db.query(Student)
        .filter(
            Student.id
            == attendance_data.student_id
        )
        .first()
    )

    if not student:

        raise HTTPException(
            status_code=404,
            detail="Student not found",
        )


    # -----------------------------------------------------
    # Check duplicate attendance
    # -----------------------------------------------------

    existing_record = (
        db.query(Attendance)
        .filter(
            Attendance.student_id
            == attendance_data.student_id,

            Attendance.date
            == attendance_date,
        )
        .first()
    )

    if existing_record:

        raise HTTPException(
            status_code=400,
            detail=(
                "Attendance already exists for "
                "this student on this date"
            ),
        )


    # -----------------------------------------------------
    # Create attendance
    # -----------------------------------------------------

    attendance = Attendance(
        student_id=(
            attendance_data.student_id
        ),
        date=attendance_date,
        status=status,
    )

    db.add(attendance)

    db.commit()

    db.refresh(attendance)


    # -----------------------------------------------------
    # Recalculate attendance percentage
    # -----------------------------------------------------

    all_records = (
        db.query(Attendance)
        .filter(
            Attendance.student_id
            == student.id
        )
        .all()
    )

    total_records = len(
        all_records
    )

    present_records = sum(
        1
        for record in all_records
        if record.status.lower()
        == "present"
    )

    if total_records > 0:

        student.attendance_percentage = round(
            (
                present_records
                / total_records
            ) * 100,
            2,
        )

    else:

        student.attendance_percentage = 0


    db.commit()

    db.refresh(student)


    return {
        "message": (
            "Attendance added successfully"
        ),

        "attendance": {
            "id": attendance.id,
            "student_id": attendance.student_id,
            "date": str(
                attendance.date
            ),
            "status": attendance.status,
        },

        "student_attendance_percentage": (
            student.attendance_percentage
        ),
    }


# =========================================================
# DELETE ATTENDANCE
# =========================================================

@app.delete("/attendance/{attendance_id}")
def delete_attendance(
    attendance_id: int,
    db: Session = Depends(get_db),
):

    attendance = (
        db.query(Attendance)
        .filter(
            Attendance.id
            == attendance_id
        )
        .first()
    )

    if not attendance:

        raise HTTPException(
            status_code=404,
            detail=(
                "Attendance record not found"
            ),
        )


    student = (
        db.query(Student)
        .filter(
            Student.id
            == attendance.student_id
        )
        .first()
    )


    db.delete(attendance)

    db.commit()


    # -----------------------------------------------------
    # Recalculate after deletion
    # -----------------------------------------------------

    if student:

        all_records = (
            db.query(Attendance)
            .filter(
                Attendance.student_id
                == student.id
            )
            .all()
        )

        total_records = len(
            all_records
        )

        present_records = sum(
            1
            for record in all_records
            if record.status.lower()
            == "present"
        )

        if total_records > 0:

            student.attendance_percentage = round(
                (
                    present_records
                    / total_records
                ) * 100,
                2,
            )

        else:

            student.attendance_percentage = 0

        db.commit()


    return {
        "message": (
            "Attendance deleted successfully"
        ),
    }


# =========================================================
# API STATUS
# =========================================================

@app.get("/api/status")
def api_status():

    return {
        "project": (
            "XYZ AI School Assistant"
        ),
        "status": "online",
        "backend": "FastAPI",
        "services": {
            "school_management": True,
            "students": True,
            "teachers": True,
            "attendance": True,
            "dashboard": True,
            "ml_prediction": True,
            "ai_assistant": True,
        },
        "message": (
            "API is working successfully"
        ),
    }