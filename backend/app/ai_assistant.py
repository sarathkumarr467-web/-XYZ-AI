from sqlalchemy.orm import Session

from .models import Student, User


def assistant_response(
    message: str,
    db: Session
):

    text = message.lower().strip()

    # =====================================================
    # GREETING
    # =====================================================

    if text in [
        "hi",
        "hello",
        "hey",
        "hai",
        "good morning",
        "good afternoon",
        "good evening"
    ]:

        return {
            "intent": "greeting",
            "response": (
                "Hello! I am XYZ AI School Assistant. "
                "I can help you with students, attendance, "
                "teachers and school information."
            )
        }

    # =====================================================
    # HELP
    # =====================================================

    if (
        "help" in text
        or "what can you do" in text
    ):

        return {
            "intent": "help",
            "response": (
                "I can help with student information, "
                "attendance, teacher information, "
                "student count and school dashboard details."
            )
        }

    # =====================================================
    # TOTAL STUDENTS
    # =====================================================

    if (
        "how many students" in text
        or "total students" in text
        or "number of students" in text
    ):

        count = db.query(Student).count()

        return {
            "intent": "student_count",
            "count": count,
            "response": (
                f"There are currently {count} "
                f"students registered in the school."
            )
        }

    # =====================================================
    # TOTAL TEACHERS
    # =====================================================

    if (
        "how many teachers" in text
        or "total teachers" in text
        or "number of teachers" in text
    ):

        count = (
            db.query(User)
            .filter(User.role == "teacher")
            .count()
        )

        return {
            "intent": "teacher_count",
            "count": count,
            "response": (
                f"There are currently {count} "
                f"teachers registered."
            )
        }

    # =====================================================
    # STUDENT LIST
    # =====================================================

    if (
        "student list" in text
        or "list students" in text
        or "show students" in text
        or "all students" in text
    ):

        students = db.query(Student).all()

        if not students:

            return {
                "intent": "student_list",
                "students": [],
                "response": (
                    "There are no students "
                    "currently registered."
                )
            }

        names = [
            student.name
            for student in students
        ]

        return {
            "intent": "student_list",
            "students": names,
            "response": (
                "Registered students: "
                + ", ".join(names)
            )
        }

    # =====================================================
    # ATTENDANCE
    # =====================================================

    if "attendance" in text:

        students = db.query(Student).all()

        if not students:

            return {
                "intent": "attendance",
                "response": (
                    "No student attendance "
                    "data is available."
                )
            }

        # -------------------------------------------------
        # Search specific student
        # -------------------------------------------------

        matched_student = None

        for student in students:

            if (
                student.name
                and student.name.lower() in text
            ):

                matched_student = student
                break

        # -------------------------------------------------
        # Specific student attendance
        # -------------------------------------------------

        if matched_student:

            percentage = (
                matched_student.attendance_percentage
                or 0
            )

            return {
                "intent": "attendance",
                "student": matched_student.name,
                "attendance_percentage": percentage,
                "response": (
                    f"{matched_student.name} currently "
                    f"has {percentage}% attendance."
                )
            }

        # -------------------------------------------------
        # Average attendance
        # -------------------------------------------------

        total = sum(
            student.attendance_percentage or 0
            for student in students
        )

        average = total / len(students)

        return {
            "intent": "attendance",
            "average_attendance": round(
                average,
                2
            ),
            "response": (
                f"The current average student "
                f"attendance is {round(average, 2)}%."
            )
        }

    # =====================================================
    # STUDENT INFORMATION
    # =====================================================

    students = db.query(Student).all()

    for student in students:

        if (
            student.name
            and student.name.lower() in text
        ):

            percentage = (
                student.attendance_percentage
                or 0
            )

            return {
                "intent": "student_information",
                "student": {
                    "id": student.id,
                    "name": student.name,
                    "class_name": student.class_name,
                    "section": student.section,
                    "attendance_percentage": percentage
                },
                "response": (
                    f"{student.name} is studying in "
                    f"{student.class_name}, section "
                    f"{student.section}. Current attendance "
                    f"is {percentage}%."
                )
            }

    # =====================================================
    # DASHBOARD
    # =====================================================

    if (
        "dashboard" in text
        or "school summary" in text
        or "school statistics" in text
        or "school status" in text
    ):

        total_students = (
            db.query(Student).count()
        )

        total_teachers = (
            db.query(User)
            .filter(User.role == "teacher")
            .count()
        )

        students = db.query(Student).all()

        if students:

            average = sum(
                student.attendance_percentage or 0
                for student in students
            ) / len(students)

        else:

            average = 0

        return {
            "intent": "dashboard",
            "total_students": total_students,
            "total_teachers": total_teachers,
            "average_attendance": round(
                average,
                2
            ),
            "response": (
                f"School summary: "
                f"{total_students} students, "
                f"{total_teachers} teachers, "
                f"and average attendance is "
                f"{round(average, 2)}%."
            )
        }

    # =====================================================
    # FALLBACK
    # =====================================================

    return {
        "intent": "unknown",
        "response": (
            "I could not understand that request. "
            "Try asking about student information, "
            "attendance, teachers, student count "
            "or dashboard."
        )
    }