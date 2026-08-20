from .database import SessionLocal, engine, Base
from .models import User, Student


Base.metadata.create_all(bind=engine)

db = SessionLocal()

# Create users only if they don't already exist

users = [
    {
        "username": "student1",
        "name": "Rahul",
        "role": "student"
    },
    {
        "username": "parent1",
        "name": "Mr. Kumar",
        "role": "parent"
    },
    {
        "username": "teacher1",
        "name": "Ms. Priya",
        "role": "teacher"
    },
    {
        "username": "principal1",
        "name": "Dr. Arun",
        "role": "principal"
    }
]

for user_data in users:

    existing = db.query(User).filter(
        User.username == user_data["username"]
    ).first()

    if not existing:
        user = User(**user_data)
        db.add(user)


# Demo students

if db.query(Student).count() == 0:

    students = [
        Student(
            name="Rahul",
            class_name="10",
            section="A",
            attendance_percentage=92.5,
            parent_id=2,
            teacher_id=3
        ),
        Student(
            name="Priya",
            class_name="10",
            section="A",
            attendance_percentage=87.0,
            parent_id=2,
            teacher_id=3
        ),
        Student(
            name="Arun",
            class_name="10",
            section="B",
            attendance_percentage=76.5,
            parent_id=None,
            teacher_id=3
        )
    ]

    db.add_all(students)


db.commit()
db.close()

print("Database seeded successfully!")