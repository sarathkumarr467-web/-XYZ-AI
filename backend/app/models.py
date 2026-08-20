from sqlalchemy import Column, Integer, String, Float, Date
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    section = Column(String, nullable=False)
    attendance_percentage = Column(Float, default=0.0)
    parent_id = Column(Integer, nullable=True)
    teacher_id = Column(Integer, nullable=True)


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    status = Column(String, nullable=False)