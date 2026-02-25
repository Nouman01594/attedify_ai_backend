from sqlalchemy import Column, Integer, String, ForeignKey, Date, Time
from sqlalchemy.orm import relationship
from database import Base
from datetime import date


# ===============================
# STUDENT TABLE
# ===============================
from sqlalchemy import Column, Integer, String, LargeBinary
from sqlalchemy.orm import relationship
from database import Base
from datetime import date
from sqlalchemy import ForeignKey, Date, Time

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    father_name = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    roll_no = Column(String, unique=True, nullable=False)
    uid = Column(String, unique=True, nullable=False, index=True)
    image_path = Column(String)
    face_encoding = Column(LargeBinary)   # 🔥 NEW COLUMN

    attendance_records = relationship("Attendance", back_populates="student")



# ===============================
# STUDENT ATTENDANCE TABLE
# ===============================
class Attendance(Base):
    __tablename__ = "student_attendance"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    uid = Column(String, index=True)

    date = Column(Date, default=date.today)
    intime = Column(Time, nullable=True)
    outtime = Column(Time, nullable=True)
    auth_method = Column(String, nullable=False, default='UID')
    

    student = relationship("Student", back_populates="attendance_records")
