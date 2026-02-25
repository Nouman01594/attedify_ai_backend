from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.orm import Session
import shutil
import os
import pickle
from datetime import datetime, date
from pathlib import Path
from PIL import Image
import numpy as np
import uuid
import cv2
import io
import logging

from database import engine, SessionLocal, Base
import models
import face_recognition

# ===============================
# LOGGING SETUP
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# -------------------------------
# Desktop student_images path
# -------------------------------
desktop_path = Path.home()
student_images_path = os.path.join(desktop_path, "student_images")
os.makedirs(student_images_path, exist_ok=True)

# Mount image folder from Desktop
app.mount("/student_images", StaticFiles(directory=student_images_path), name="student_images")

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===============================
# HELPER: Convert any image to RGB numpy array
# ===============================
def load_image_for_face_recognition(image_path):
    pil_image = Image.open(image_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    np_arr = np.frombuffer(buffer.read(), dtype=np.uint8)
    bgr_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


# ===============================
# ADD STUDENT API
# ===============================
@app.post("/students/add")
def add_student(
    name: str = Form(...),
    father_name: str = Form(...),
    class_name: str = Form(...),
    roll_no: str = Form(...),
    uid: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # ✅ Normalize UID
    uid = uid.strip().upper()

    unique_filename = f"{uuid.uuid4()}_{image.filename}"
    image_path = os.path.join(student_images_path, unique_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        loaded_image = load_image_for_face_recognition(image_path)
        encodings = face_recognition.face_encodings(loaded_image)

        if not encodings:
            os.remove(image_path)
            return {"error": "No face detected in image"}

        encoding_bytes = pickle.dumps(encodings[0])

    except Exception as e:
        if os.path.exists(image_path):
            os.remove(image_path)
        return {"error": f"Image processing failed: {str(e)}"}

    new_student = models.Student(
        name=name,
        father_name=father_name,
        class_name=class_name,
        roll_no=roll_no,
        uid=uid,
        image_path=image_path,
        face_encoding=encoding_bytes
    )

    db.add(new_student)
    db.commit()
    db.refresh(new_student)

    from sqlalchemy import text
    result = db.execute(text("SELECT current_database();"))
    current_db = result.fetchone()[0]

    logger.info(f"Student added: {name} (Roll: {roll_no}, UID: {uid})")

    return {
        "message": "Student added successfully",
        "student_id": new_student.id,
        "database_used": current_db
    }


# ===============================
# GET ALL STUDENTS
# ===============================
from typing import List
import schemas

@app.get("/students/", response_model=List[schemas.StudentResponse])
def get_students(db: Session = Depends(get_db)):
    students = db.query(models.Student).all()
    return students

@app.get("/students/roll/{roll_no}", response_model=schemas.StudentResponse)
def get_student_by_roll(roll_no: str, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(models.Student.roll_no == roll_no).first()
    if not student:
        return {"error": "Student not found"}
    return student


@app.delete("/students/roll/{roll_no}")
def delete_student_by_roll(roll_no: str, db: Session = Depends(get_db)):
    student = db.query(models.Student).filter(models.Student.roll_no == roll_no).first()
    if not student:
        return {"error": "Student not found"}
    db.delete(student)
    db.commit()
    logger.info(f"Student deleted: Roll {roll_no}")
    return {"message": "Student deleted successfully"}


@app.put("/students/roll/{roll_no}")
def update_student_by_roll(
    roll_no: str,
    name: str = Form(...),
    father_name: str = Form(...),
    class_name: str = Form(...),
    uid: str = Form(...),
    db: Session = Depends(get_db)
):
    student = db.query(models.Student).filter(models.Student.roll_no == roll_no).first()
    if not student:
        return {"error": "Student not found"}

    student.name = name
    student.father_name = father_name
    student.class_name = class_name
    student.uid = uid.strip().upper()  # ✅ Normalize UID

    db.commit()
    db.refresh(student)

    logger.info(f"Student updated: Roll {roll_no}")
    return {"message": "Student updated successfully"}


# ===============================
# ATTENDANCE ENDPOINTS
# ===============================

@app.post("/attendance/mark")
def mark_attendance(
    uid: str = Form(None),
    image: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    if not uid and not image:
        raise HTTPException(
            status_code=400,
            detail="Either UID or image must be provided"
        )

    # ✅ Normalize UID
    if uid:
        uid = uid.strip().upper()

    matched_student = None
    auth_method = None

    # 1️⃣ UID MATCH
    if uid:
        logger.info(f"Attempting UID match: {uid}")
        matched_student = db.query(models.Student).filter(
            models.Student.uid == uid
        ).first()
        if matched_student:
            auth_method = "RFID"
            logger.info(f"Student matched by UID: {matched_student.name}")

    # 2️⃣ FACE MATCH
    if image and not matched_student:
        logger.info("Attempting face recognition")
        temp_filename = f"temp_{uuid.uuid4()}_{image.filename}"
        temp_path = os.path.join(student_images_path, temp_filename)

        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            unknown_image = load_image_for_face_recognition(temp_path)
            unknown_encodings = face_recognition.face_encodings(unknown_image)

            if not unknown_encodings:
                logger.warning("No face detected in uploaded image")
            else:
                unknown_encoding = unknown_encodings[0]
                students = db.query(models.Student).all()

                for student in students:
                    if student.face_encoding:
                        known_encoding = pickle.loads(student.face_encoding)
                        matches = face_recognition.compare_faces(
                            [known_encoding], unknown_encoding, tolerance=0.6
                        )
                        if matches[0]:
                            matched_student = student
                            auth_method = "FACE"
                            logger.info(f"Student matched by face: {student.name}")
                            break

        except Exception as e:
            logger.error(f"Face recognition error: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 3️⃣ AUTHENTICATION FAILED
    if not matched_student:
        logger.warning("Authentication failed - no match found")
        return {
            "status": "failed",
            "message": "Student not recognized. Please try again or contact admin.",
            "auth_method": None
        }

    # 4️⃣ ATTENDANCE MARKING LOGIC
    today = date.today()
    current_time = datetime.now().time()

    records_today = db.query(models.Attendance).filter(
        models.Attendance.student_id == matched_student.id,
        models.Attendance.date == today
    ).order_by(models.Attendance.id.desc()).all()

    completed_entries = sum(1 for r in records_today if r.intime and r.outtime)

    if completed_entries >= 10:
        return {
            "status": "failed",
            "message": "Daily attendance limit reached (10/10).",
            "student_name": matched_student.name,
            "completed_entries": completed_entries
        }

    latest_record = records_today[0] if records_today else None

    if not latest_record:
        new_attendance = models.Attendance(
            student_id=matched_student.id,
            uid=matched_student.uid,
            date=today,
            intime=current_time,
            outtime=None,
            auth_method=auth_method
        )
        db.add(new_attendance)
        db.commit()
        db.refresh(new_attendance)
        return {
            "status": "success",
            "type": "IN",
            "message": f"Welcome {matched_student.name}! Entry time recorded.",
            "student_name": matched_student.name,
            "time": current_time.strftime("%I:%M %p"),
            "auth_method": auth_method,
            "entries_today": 1,
            "entries_remaining": 9
        }

    if latest_record.intime and latest_record.outtime is None:
        latest_record.outtime = current_time
        db.commit()
        duration = datetime.combine(today, current_time) - datetime.combine(today, latest_record.intime)
        duration_str = str(duration).split('.')[0]
        return {
            "status": "success",
            "type": "OUT",
            "message": f"Goodbye {matched_student.name}! Exit time recorded.",
            "student_name": matched_student.name,
            "time": current_time.strftime("%I:%M %p"),
            "duration": duration_str,
            "auth_method": auth_method,
            "entries_today": completed_entries + 1,
            "entries_remaining": 10 - (completed_entries + 1)
        }

    new_attendance = models.Attendance(
        student_id=matched_student.id,
        uid=matched_student.uid,
        date=today,
        intime=current_time,
        outtime=None,
            auth_method=auth_method
    )
    db.add(new_attendance)
    db.commit()
    db.refresh(new_attendance)
    return {
        "status": "success",
        "type": "IN",
        "message": f"Welcome back {matched_student.name}! Entry time recorded.",
        "student_name": matched_student.name,
        "time": current_time.strftime("%I:%M %p"),
        "auth_method": auth_method,
        "entries_today": completed_entries + 1,
        "entries_remaining": 10 - (completed_entries + 1)
    }


# ===============================
# GET ATTENDANCE RECORDS
# ===============================
@app.get("/attendance/records")
def get_attendance_records(
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.Attendance).order_by(models.Attendance.id.desc())

    if start_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            query = query.filter(models.Attendance.date >= start)
        except ValueError:
            logger.warning(f"Invalid start_date format: {start_date}")

    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
            query = query.filter(models.Attendance.date <= end)
        except ValueError:
            logger.warning(f"Invalid end_date format: {end_date}")

    records = query.all()

    result = []
    for record in records:
        # ✅ Get student roll_no
        student = db.query(models.Student).filter(
            models.Student.id == record.student_id
        ).first()

        result.append({
            "id": record.id,
            "student_id": record.student_id,
            "roll_no": student.roll_no if student else None,  # ✅ ADDED
            "uid": record.uid,
            "date": str(record.date) if record.date else None,
            "intime": str(record.intime) if record.intime else None,
            "outtime": str(record.outtime) if record.outtime else None,
        })

    logger.info(f"Returning {len(result)} attendance records")
    return result


# ===============================
# GET TODAY'S ATTENDANCE
# ===============================
@app.get("/attendance/today")
def get_today_attendance(db: Session = Depends(get_db)):
    today = date.today()
    records = db.query(models.Attendance).filter(
        models.Attendance.date == today
    ).order_by(models.Attendance.id.desc()).all()

    result = []
    for record in records:
        student = db.query(models.Student).filter(
            models.Student.id == record.student_id
        ).first()

        result.append({
            "id": record.id,
            "student_id": record.student_id,
            "roll_no": student.roll_no if student else None,  # ✅ ADDED
            "uid": record.uid,
            "date": str(record.date),
            "intime": str(record.intime) if record.intime else None,
            "outtime": str(record.outtime) if record.outtime else None,
        })

    logger.info(f"Returning {len(result)} records for today ({today})")
    return result


# ===============================
# GET ATTENDANCE STATS
# ===============================
@app.get("/attendance/stats")
def get_attendance_stats(db: Session = Depends(get_db)):
    today = date.today()
    total_records = db.query(models.Attendance).count()
    today_entries = db.query(models.Attendance).filter(
        models.Attendance.date == today
    ).count()
    unique_students_today = db.query(models.Attendance.student_id).filter(
        models.Attendance.date == today
    ).distinct().count()

    return {
        "total_records": total_records,
        "today_entries": today_entries,
        "unique_students_today": unique_students_today,
        "date": str(today)
    }