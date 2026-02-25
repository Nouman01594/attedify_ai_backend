from pydantic import BaseModel

class StudentResponse(BaseModel):
    id: int
    name: str
    father_name: str
    class_name: str
    roll_no: str
    uid: str
    image_path: str

    class Config:
        orm_mode = True
