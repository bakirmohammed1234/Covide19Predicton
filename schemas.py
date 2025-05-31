from pydantic import BaseModel, EmailStr

class StudentCreate(BaseModel):
    name: str
    age: int
    email: str

class Patient(BaseModel):
    USMER: int
    MEDICAL_UNIT: int
    SEX: int
    PATIENT_TYPE: int
    INTUBED: int
    PNEUMONIA: int
    AGE: int
    PREGNANT: int
    DIABETES: int
    COPD: int
    ASTHMA: int
    INMSUPR: int
    HIPERTENSION: int
    OTHER_DISEASE: int
    CARDIOVASCULAR: int
    OBESITY: int
    RENAL_CHRONIC: int
    TOBACCO: int
    CLASIFFICATION_FINAL: int
    ICU: int
