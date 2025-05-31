from fastapi import FastAPI, Depends
from sklearn.exceptions import NotFittedError

from connexion import get_cassandra_session
from schemas import StudentCreate, Patient
from uuid import uuid4
import joblib
import dill
import numpy as np
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import joblib, dill

columns = [
    "USMER", "MEDICAL_UNIT", "SEX", "PATIENT_TYPE", "INTUBED", "PNEUMONIA", "AGE", "PREGNANT",
    "DIABETES", "COPD", "ASTHMA", "INMSUPR", "HIPERTENSION", "OTHER_DISEASE", "CARDIOVASCULAR",
    "OBESITY", "RENAL_CHRONIC", "TOBACCO", "CLASIFFICATION_FINAL", "ICU"
]
# Charger le modèle
#model = joblib.load("mon_modele.pkl")
# Charger le modèle
with open("Covid-19_Death_Predict_Pipeline.pkl", "rb") as f:
    model = dill.load(f)

app = FastAPI()

@app.get("/students")
def read_users(session=Depends(get_cassandra_session)):
    rows = session.execute("SELECT * FROM students")
    return [dict(row._asdict()) for row in rows]

@app.post("/students")
def create_Student(student:StudentCreate,session=Depends(get_cassandra_session)):
        new_id= uuid4()
        query = """
               INSERT INTO students (student_id,name,age,email) 
               VALUES(%s, %s, %s,%s)
        """
        session.execute(query, (new_id ,student.name,student.age,student.email))

        return {'message':'student create successfully','id':str(new_id),'name':student.name}

@app.post("/predict")
async def predict_patient_risk(patient: Patient):
    # Convertir le patient en format que le modèle comprend (ex: tableau)
    input_data = np.array([
        patient.USMER,
        patient.MEDICAL_UNIT,
        patient.SEX,
        patient.PATIENT_TYPE,
        patient.INTUBED,
        patient.PNEUMONIA,
        patient.AGE,
        patient.PREGNANT,
        patient.DIABETES,
        patient.COPD,
        patient.ASTHMA,
        patient.INMSUPR,
        patient.HIPERTENSION,
        patient.OTHER_DISEASE,
        patient.CARDIOVASCULAR,
        patient.OBESITY,
        patient.RENAL_CHRONIC,
        patient.TOBACCO,
        patient.CLASIFFICATION_FINAL,
        patient.ICU
    ])
    a = pd.DataFrame([input_data], columns=columns)
    print(input_data)

    # Prédiction
    prediction = model.predict(a)[0]

    return {
        "prediction": int(prediction)
    }

