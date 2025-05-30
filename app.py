from fastapi import FastAPI, Depends
from connexion import get_cassandra_session
from schemas import StudentCreate
from uuid import uuid4

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


