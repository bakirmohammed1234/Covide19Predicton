from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
import pandas as pd
from uuid import uuid4
import time

# --- Configuration Cassandra ---
KEYSPACE = "bigdataproject"
TABLE = "covide119"


def insert_data(session, dataframe):
    insert_stmt = session.prepare(f"""
        INSERT INTO {TABLE} (
            id, USMER, MEDICAL_UNIT, SEX, PATIENT_TYPE, INTUBED, PNEUMONIA, AGE,
            PREGNANT, DIABETES, COPD, ASTHMA, INMSUPR, HIPERTENSION, OTHER_DISEASE,
            CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO, CLASIFFICATION_FINAL,
            ICU, Y
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    dataframe = dataframe.fillna(0)
    total = len(dataframe)
    count = 0

    print(f"📄 {total} lignes à insérer dans Cassandra (sans batch)...")

    for index, row in dataframe.iterrows():
        try:
            session.execute(insert_stmt, (
                uuid4(),
                int(row['USMER']),
                int(row['MEDICAL_UNIT']),
                int(row['SEX']),
                int(row['PATIENT_TYPE']),
                int(row['INTUBED']),
                int(row['PNEUMONIA']),
                int(row['AGE']),
                int(row['PREGNANT']),
                int(row['DIABETES']),
                int(row['COPD']),
                int(row['ASTHMA']),
                int(row['INMSUPR']),
                int(row['HIPERTENSION']),
                int(row['OTHER_DISEASE']),
                int(row['CARDIOVASCULAR']),
                int(row['OBESITY']),
                int(row['RENAL_CHRONIC']),
                int(row['TOBACCO']),
                int(row['CLASIFFICATION_FINAL']),
                int(row['ICU']),
                int(row['Y'])
            ))
            count += 1
            if count % 1000 == 0:
                print(f"✅ {count}/{total} lignes insérées...")
        except Exception as e:
            print(f"❌ Erreur ligne {index}: {e}")
            continue

    print(f"✅ Insertion terminée : {count}/{total} lignes insérées.")


def main():
    cluster = Cluster(['127.0.0.1'])
    session = cluster.connect(KEYSPACE)

    try:
        df = pd.read_csv("D:\\MlAIM\\BigData\\Covid Data1.csv")
        print("📥 Fichier CSV chargé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du CSV : {e}")
        return

    insert_data(session, df)

    cluster.shutdown()


if __name__ == "__main__":
    main()
