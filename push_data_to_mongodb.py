import pymongo
import os
import pandas as pd
import json
from dotenv import load_dotenv
load_dotenv()
MONGO_DB_URL=os.getenv("MONGO_DB_URL")
DATABASE_NAME="HealthAndFitness"
COLLECTION_NAME="Raw Data"
CSV_FILE_PATH="/Users/malik/Desktop/HealthAndFitness/Data/Final_data.csv"

client=pymongo.MongoClient(MONGO_DB_URL)
db=client[DATABASE_NAME]
collection=db[COLLECTION_NAME]

df=pd.read_csv(CSV_FILE_PATH)
records=json.loads(df.to_json(orient="records"))
collection.insert_many(records)
print("Successfully inserted data in MongoDB")