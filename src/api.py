import os
import mlflow
import dotenv
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json
dotenv.load_dotenv("../.env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
CHOSEN_MODEL = os.getenv("MODEL_ALIAS")
mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn Experiment")
model = mlflow.pyfunc.load_model(f"models:/RandomForest@{CHOSEN_MODEL}")
class api_data(BaseModel):
    CreditScore : float
    Geography : float
    Gender : float
    Age : float
    Tenure : float
    Balance : float
    NumOfProducts : float
    HasCrCard : float
    IsActiveMember : float
    EstimatedSalary : float
app = FastAPI()
@app.get("/")
def home():
	    return "Hello, FastAPI up!"
@app.post("/predict/")
def predict(data: api_data):
    data = pd.DataFrame({'CreditScore': [data.CreditScore], 'Geography': [data.Geography], 'Gender': [data.Gender], 'Age': [data.Age], 'Tenure': [data.Tenure], 'Balance': [data.Balance], 'NumOfProducts': [data.NumOfProducts], 'HasCrCard': [data.HasCrCard], 'IsActiveMember': [data.IsActiveMember], 'EstimatedSalary': [data.EstimatedSalary]}) 
    y_pred = int(model.predict(data)[0])
    print(y_pred)
    return {"res" : y_pred, "error_msg": ""}
if __name__ == "__main__":
	    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
