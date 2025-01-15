import os
import mlflow
import dotenv
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
dotenv.load_dotenv(".env")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
CHOSEN_MODEL = os.getenv("MODEL_ALIAS")
mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI)
mlflow.set_experiment("Flower Classification")
model = mlflow.pyfunc.load_model(f"models:/Untouch Logistic Regression@{CHOSEN_MODEL}")
class api_data(BaseModel):
    x1 : float
    x2 : float
    x3 : float
    x4 : float
    x5 : float
    x6 : float
    x7 : float
    x8 : float
    x9 : float
    x10 : float
app = FastAPI()
@app.get("/")
def home():
	    return "Hello, FastAPI up!"
@app.post("/predict/")
def predict(data: api_data):
    data = np.array([[data.x1, data.x2, data.x3, data.x4,data.x5,data.x6,data.x7, data.x8,data.x9,data.x10]]).astype(np.float64)
    y_pred = int(model.predict(data)[0])
    return {"res" : y_pred, "error_msg": ""}
if __name__ == "__main__":
	    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
