import os
import dotenv
import mlflow
import requests
import json
import numpy as np
dotenv.load_dotenv("../.env")
def test_model_availability():
    # Arrange
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    print(MLFLOW_TRACKING_URI)
    CHOSEN_MODEL = os.getenv("MODEL_ALIAS")
    print(CHOSEN_MODEL)
    mlflow.set_tracking_uri(uri = MLFLOW_TRACKING_URI ) 
    mlflow.set_experiment("Churn Experiment")
    
    # Act
    #chosen_model = mlflow.pyfunc.load_model(f"models:/RandomForest@stage")
    # Assert
    #assert type(chosen_model) == mlflow.pyfunc.PyFuncModel
def test_predict_on_ci():
    # Arrange
    MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
    print (MODEL_SERVER_IP)
    data = {"CreditScore": -0.3262214220367463, "Geography": 0.0, "Gender": 0.0, "Age": 0.2935174228967471, "Tenure": 2.0, "Balance": -1.2258476714090278,"NumOfProducts": 1.0,"HasCrCard": 1.0,"IsActiveMember": 1.0,"EstimatedSalary": 101348.88}

    # Act
    #res = requests.post("http://localhost:8080/predict/", json = data)

    #Assert
    #assert res.status_code == 200




