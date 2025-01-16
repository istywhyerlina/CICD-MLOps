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
    CHOSEN_MODEL = os.getenv("MODEL_ALIAS")
    mlflow.set_tracking_uri(uri = "http://localhost:5000") 
    mlflow.set_experiment("Churn Experiment")
    
    # Act
    chosen_model = mlflow.pyfunc.load_model(f"models:/RandomForest@{CHOSEN_MODEL}")
    # Assert
    assert type(chosen_model) == mlflow.pyfunc.PyFuncModel
def test_predict_on_ci():
    # Arrange
    MODEL_SERVER_IP = os.getenv("MODEL_SERVER_IP")
    data = {"CreditScore": -0.3262214220367463, "Geography": 0.0, "Gender": 0.0, "Age": 0.2935174228967471, "Tenure": 2.0, "Balance": -1.2258476714090278,"NumOfProducts": 1.0,"HasCrCard": 1.0,"IsActiveMember": 1.0,"EstimatedSalary": 101348.88}
    #api_data = {"CreditScore": np.double(-0.3262214220367463), "Geography": np.double(0.0), "Gender": np.double(0.0), "Age": np.double(0.2935174228967471), "Tenure": np.double(2.0), "Balance": np.double(-1.2258476714090278),"NumOfProducts": np.double(1.0),"HasCrCard": np.double(1.0),"IsActiveMember": np.double(1.0),"EstimatedSalary": np.double(101348.88)}

    # Act
    res = requests.post("http://localhost:8080/predict/", json = data)
    #print(res.status_code)
    #print(res.text)
    #Assert
    assert res.status_code == 200
test_model_availability()
test_predict_on_ci()



