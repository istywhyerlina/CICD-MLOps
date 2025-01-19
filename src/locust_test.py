import numpy as np

from random import randint
from locust import TaskSet, constant, task, HttpUser

class HitServer(TaskSet):
    @task
    def get_url(self):
        self.client.get("/")

class HitEndPoint(TaskSet):
    @task
    def post_predict(self):
        data= {
                "CreditScore" : float(np.random.uniform(-1,1,1)[0])
                "Geography" : float(randint(0,1))
                "Gender" : float(randint(0,1))
                "Age" : float(np.random.uniform(-1,1,1)[0])
                "Tenure" : float(randint(0,10))
                "Balance" : float(np.random.uniform(-1,1,1)[0])
                "NumOfProducts" : float(randint(0,7))
                "HasCrCard" : float(randint(0,10))
                "IsActiveMember" : float(randint(0,10))
                "EstimatedSalary" : float(np.random.uniform(20000,100000,1)[0])

        }

        self.client.post(
            "/predict",
            json = data
        )

class UserLoadTest(HttpUser):
    host = "https://istyy.crabdance.com/"
    task = [HitEndPoint]
    wait_time = constant(1)

