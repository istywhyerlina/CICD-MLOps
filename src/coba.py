import mlflow
import numpy as np
import pandas as pd
from sklearn import datasets
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

'''from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

'''
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score


data = pd.read_csv("../../project/data/processed/data.csv", sep = "\t")
data['Geography'] = data.Geography.astype(float)
data['Gender'] = data.Gender.astype(float)
data['Tenure'] = data.Tenure.astype(float)
data['NumOfProducts'] = data.NumOfProducts.astype(float)
data['HasCrCard'] = data.HasCrCard.astype(float)
data['IsActiveMember'] = data.IsActiveMember.astype(float)

data['Exited'] = data.Exited.astype(str)
y=data['Exited']
X=data.drop('Exited',  axis='columns')
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size = 0.2, random_state=17
) 

#SMOTE (for imbalanced data)
over = SMOTE(sampling_strategy='auto', random_state=33)
X_train, y_train = over.fit_resample(X_train, y_train)


params={"bootstrap": True,
    "criterion": "gini",
    "max_depth": 6,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
    "n_estimators": 100}

rf = RandomForestClassifier(**params)

rf = rf.fit(X_train, y_train)
tes = pd.DataFrame({'CreditScore': [-0.3262214220367463, 0.7], 'Geography':[0.0,0.0], 'Gender': [0.0,0.0], 'Age':[0.2935174228967471,-0.957375], 'Tenure': [2.0,4.0], 'Balance':[ -1.2258476714090278,1.0], 'NumOfProducts':[ 1.0,1.0], 'HasCrCard':[ 1.0,1.0], 'IsActiveMember': [1.0,1.0], 'EstimatedSalary': [101348.88,101]}) 

#tes = np.array([[-0.3262214220367463, 0.0,  0.0,  0.2935174228967471, 2.0, -1.2258476714090278, 1.0, 1.0, 1.0,101348.88]])
print(tes)
print(X_test)

y_pred = rf.predict(tes) 
#accuracy = accuracy_score(y_test, y_pred) 
print(y_pred)
