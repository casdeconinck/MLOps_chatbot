import mlflow
from mlflow.models import infer_signature


mlflow.set_tracking_uri(uri="http://localhost:8080")

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

mlflow.set_experiment("MLfow Test")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy_score", accuracy)
    mlflow.set_tag("Train info", "Basic LR model for iris data")
    signature = infer_signature(X_train, lr.predict(X_train))

    #log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking_quickstart",
    )
