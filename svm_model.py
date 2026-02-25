import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

C = 1.0
epsilon = 0.1

mlflow.set_experiment("Wine_Quality_Comparison")

with mlflow.start_run(run_name="SVM"):
    # SVM needs scaling, so we use a Pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(C=C, epsilon=epsilon))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_param("C", C)
    mlflow.log_param("epsilon", epsilon)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "svm_model")

    print(f"SVM → RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")