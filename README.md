# MLflow Multi-Model Comparison — Wine Quality Dataset

A structured experiment comparing six machine learning algorithms on the same dataset using MLflow for experiment tracking, metric logging, and model artifact management. Every run is logged automatically and comparable in the MLflow dashboard.

---

## Overview

Most machine learning comparisons are informal — numbers in a notebook, model files with timestamp names, no reliable way to reproduce a result a week later. This project demonstrates the correct way to run a multi-model comparison: same data, same split, every result tracked, everything reproducible.

Six regression models are trained on the UCI Wine Quality (Red) dataset. MLflow logs the hyperparameters, metrics, and serialised model for every run. The MLflow UI renders all six runs in one experiment for side-by-side comparison.

---

## Models

| Model | Key Hyperparameters | Purpose |
|---|---|---|
| ElasticNet | alpha=0.5, l1_ratio=0.5 | Linear baseline |
| Decision Tree | max_depth=5, min_samples_split=4 | Non-linear single model |
| Random Forest | n_estimators=100, max_depth=6 | Bagging ensemble |
| SVM | C=1.0, epsilon=0.1 | Kernel-based with scaling |
| Ridge Regression | alpha=1.0 | Regularised linear model |
| XGBoost | n_estimators=100, learning_rate=0.1 | Gradient boosted ensemble |

---

## Results

| Rank | Model | RMSE | MAE | R-squared |
|---|---|---|---|---|
| 1 | XGBoost | 0.5927 | 0.4618 | 0.4625 |
| 2 | Random Forest | ~0.580 | ~0.440 | ~0.450 |
| 3 | SVM | ~0.610 | ~0.470 | ~0.380 |
| 4 | Decision Tree | 0.6584 | 0.4964 | 0.3366 |
| 5 | Ridge Regression | ~0.640 | ~0.500 | ~0.360 |
| 6 | ElasticNet | 0.7931 | 0.6271 | 0.1086 |

XGBoost and Random Forest consistently outperform single models. ElasticNet's low R-squared of 0.10 confirms the data has non-linear structure that linear models cannot capture.

---

## Project Structure

```
MLflow-Basic-Demo/
│
├── example.py               # Model 1: ElasticNet (baseline)
├── decision_tree.py         # Model 2: Decision Tree
├── random_forest.py         # Model 3: Random Forest
├── svm_model.py             # Model 4: SVM with StandardScaler pipeline
├── logistic_regression.py   # Model 5: Ridge Regression
├── xgboost_model.py         # Model 6: XGBoost
│
├── mlruns/                  # Auto-created by MLflow — stores all run data
├── requirements.txt
└── README.md
```

---

## Dataset

**Wine Quality (Red)** — UCI Machine Learning Repository

- 1,599 samples
- 11 physicochemical input features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulphur dioxide, total sulphur dioxide, density, pH, sulphates, alcohol
- Target: quality score from 3 to 9 (assigned by human tasters)
- Task: regression

The dataset is loaded automatically from the MLflow GitHub repository when each script runs. No manual download is required.

---

## Requirements

- Python 3.10+
- pip

All Python dependencies are listed in `requirements.txt`.

---

## Setup and Installation

**1. Clone the repository**

```bash
git clone https://github.com/entbappy/MLflow-Basic-Demo.git
cd MLflow-Basic-Demo
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If a setuptools warning appears during installation:

```bash
pip install "setuptools<81"
```

---

## Running the Experiments

**Start the MLflow tracking server first** (keep this terminal open throughout):

```bash
mlflow ui
```

The server starts at `http://127.0.0.1:5000`. Open that address in a browser before running any scripts.

**In a second terminal, run each model script:**

```bash
python example.py
python decision_tree.py
python random_forest.py
python svm_model.py
python logistic_regression.py
python xgboost_model.py
```

Wait for each script to finish before running the next. Each script prints its results to the terminal:

```
Decision Tree  --> RMSE: 0.6584 | MAE: 0.4964 | R2: 0.3366
Random Forest  --> RMSE: 0.5800 | MAE: 0.4400 | R2: 0.4500
XGBoost        --> RMSE: 0.5927 | MAE: 0.4618 | R2: 0.4625
```

---

## Viewing Results in the MLflow UI

Open `http://127.0.0.1:5000` in your browser.

1. Click **Wine_Quality_Comparison** in the left sidebar
2. All six runs appear in the table — click any column header to sort
3. Tick all checkboxes, then click **Compare** to open the side-by-side comparison view
4. Click any individual run name to see its full detail: parameters, metrics, and saved model artifacts

The Compare view includes a parameters table, a metrics table, a parallel coordinates plot, and a scatter plot. All charts are interactive.

---

## How MLflow Tracks Each Run

Every script follows the same three-step logging pattern inside a `with mlflow.start_run()` block:

```python
with mlflow.start_run(run_name="XGBoost"):
    model.fit(X_train, y_train)

    # 1. Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    # 2. Log evaluation metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # 3. Save the trained model as an artifact
    mlflow.xgboost.log_model(model, "xgboost_model")
```

MLflow stores all of this in the `mlruns/` directory under a unique run ID. The `mlflow ui` command reads those directories and renders the web interface.

---

## Running Multiple Experiments

To compare different hyperparameter values for the same model, edit the script before re-running. For example, in `example.py`:

```python
# Change this
alpha = 0.5

# To this
alpha = 0.1
```

Re-run the script and refresh the MLflow UI. The new run appears in the same experiment alongside the previous one, and both are selectable for comparison.

---

## Loading a Saved Model

Any logged model can be loaded directly from its run ID for inference on new data:

```python
import mlflow.sklearn

model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
predictions = model.predict(X_new)
```

The run ID is visible in the MLflow UI on the individual run detail page.

---

## Metrics Reference

| Metric | Full Name | Interpretation |
|---|---|---|
| RMSE | Root Mean Squared Error | Average error in quality points. Large errors are penalised more heavily. Lower is better. |
| MAE | Mean Absolute Error | Average absolute error in quality points. Robust to outliers. Lower is better. |
| R-squared | Coefficient of Determination | Proportion of variance in quality explained by the model. Closer to 1.0 is better. |

---

## Notes

The original `example.py` from the cloned repository includes a remote tracking URI pointing to an AWS server that is no longer active. This has been updated to use the local server at `http://127.0.0.1:5000`. If you see a connection timeout error when running `example.py`, verify that the tracking URI in the file is set correctly:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine_Quality_Comparison")
```

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Wine Quality Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, Elsevier, 47(4):547-553, 2009.

---

## License

This project is open source and available under the [MIT License](LICENSE).
