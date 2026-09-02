"""
Step 1: Training, with experiment tracking.

This is the ONLY file that changes when you swap in your real fraud dataset later.
Everything downstream (serving, docker, monitoring) just expects a `model.pkl`
that exposes .predict_proba() and a `feature_names.json` describing input columns.

Run:
    python train.py

Then inspect results:
    mlflow ui
    (open http://localhost:5000 in a browser)
"""
import json

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Breast cancer dataset: binary classification (malignant/benign), similar shape
# to a fraud problem (binary target, tabular numeric features) but small and fast.
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# mlflow.set_tracking_uri defaults to a local ./mlruns folder if unset -- fine for now.
# In a real team setting this would point to a shared tracking server.
mlflow.set_experiment("toy-mlops-pipeline")

params = {
    "n_estimators": 200,
    "max_depth": 6,
    "random_state": 42,
}

with mlflow.start_run():
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)

    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("avg_precision", ap)

    # Logs the model as an MLflow artifact too (versioned, alongside the run) --
    # separate from the joblib dump below, which is what the API actually loads.
    mlflow.sklearn.log_model(model, "model")

    print(f"ROC-AUC: {auc:.4f}")
    print(f"Avg Precision: {ap:.4f}")

# This is the file the serving layer (app.py) actually loads at inference time.
# Keeping it a plain joblib dump (rather than requiring MLflow at serving time)
# keeps the Docker image serving-only lighter and decoupled from the tracking server.
joblib.dump(model, "model.pkl")
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("\nSaved model.pkl + feature_names.json for serving.")
print("Run `mlflow ui` to inspect this run in the experiment tracking dashboard.")
