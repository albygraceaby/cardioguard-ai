"""
CardioGuard AI — Machine Learning Module
Heart Disease Prediction using RandomForestClassifier
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "Slope of Peak ST",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia",
}


def _generate_synthetic_data(n_samples=900):
    """Generate a synthetic heart disease dataset when heart.csv is unavailable."""
    rng = np.random.RandomState(42)
    data = {
        "age": rng.randint(29, 78, n_samples),
        "sex": rng.randint(0, 2, n_samples),
        "cp": rng.randint(0, 4, n_samples),
        "trestbps": rng.randint(94, 200, n_samples),
        "chol": rng.randint(126, 564, n_samples),
        "fbs": rng.randint(0, 2, n_samples),
        "restecg": rng.randint(0, 3, n_samples),
        "thalach": rng.randint(71, 202, n_samples),
        "exang": rng.randint(0, 2, n_samples),
        "oldpeak": np.round(rng.uniform(0, 6.2, n_samples), 1),
        "slope": rng.randint(0, 3, n_samples),
        "ca": rng.randint(0, 5, n_samples),
        "thal": rng.randint(0, 4, n_samples),
    }
    df = pd.DataFrame(data)
    # Simulate target with realistic correlations
    risk_score = (
        0.02 * df["age"]
        + 0.3 * df["cp"]
        - 0.01 * df["thalach"]
        + 0.15 * df["oldpeak"]
        + 0.2 * df["ca"]
        + 0.1 * df["exang"]
        + rng.normal(0, 0.3, n_samples)
    )
    df["target"] = (risk_score > np.median(risk_score)).astype(int)
    return df


def load_model():
    """
    Load heart.csv or generate synthetic data, train a RandomForest model.
    Returns: (model, feature_names, X_test, y_test)
    """
    csv_path = os.path.join(os.path.dirname(__file__), "heart.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Ensure expected columns exist
        if "target" not in df.columns:
            df = _generate_synthetic_data()
    else:
        df = _generate_synthetic_data()

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model, FEATURE_NAMES, X_test, y_test


def predict(model, input_data):
    """
    Predict heart disease risk.
    Args:
        model: trained sklearn model
        input_data: list or 1-D array of feature values (len = 13)
    Returns: (prediction: int, probability: float)
    """
    arr = np.array(input_data, dtype=float).reshape(1, -1)
    prediction = int(model.predict(arr)[0])
    probability = float(model.predict_proba(arr)[0][1])
    return prediction, probability


def evaluate_model(model, X_test, y_test):
    """Return accuracy, confusion matrix, and classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, cm, report


def get_top_features(model, feature_names, top_n=3):
    """
    Return the top-N most important features.
    Returns: list of (feature_label, importance_score) tuples
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return [
        (FEATURE_LABELS.get(feature_names[i], feature_names[i]), importances[i])
        for i in indices
    ]


def get_all_feature_importances(model, feature_names):
    """Return all feature importances as a sorted DataFrame."""
    importances = model.feature_importances_
    labels = [FEATURE_LABELS.get(f, f) for f in feature_names]
    df = pd.DataFrame({"Feature": labels, "Importance": importances})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)
