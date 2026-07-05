import os
import pickle

import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "loan_model.pkl")

FEATURE_INFO = {
    "Age":         {"label": "Age (years)",     "min": 18,  "max": 100,        "step": 1},
    "Income":      {"label": "Annual Income",   "min": 0,   "max": 10_000_000, "step": 1000},
    "LoanAmount":  {"label": "Loan Amount",     "min": 0,   "max": 1_000_000,  "step": 500},
    "CreditScore": {"label": "Credit Score",    "min": 300, "max": 850,        "step": 1},
}


def load_bundle(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'.\n"
            "Train the model first by running train.py (it writes model/loan_model.pkl)."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def preprocess(raw, bundle):
    features = bundle["features"]
    bounds = bundle["bounds"]

    row = {feat: float(raw[feat]) for feat in features}
    df = pd.DataFrame([row], columns=features)

    for feat, (low, high) in bounds.items():
        df[feat] = df[feat].clip(low, high)

    return df


def predict(raw, bundle=None):
    if bundle is None:
        bundle = load_bundle()

    X = preprocess(raw, bundle)
    model = bundle["model"]

    approved = int(model.predict(X)[0])
    proba_approved = float(model.predict_proba(X)[0][1])
    confidence = proba_approved if approved == 1 else 1.0 - proba_approved

    return {
        "approved": approved,
        "label": "Approved" if approved == 1 else "Rejected",
        "probability": proba_approved,
        "confidence": confidence,
    }
