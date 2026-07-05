"""
Shared inference utilities for the loan-approval Decision Tree model.

Both the local test script (``test_model.py``) and the Flask web app
(``app.py``) import from this module so that the *exact same* preprocessing
that was applied during training is also applied at prediction time.

The trained model is stored as a "bundle" (a dict) inside ``model/loan_model.pkl``:

    {
        "model":            <fitted sklearn DecisionTreeClassifier>,
        "features":         ["Age", "Income", "LoanAmount", "CreditScore"],
        "bounds":           {feature: (low, high), ...},   # winsorisation limits
        "metrics":          {...},                          # test-set scores
        "sklearn_version":  "1.6.1",
        "trained_at":       "2026-07-05T12:00:00",
    }
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd

# Absolute path to model/loan_model.pkl, resolved relative to THIS file so it
# works no matter what directory the app/test script is launched from.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "loan_model.pkl")

# Human-friendly metadata for each input feature. Used by the Flask form and
# for validating/clamping user input to a sensible domain.
FEATURE_INFO = {
    "Age":         {"label": "Age (years)",     "min": 18,  "max": 100,     "step": 1},
    "Income":      {"label": "Annual Income",   "min": 0,   "max": 10_000_000, "step": 1000},
    "LoanAmount":  {"label": "Loan Amount",     "min": 0,   "max": 1_000_000,  "step": 500},
    "CreditScore": {"label": "Credit Score",    "min": 300, "max": 850,     "step": 1},
}


def load_bundle(path: str = MODEL_PATH) -> Dict[str, Any]:
    """Load the pickled model bundle. Raises FileNotFoundError with guidance."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'.\n"
            "Train the model first by running the notebook "
            "'loan_prediction_training.ipynb' (it writes model/loan_model.pkl)."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def preprocess(raw: Dict[str, Any], bundle: Dict[str, Any]) -> pd.DataFrame:
    """
    Turn a dict of raw feature values into a single-row DataFrame that matches
    the training pipeline: correct column order + winsorisation (clipping) using
    the bounds learned from the training data.
    """
    features = bundle["features"]
    bounds = bundle["bounds"]

    # Build the row in the exact column order the model was trained on.
    row = {feat: float(raw[feat]) for feat in features}
    df = pd.DataFrame([row], columns=features)

    # Apply the same outlier clipping that was used at training time.
    for feat, (low, high) in bounds.items():
        df[feat] = df[feat].clip(low, high)

    return df


def predict(raw: Dict[str, Any], bundle: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Predict loan approval for one applicant.

    Parameters
    ----------
    raw : dict
        Keys must include every feature name (Age, Income, LoanAmount, CreditScore).
    bundle : dict, optional
        A pre-loaded model bundle. If None, the bundle is loaded from disk.

    Returns
    -------
    dict with keys: approved (int 0/1), label (str), probability (float 0-1),
    confidence (float 0-1 for the predicted class).
    """
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
        "probability": proba_approved,   # P(Approved)
        "confidence": confidence,        # P(predicted class)
    }
