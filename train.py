"""
Standalone training script - produces model/loan_model.pkl.

This is the script version of loan_prediction_training.ipynb (same preprocessing
and model), with no Jupyter/plotting dependencies, so it can run on a server to
(re)generate the model bundle.

Run:
    python train.py
"""

import os
import pickle
from datetime import datetime

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

RANDOM_STATE = 42
FEATURES = ["Age", "Income", "LoanAmount", "CreditScore"]
TARGET = "Approved"


def main():
    # 1. Load + drop duplicates
    df = pd.read_csv("loan_data.csv").drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]

    # 2. Stratified train/test split (data is imbalanced ~78/22)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Winsorise: learn 1st-99th percentile bounds from the TRAIN set only
    bounds = {c: (X_train[c].quantile(0.01), X_train[c].quantile(0.99)) for c in FEATURES}

    def clip(frame):
        out = frame.copy()
        for col, (low, high) in bounds.items():
            out[col] = out[col].clip(low, high)
        return out

    X_train, X_test = clip(X_train), clip(X_test)

    # 4. Train the Decision Tree (class_weight balances the rare 'approved' class)
    model = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # 6. Save the bundle (model + preprocessing bounds + metadata)
    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "bounds": bounds,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join("model", "loan_model.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)

    print("Saved model/loan_model.pkl")
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
