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
    df = pd.read_csv("loan_data.csv").drop_duplicates().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    bounds = {c: (X_train[c].quantile(0.01), X_train[c].quantile(0.99)) for c in FEATURES}

    def clip(frame):
        out = frame.copy()
        for col, (low, high) in bounds.items():
            out[col] = out[col].clip(low, high)
        return out

    X_train, X_test = clip(X_train), clip(X_test)

    model = DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    params = model.get_params()
    tree_info = {
        "max_depth": params["max_depth"],
        "depth": int(model.get_depth()),
        "nodes": int(model.tree_.node_count),
        "leaves": int(model.get_n_leaves()),
        "min_samples_leaf": params["min_samples_leaf"],
    }

    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "bounds": bounds,
        "metrics": metrics,
        "tree_info": tree_info,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join("model", "loan_model.pkl"), "wb") as fh:
        pickle.dump(bundle, fh)

    print("Saved model/loan_model.pkl")
    print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("Tree info:", tree_info)


if __name__ == "__main__":
    main()
