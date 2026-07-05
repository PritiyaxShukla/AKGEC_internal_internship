# Loan Approval Prediction - Decision Tree

An end-to-end machine-learning demonstration: **data preprocessing -> model
training -> evaluation -> deployment**. A `DecisionTreeClassifier` (scikit-learn)
predicts whether a loan application is **Approved (1)** or **Rejected (0)**, and a
**Flask** web app serves the model behind an HTML form.

---

## Project structure

```
AKGEC_internal_internship_/
- loan_data.csv                     # dataset (45,000 rows)
- loan_prediction_training.ipynb    # 1) preprocessing + training + evaluation notebook
- model/
  - loan_model.pkl                  #    trained model bundle (created by the notebook)
- predict_utils.py                  # shared loading + preprocessing + prediction helpers
- test_model.py                     # 2) local command-line testing
- app.py                            # 3) Flask web app
- templates/
  - index.html                      #    HTML form + result page
- static/
  - style.css                       #    styling
- requirements.txt
- README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Core libraries: pandas, numpy, scikit-learn, Flask
(plus jupyter/matplotlib/seaborn to run the notebook).

---

## How to run (3 steps)

### 1. Train the model (Jupyter notebook)

Open and **run all cells** in `loan_prediction_training.ipynb`. It performs the
full workflow and writes the trained model to `model/loan_model.pkl`.

```bash
jupyter notebook loan_prediction_training.ipynb
```

### 2. Test locally (command line)

```bash
python test_model.py                                          # runs sample applicants
python test_model.py --age 30 --income 95000 --loan 8000 --credit 700   # custom input
```

### 3. Run the web app (Flask)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** and fill in the form.

---

## Data preprocessing (what the notebook does and why)

The dataset has four numeric features (`Age`, `Income`, `LoanAmount`,
`CreditScore`) and a binary target (`Approved`). Preprocessing decisions come from
the EDA:

| Step | Reason |
|---|---|
| **Drop duplicate rows** | Duplicates bias the model and leak across train/test. |
| **Stratified train/test split (80/20)** | The classes are imbalanced (~78% rejected / ~22% approved); stratifying preserves that ratio. |
| **Winsorisation (clip to 1st-99th percentile)** | The raw data has impossible values (`Age` up to 144) and extreme incomes (millions). Bounds are learned from the **training set only** (no leakage) and saved with the model so the *same* clipping is applied to new inputs at serving time. |
| **No feature scaling** | Decision trees split on per-feature thresholds and are invariant to scaling, so standardisation would add deployment complexity for zero benefit. |
| **`class_weight="balanced"`** | Makes the rare *approved* class count more so the tree does not just predict "reject" for everyone. |

### Model

`DecisionTreeClassifier(max_depth=7, min_samples_leaf=50, class_weight="balanced",
random_state=42)`

### Test-set performance

| Metric | Score |
|---|---|
| Accuracy | ~0.78 |
| Precision | ~0.50 |
| Recall | ~0.65 |
| F1 | ~0.56 |
| ROC-AUC | ~0.80 |

> Because the data is imbalanced, **accuracy alone is misleading** - a model that
> always predicts "reject" already scores ~78%. That is exactly why we also report
> precision, recall, F1 and ROC-AUC.

---

## How training and serving stay consistent

`predict_utils.py` is imported by **both** `test_model.py` and `app.py`. It loads
the pickled bundle - which stores the fitted model **plus** the feature order and
the winsorisation bounds - and re-applies the identical preprocessing to every new
applicant. There is no way for training and serving preprocessing to drift apart.

---

## Disclaimer

This is an **educational demo**. `loan_data.csv` is a synthetic dataset whose
label patterns are not real-world credit logic (e.g. in this data lower incomes are
associated with *higher* approval rates and credit score has almost no effect). The
model faithfully learns those dataset patterns, so its predictions **must not** be
used for any real lending decision.
