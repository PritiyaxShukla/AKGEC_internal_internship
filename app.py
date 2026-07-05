"""
Flask web app that serves the loan-approval Decision Tree model.

Run:
    python app.py
Then open http://127.0.0.1:5000 in a browser.

The app renders an HTML form (templates/index.html). On submit it validates the
inputs, runs them through `predict_utils.predict` (same preprocessing as training),
and re-renders the page with the decision and the model's confidence.
"""

from flask import Flask, render_template, request

from predict_utils import FEATURE_INFO, load_bundle, predict

app = Flask(__name__)

# Load the model once, at start-up, so every request is fast.
# If the model file is missing we keep the app running and show a helpful error.
try:
    BUNDLE = load_bundle()
    MODEL_ERROR = None
except FileNotFoundError as exc:
    BUNDLE = None
    MODEL_ERROR = str(exc)


def validate_form(form):
    """
    Validate raw form values. Returns (values_dict, errors_list).
    `values_dict` contains parsed floats for valid fields (str for invalid ones,
    so the form can be re-populated with what the user typed).
    """
    values, errors = {}, []
    for name, info in FEATURE_INFO.items():
        raw = (form.get(name) or "").strip()
        if raw == "":
            errors.append(f"{info['label']} is required.")
            values[name] = ""
            continue
        try:
            num = float(raw)
        except ValueError:
            errors.append(f"{info['label']} must be a number.")
            values[name] = raw
            continue
        if num < info["min"] or num > info["max"]:
            errors.append(f"{info['label']} must be between {info['min']:,} and {info['max']:,}.")
        # Display whole numbers without a trailing ".0" when re-populating the form.
        values[name] = int(num) if num == int(num) else num
    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    # Sensible defaults so the form is pre-filled on first load.
    values = {"Age": 30, "Income": 95000, "LoanAmount": 8000, "CreditScore": 700}

    if request.method == "POST":
        if BUNDLE is None:
            errors = [MODEL_ERROR]
        else:
            values, errors = validate_form(request.form)
            if not errors:
                result = predict(values, bundle=BUNDLE)

    metrics = BUNDLE["metrics"] if BUNDLE else None
    return render_template(
        "index.html",
        feature_info=FEATURE_INFO,
        values=values,
        result=result,
        errors=errors,
        metrics=metrics,
    )


@app.route("/health")
def health():
    """Simple health/readiness endpoint."""
    return {"status": "ok", "model_loaded": BUNDLE is not None}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
