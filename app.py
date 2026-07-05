from flask import Flask, render_template, request

from predict_utils import FEATURE_INFO, load_bundle, predict

app = Flask(__name__)

try:
    BUNDLE = load_bundle()
    MODEL_ERROR = None
except FileNotFoundError as exc:
    BUNDLE = None
    MODEL_ERROR = str(exc)


def validate_form(form):
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
        values[name] = int(num) if num == int(num) else num
    return values, errors


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    errors = []
    values = {"Age": 30, "Income": 95000, "LoanAmount": 8000, "CreditScore": 700}

    if request.method == "POST":
        if BUNDLE is None:
            errors = [MODEL_ERROR]
        else:
            values, errors = validate_form(request.form)
            if not errors:
                result = predict(values, bundle=BUNDLE)

    metrics = BUNDLE["metrics"] if BUNDLE else None
    tree_info = BUNDLE.get("tree_info") if BUNDLE else None
    return render_template(
        "index.html",
        feature_info=FEATURE_INFO,
        values=values,
        result=result,
        errors=errors,
        metrics=metrics,
        tree_info=tree_info,
    )


@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
