import argparse
import sys

from predict_utils import load_bundle, predict


SAMPLE_APPLICANTS = [
    {"Age": 30, "Income": 95000,  "LoanAmount": 8000,  "CreditScore": 700},
    {"Age": 22, "Income": 13000,  "LoanAmount": 30000, "CreditScore": 520},
    {"Age": 45, "Income": 250000, "LoanAmount": 15000, "CreditScore": 780},
    {"Age": 26, "Income": 40000,  "LoanAmount": 20000, "CreditScore": 610},
    {"Age": 24, "Income": 70000,  "LoanAmount": 35000, "CreditScore": 650},
]


def print_model_info(bundle):
    print("=" * 70)
    print("MODEL INFO")
    print("=" * 70)
    print(f"  Trained at      : {bundle.get('trained_at', 'n/a')}")
    print(f"  scikit-learn    : {bundle.get('sklearn_version', 'n/a')}")
    print(f"  Features        : {', '.join(bundle['features'])}")
    metrics = bundle.get("metrics", {})
    if metrics:
        print("  Test metrics    : " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    print()


def run_batch(bundle):
    print("=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    header = f"{'Age':>4} {'Income':>10} {'Loan':>8} {'Credit':>7} | {'Decision':>9} {'P(approve)':>11}"
    print(header)
    print("-" * len(header))
    for applicant in SAMPLE_APPLICANTS:
        result = predict(applicant, bundle=bundle)
        print(
            f"{applicant['Age']:>4} {applicant['Income']:>10,} {applicant['LoanAmount']:>8,} "
            f"{applicant['CreditScore']:>7} | {result['label']:>9} {result['probability']:>11.2%}"
        )
    print()


def run_single(bundle, args):
    applicant = {
        "Age": args.age,
        "Income": args.income,
        "LoanAmount": args.loan,
        "CreditScore": args.credit,
    }
    result = predict(applicant, bundle=bundle)
    print("=" * 70)
    print("CUSTOM APPLICANT")
    print("=" * 70)
    for key, value in applicant.items():
        print(f"  {key:<12}: {value:,}")
    print("-" * 70)
    print(f"  Decision     : {result['label']}")
    print(f"  P(approve)   : {result['probability']:.2%}")
    print(f"  Confidence   : {result['confidence']:.2%}")
    print()


def build_parser():
    p = argparse.ArgumentParser(description="Test the loan approval Decision Tree model.")
    p.add_argument("--age", type=float, help="Applicant age in years")
    p.add_argument("--income", type=float, help="Annual income")
    p.add_argument("--loan", type=float, help="Requested loan amount")
    p.add_argument("--credit", type=float, help="Credit score (300-850)")
    return p


def main():
    args = build_parser().parse_args()

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print_model_info(bundle)

    single_args = [args.age, args.income, args.loan, args.credit]
    if any(v is not None for v in single_args):
        if any(v is None for v in single_args):
            print("ERROR: to test a custom applicant provide all of "
                  "--age --income --loan --credit.", file=sys.stderr)
            sys.exit(2)
        run_single(bundle, args)
    else:
        run_batch(bundle)


if __name__ == "__main__":
    main()
