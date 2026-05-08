"""
Production-style eligibility engine: rules + ML + fraud + doc classifier.

Usage:
    python eligibility_prod.py train          # train all models
    python eligibility_prod.py predict_one    # run single-case demo
    python eligibility_prod.py batch_score    # score cases from CSV

Dependencies:
    pip install numpy pandas scikit-learn joblib
"""

import sys
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Document:
    doc_type: str
    uploaded: bool
    valid: bool
    issued_on: date | None
    expires_on: date | None
    text: str | None = None  # for classification


@dataclass
class Applicant:
    income_monthly: float
    expenses_monthly: float
    credit_score: int
    disability: bool
    hardship: bool
    hardship_date: date | None
    household_size: int
    ami_annual: float
    program_name: str
    servicer: str
    investor: str
    locality: str
    prior_delinquency_days: int
    doc_completeness_score: float
    documents: List[Document]


# ============================================================
# RULE CONFIGURATION (HUD/Treasury/Servicer/Investor style)
# ============================================================

AMI_BANDS = {
    "low": 0.80,
    "moderate": 1.20,
    "high": 9.99
}

PROGRAM_RULES = {
    "HAF": {
        "max_dti": 0.55,
        "min_credit": 500,
        "max_ami_band": "moderate",
        "require_hardship": True,
        "hardship_within_months": 18,
        "income_docs_max_age_days": 60,
        "allow_disability_override": True,
        "max_delinquency_days": 180
    },
    "HUD_X": {
        "max_dti": 0.45,
        "min_credit": 580,
        "max_ami_band": "low",
        "require_hardship": False,
        "hardship_within_months": None,
        "income_docs_max_age_days": 45,
        "allow_disability_override": False,
        "max_delinquency_days": 120
    }
}

SERVICER_RULES = {
    "ServicerA": {"max_dti": 0.50, "min_credit": 520},
    "ServicerB": {"max_dti": 0.47, "min_credit": 540}
}

INVESTOR_RULES = {
    "FannieMae": {"min_credit": 620},
    "FreddieMac": {"min_credit": 620},
    "PrivateX": {"min_credit": 580}
}

COMBO_RULES = {
    ("ServicerA", "FannieMae"): {
        "max_dti": 0.48,
        "min_credit": 630,
        "extra_docs": ["4506C"]
    },
    ("ServicerB", "PrivateX"): {
        "max_dti": 0.50,
        "min_credit": 560,
        "extra_docs": []
    }
}

REQUIRED_DOCS = {
    "HAF": ["id", "paystub", "bank_statement", "hardship_letter"],
    "HUD_X": ["id", "paystub", "award_letter"]
}


# ============================================================
# RULE ENGINE
# ============================================================

def calc_dti(app: Applicant) -> float:
    if app.income_monthly <= 0:
        return 1.0
    return app.expenses_monthly / app.income_monthly


def ami_band(app: Applicant) -> str:
    ratio = (app.income_monthly * 12) / app.ami_annual
    if ratio <= AMI_BANDS["low"]:
        return "low"
    elif ratio <= AMI_BANDS["moderate"]:
        return "moderate"
    return "high"


def check_documents(app: Applicant, program_rules: dict, combo_rules: dict) -> List[str]:
    issues = []
    today = date.today()

    required = REQUIRED_DOCS.get(app.program_name, [])
    docs_by_type = {d.doc_type: d for d in app.documents}

    for req in required:
        doc = docs_by_type.get(req)
        if not doc:
            issues.append(f"Missing required document: {req}")
            continue
        if not doc.uploaded:
            issues.append(f"Document not uploaded: {req}")
        if not doc.valid:
            issues.append(f"Document invalid: {req}")
        if doc.expires_on and doc.expires_on < today:
            issues.append(f"Document expired: {req}")

    max_age = program_rules.get("income_docs_max_age_days", 60)
    for doc_type in ["paystub", "bank_statement"]:
        doc = docs_by_type.get(doc_type)
        if doc and doc.issued_on:
            age = (today - doc.issued_on).days
            if age > max_age:
                issues.append(f"{doc_type} older than {max_age} days")

    extra = combo_rules.get("extra_docs", [])
    for req in extra:
        if req not in docs_by_type:
            issues.append(f"Missing combo-required document: {req}")

    if app.doc_completeness_score < 0.6:
        issues.append(f"Document completeness score too low: {app.doc_completeness_score:.2f}")

    return issues


def evaluate_rules(app: Applicant) -> dict:
    failed: List[str] = []
    today = date.today()

    program = PROGRAM_RULES.get(app.program_name, {})
    servicer = SERVICER_RULES.get(app.servicer, {})
    investor = INVESTOR_RULES.get(app.investor, {})
    combo = COMBO_RULES.get((app.servicer, app.investor), {})

    dti = calc_dti(app)
    band = ami_band(app)

    # Program rules
    if dti > program.get("max_dti", 1):
        failed.append(f"DTI {dti:.2f} exceeds program max {program['max_dti']}")
    if app.credit_score < program.get("min_credit", 0):
        failed.append(f"Credit score {app.credit_score} below program minimum {program['min_credit']}")
    order = ["low", "moderate", "high"]
    max_band = program.get("max_ami_band", "high")
    if order.index(band) > order.index(max_band):
        failed.append(f"AMI band {band} exceeds program max {max_band}")
    if app.prior_delinquency_days > program.get("max_delinquency_days", 9999):
        failed.append(f"Prior delinquency {app.prior_delinquency_days} days exceeds program max")

    if program.get("require_hardship") and not app.hardship:
        failed.append("Program requires documented hardship")
    if app.hardship and program.get("hardship_within_months"):
        months = program["hardship_within_months"]
        if app.hardship_date and (today - app.hardship_date).days > months * 30:
            failed.append(f"Hardship older than {months} months")

    # Servicer rules
    if dti > servicer.get("max_dti", 1):
        failed.append(f"DTI {dti:.2f} exceeds servicer max {servicer['max_dti']}")
    if app.credit_score < servicer.get("min_credit", 0):
        failed.append(f"Credit score {app.credit_score} below servicer minimum {servicer['min_credit']}")

    # Investor rules
    if app.credit_score < investor.get("min_credit", 0):
        failed.append(f"Credit score {app.credit_score} below investor minimum {investor['min_credit']}")

    # Combo rules
    if combo:
        if dti > combo.get("max_dti", 1):
            failed.append(f"DTI {dti:.2f} exceeds combo max {combo['max_dti']}")
        if app.credit_score < combo.get("min_credit", 0):
            failed.append(f"Credit score {app.credit_score} below combo minimum {combo['min_credit']}")

    # Disability override
    if program.get("allow_disability_override") and app.disability:
        failed = [r for r in failed if "Credit score" not in r]

    doc_issues = check_documents(app, program, combo)
    eligible = len(failed) == 0 and len(doc_issues) == 0

    return {
        "eligible": eligible,
        "failed_rules": failed,
        "doc_issues": doc_issues,
        "dti": dti,
        "ami_band": band
    }


# ============================================================
# ML TRAINING: ELIGIBILITY + FRAUD
# ============================================================

def load_eligibility_training_data(path: str = "eligibility_training.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_eligibility_and_fraud_models():
    df = load_eligibility_training_data()

    feature_cols = [
        "income_monthly",
        "expenses_monthly",
        "credit_score",
        "hardship",
        "disability",
        "household_size",
        "ami_annual",
        "prior_delinquency_days",
        "doc_completeness_score"
    ]
    X = df[feature_cols].values
    y = df["approved"].values
    fraud_y = df["fraud_flag"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

    y_pred_lr = lr.predict(X_val)
    y_pred_rf = rf.predict(X_val)
    print("Eligibility LR accuracy:", accuracy_score(y_val, y_pred_lr))
    print("Eligibility RF accuracy:", accuracy_score(y_val, y_pred_rf))

    iso = IsolationForest(contamination=0.1, random_state=42).fit(X)
    fraud_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, fraud_y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(lr, "models/elig_lr.pkl")
    joblib.dump(rf, "models/elig_rf.pkl")
    joblib.dump(iso, "models/fraud_iso.pkl")
    joblib.dump(fraud_rf, "models/fraud_rf.pkl")
    print("Saved eligibility + fraud models.")


def load_eligibility_and_fraud_models():
    lr = joblib.load("models/elig_lr.pkl")
    rf = joblib.load("models/elig_rf.pkl")
    iso = joblib.load("models/fraud_iso.pkl")
    fraud_rf = joblib.load("models/fraud_rf.pkl")
    return lr, rf, iso, fraud_rf


def ml_features_from_app(app: Applicant) -> np.ndarray:
    return np.array([[
        app.income_monthly,
        app.expenses_monthly,
        app.credit_score,
        1 if app.hardship else 0,
        1 if app.disability else 0,
        app.household_size,
        app.ami_annual,
        app.prior_delinquency_days,
        app.doc_completeness_score
    ]])


# ============================================================
# DOCUMENT CLASSIFIER MODEL
# ============================================================

def load_doc_training_data(path: str = "documents_training.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def train_document_classifier():
    df = load_doc_training_data()
    X_text = df["text"].values
    y = df["doc_type"].values

    vectorizer = TfidfVectorizer(max_features=2000)
    X_vec = vectorizer.fit_transform(X_text)

    clf = LogisticRegression(max_iter=1000).fit(X_vec, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/doc_vec.pkl")
    joblib.dump(clf, "models/doc_clf.pkl")
    print("Saved document classifier.")


def load_document_classifier():
    vec = joblib.load("models/doc_vec.pkl")
    clf = joblib.load("models/doc_clf.pkl")
    return vec, clf


def classify_document_text(text: str, vec, clf) -> str:
    X = vec.transform([text])
    return clf.predict(X)[0]


# ============================================================
# TRAIN ALL
# ============================================================

def train_all():
    print("Training eligibility + fraud models...")
    train_eligibility_and_fraud_models()
    print("Training document classifier...")
    train_document_classifier()
    print("✅ All models trained.")


# ============================================================
# PREDICT ONE (DEMO)
# ============================================================

def predict_one_demo():
    lr, rf, iso, fraud_rf = load_eligibility_and_fraud_models()
    vec, doc_clf = load_document_classifier()

    today = date.today()

    docs = [
        Document("id", True, True, today - timedelta(days=10), today + timedelta(days=365),
                 text="Driver license for John Doe exp 2028"),
        Document("paystub", True, True, today - timedelta(days=20), None,
                 text="Pay stub for John Doe covering 01/01 to 01/15"),
        Document("bank_statement", True, True, today - timedelta(days=40), None,
                 text="Bank of America statement December 2024"),
        Document("hardship_letter", True, True, today - timedelta(days=5), None,
                 text="Hardship letter explaining job loss"),
        Document("4506C", True, True, today - timedelta(days=5), None,
                 text="IRS 4506-C form signed")
    ]

    # (Optional) reclassify doc types from text
    for d in docs:
        if d.text:
            predicted_type = classify_document_text(d.text, vec, doc_clf)
            # you could compare predicted_type vs d.doc_type for consistency

    app = Applicant(
        income_monthly=2800,
        expenses_monthly=1500,
        credit_score=610,
        disability=False,
        hardship=True,
        hardship_date=today - timedelta(days=200),
        household_size=3,
        ami_annual=45000,
        program_name="HAF",
        servicer="ServicerA",
        investor="FannieMae",
        locality="TX-Harris",
        prior_delinquency_days=30,
        doc_completeness_score=0.9,
        documents=docs
    )

    rules = evaluate_rules(app)
    feats = ml_features_from_app(app)

    lr_score = float(lr.predict_proba(feats)[0, 1])
    rf_score = float(rf.predict_proba(feats)[0, 1])
    iso_flag = iso.predict(feats)[0] == -1
    fraud_prob = float(fraud_rf.predict_proba(feats)[0, 1])

    result = {
        "eligible_by_rules": rules["eligible"],
        "failed_rules": rules["failed_rules"],
        "doc_issues": rules["doc_issues"],
        "dti": rules["dti"],
        "ami_band": rules["ami_band"],
        "ml_lr_approval_prob": lr_score,
        "ml_rf_approval_prob": rf_score,
        "fraud_anomaly_flag": iso_flag,
        "fraud_probability": fraud_prob
    }

    from pprint import pprint
    pprint(result)


# ============================================================
# BATCH SCORING
# ============================================================

def batch_score(input_csv: str = "eligibility_batch.csv", output_csv: str = "eligibility_scored.csv"):
    lr, rf, iso, fraud_rf = load_eligibility_and_fraud_models()

    df = pd.read_csv(input_csv)

    feature_cols = [
        "income_monthly",
        "expenses_monthly",
        "credit_score",
        "hardship",
        "disability",
        "household_size",
        "ami_annual",
        "prior_delinquency_days",
        "doc_completeness_score"
    ]
    X = df[feature_cols].values

    lr_scores = lr.predict_proba(X)[:, 1]
    rf_scores = rf.predict_proba(X)[:, 1]
    iso_flags = iso.predict(X) == -1
    fraud_probs = fraud_rf.predict_proba(X)[:, 1]

    df["ml_lr_approval_prob"] = lr_scores
    df["ml_rf_approval_prob"] = rf_scores
    df["fraud_anomaly_flag"] = iso_flags
    df["fraud_probability"] = fraud_probs

    df.to_csv(output_csv, index=False)
    print(f"Saved batch scores to {output_csv}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eligibility_prod.py [train|predict_one|batch_score]")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        train_all()
    elif mode == "predict_one":
        predict_one_demo()
    elif mode == "batch_score":
        if len(sys.argv) >= 3:
            batch_score(sys.argv[2])
        else:
            batch_score()
    else:
        print("Unknown mode. Use train, predict_one, or batch_score.")
        sys.exit(1)
