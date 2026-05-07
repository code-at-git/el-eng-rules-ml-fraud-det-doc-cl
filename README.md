# el-eng-rules-ml-fraud-det-doc-cl
# 📘 **Eligibility Engine — Rules + Machine Learning + Fraud Detection + Document Classification**

A fully local, production‑grade eligibility determination:

- **HUD/Treasury/Servicer/Investor rule engine**
- **Machine learning eligibility prediction**
- **Fraud detection model**
- **Document classifier model**
- **Batch scoring**
- **Single‑case scoring**
- **Training + validation pipeline**
- **Explainable outputs**

This system is designed for **mortgage assistance**, **HAF**, **HUD programs**, **servicer/investor‑driven eligibility**, and any workflow requiring strict rule compliance combined with predictive intelligence.

Everything runs **locally**, requires **no cloud**, and uses **open‑source** components.

---

# 🚀 Features

### ✔ Full Rule Engine (HUD/Treasury/Servicer/Investor)
- AMI band calculation (low/moderate/high)
- DTI calculation + thresholds
- Credit score thresholds
- Hardship requirements + recency rules
- Disability override logic
- Required document validation
- Document expiration + age rules
- Servicer‑specific rules
- Investor‑specific rules
- Servicer + Investor combination rules
- Prior delinquency rules
- Document completeness scoring

### ✔ Machine Learning Models
- **Eligibility prediction** (Logistic Regression + Random Forest)
- **Fraud detection** (Isolation Forest + Fraud RF model)
- **Document classifier** (TF‑IDF + Logistic Regression)

### ✔ Training Pipeline
- Train all models with one command
- Automatic validation split
- Accuracy reporting
- Model persistence (`./models/*.pkl`)

### ✔ Prediction Modes
- **Single‑case prediction** (rule engine + ML + fraud + doc classifier)
- **Batch scoring** from CSV
- Full explainability (failed rules, doc issues, ML scores)

### ✔ 100% Local
- No API keys  
- No cloud  
- No external services  

---

# 📂 Project Structure

```
eligibility_prod.py
eligibility_training.csv
documents_training.csv
eligibility_batch.csv (optional)
models/
    elig_lr.pkl
    elig_rf.pkl
    fraud_iso.pkl
    fraud_rf.pkl
    doc_vec.pkl
    doc_clf.pkl
```

---

# 📄 Training Data Templates

## **eligibility_training.csv**
Each row = **one borrower/case**.

```csv
income_monthly,expenses_monthly,credit_score,hardship,disability,household_size,ami_annual,program_name,servicer,investor,locality,prior_delinquency_days,doc_completeness_score,approved,fraud_flag
3000,1200,640,1,0,3,45000,HAF,ServicerA,FannieMae,TX-Harris,0,0.9,1,0
2200,1500,580,1,1,2,40000,HAF,ServicerA,FannieMae,TX-Harris,45,0.7,1,0
1800,1400,540,0,0,1,38000,HUD_X,ServicerB,PrivateX,TX-Dallas,90,0.4,0,0
5000,1800,720,0,0,4,60000,HAF,ServicerA,FannieMae,TX-Harris,0,1.0,1,0
2600,1700,560,1,0,2,42000,HUD_X,ServicerB,PrivateX,TX-Dallas,60,0.5,0,1
```

## **documents_training.csv**
Used to train the document classifier.

```csv
text,doc_type
"Pay stub for John Doe covering 01/01 to 01/15",paystub
"Bank of America statement December 2024",bank_statement
"Driver license for Jane Doe exp 2028",id
"Hardship letter explaining job loss",hardship_letter
"IRS 4506-C form signed",4506C
```

---

# 🧠 How the System Works

## 1. **Rule Engine (Deterministic Eligibility)**
The rule engine enforces:

- HUD/Treasury program rules  
- Servicer rules  
- Investor rules  
- Combination rules  
- Document requirements  
- Hardship rules  
- AMI rules  
- DTI rules  
- Credit score rules  
- Disability overrides  
- Delinquency rules  

If any rule fails → **ineligible**.

The rule engine is the **source of truth** for compliance.

---

## 2. **Machine Learning (Predictive Intelligence)**

ML does **not** determine eligibility.  
ML provides:

- Approval probability  
- Risk scoring  
- Fraud probability  
- Anomaly detection  
- Case prioritization  

Models:

- Logistic Regression (eligibility)
- Random Forest (eligibility)
- Isolation Forest (fraud anomaly)
- Random Forest (fraud probability)
- TF‑IDF + Logistic Regression (document classifier)

---

## 3. **Document Classifier**
Automatically identifies document types from text:

- Paystub  
- Bank statement  
- ID  
- Hardship letter  
- 4506‑C  

Useful for:

- Auto‑tagging documents  
- Detecting mismatches  
- Document completeness scoring  

---

## 4. **Fraud Detection**
Two models:

- Isolation Forest → anomaly detection  
- Random Forest → fraud probability  

Features include:

- Income vs expenses  
- Credit score  
- Hardship  
- Disability  
- Delinquency  
- Document completeness  

---

# 🛠 Installation

```bash
pip install numpy pandas scikit-learn joblib
```

---

# 🏋️ Training All Models

```bash
python eligibility_prod.py train
```

This will:

- Train eligibility models  
- Train fraud models  
- Train document classifier  
- Save all models to `./models/`  

---

# 🔍 Single Case Prediction

```bash
python eligibility_prod.py predict_one
```

Outputs:

- Rule eligibility (True/False)
- Failed rules
- Document issues
- DTI
- AMI band
- ML approval probability (LR + RF)
- Fraud anomaly flag
- Fraud probability

---

# 📊 Batch Scoring

Prepare a CSV:

```csv
income_monthly,expenses_monthly,credit_score,hardship,disability,household_size,ami_annual,prior_delinquency_days,doc_completeness_score
3000,1200,640,1,0,3,45000,0,0.9
2200,1500,580,1,1,2,40000,45,0.7
```

Run:

```bash
python eligibility_prod.py batch_score eligibility_batch.csv
```

Outputs:

```
eligibility_scored.csv
```

With:

- ML approval probabilities  
- Fraud anomaly flag  
- Fraud probability  

---

# 🧩 Architecture Overview

```
┌──────────────────────────┐
│        Applicant         │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│      Rule Engine         │
│ HUD/Treasury/Servicer    │
│ Investor/Combo/Docs/AMI  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     ML Eligibility       │
│ Logistic + RandomForest  │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     Fraud Detection      │
│ IsolationForest + RF     │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│   Document Classifier    │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│     Final Verdict        │
│  + Explanation           │
└──────────────────────────┘
```

---

# 🧾 Output Example

```json
{
  "eligible_by_rules": false,
  "failed_rules": [
    "DTI 0.54 exceeds servicer max 0.5",
    "Credit score 610 below investor minimum 620",
    "DTI 0.54 exceeds combo max 0.48",
    "Credit score 610 below combo minimum 630"
  ],
  "doc_issues": [],
  "dti": 0.5357,
  "ami_band": "low",
  "ml_lr_approval_prob": 0.98,
  "ml_rf_approval_prob": 0.97,
  "fraud_anomaly_flag": false,
  "fraud_probability": 0.12
}
```
