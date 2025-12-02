# **Credit Risk Prediction System — End-to-End MLOps Pipeline**

This project implements a full **production-grade MLOps system** for credit-risk prediction, combining **scikit-learn**, **FastAPI**, **Docker**, **GitHub Actions**, and **AWS EC2**.
Unlike typical ML projects, this repository demonstrates **a fully automated model lifecycle** — from data preprocessing and training to deployment, monitoring readiness, and automated retraining.

The goal is to showcase real-world MLOps engineering skills, not just model training.

---

# **Table of Contents**

1. Summary
2. Project Overview
3. Business Value
4. Architecture
5. ML Pipeline Internals
6. MLOps Workflow
7. Demo
8. Run Locally
9. Deployment (CI/CD + EC2)
10. API Usage
11. Future Improvements

---

# **Summary**

This system predicts credit default risk and is deployed as a **containerized, scalable, and automated ML service**.
The project demonstrates all pillars of modern MLOps:

* Reproducible **training pipeline**
* Clean **feature engineering & preprocessing**
* Full-pipeline **artifact packaging**
* **Dockerized** inference service
* Automated **CI/CD deployment** to AWS EC2
* **Scheduled retraining** every week
* **Infrastructure-independent** inference via FastAPI
* MLflow logging for experiment tracking
* Version-controlled model artifacts

This is not a demo — it is a **realistic ML production workflow** ready for enterprise use.

---

# **Project Overview**

The system predicts **probability of default** for loan applicants using features such as:

* Age
* Employment length
* Income
* Loan amount & percent of income
* Credit history length
* Home ownership
* Loan purpose
* Loan grade
* Previous defaults

Included components:

- **Training pipeline** (datasets → preprocessing → feature engineering → XGBoost/RandomForest → metrics → artifact)

- **Web UI + API** using FastAPI

- **Docker images** for training & inference

- **CI/CD workflows** for testing & deployment

- **Retraining pipeline** triggered weekly

- **Static HTML/JS front-end** for user predictions

- **Artifact storage** & versioning

- **EC2 deployment** (secure, isolated Linux server)


---



# **Business Value**

Banks and lenders lose money when they approve loans for people who can’t repay them. This system helps reduce those losses by giving a fast, consistent estimate of how risky a borrower is.

### **What this system improves**

**• Better decisions:**
It catches high-risk applications early, helping lenders avoid unnecessary defaults.

**• Fairer approvals:**
Good borrowers are less likely to be rejected because the model judges risk more accurately than manual review.

**• Faster processing:**
Loan decisions that normally take hours can now be made in seconds, improving customer experience and increasing throughput.

**• Always up-to-date:**
Weekly automated retraining ensures the model adapts to new borrower trends and economic conditions.

In short: the system helps lenders save money, approve more good customers, and make decisions faster and more consistently.

---

# **Architecture**

Below is the simplified architecture of the system:

```
                        (Weekly Retraining)
  ┌──────────────┐         GitHub Actions
  │   Training    │  -----------------------------→ New Model Artifact (.joblib)
  │   Pipeline    │
  └───────▲──────┘                                           │
          │                                                   │
          │                                          Loaded at Startup
          │                                                   ▼
  ┌──────────────┐     CI/CD Deployment         ┌────────────────────────────┐
  │   GitHub      │    ----------------------→  │   FastAPI Inference Server │
  │   Actions     │                              │    (Docker on EC2)        │
  └───────▲──────┘                              └───────────┬────────────────┘
          │                                                  │
          │                                                  ▼
  Local Dev → Commit → Push                           Browser UI or API Client
```

### **Major components**

* **Training container**
  Performs preprocessing, training, evaluation, and artifact generation.

* **Inference container**
  Loads the packaged pipeline and serves predictions via REST.

* **EC2 host**
  Runs the inference service using Docker Compose.

* **GitHub Actions pipelines:**

  * `test.yml` → CI
  * `deploy.yml` → CD
  * `weekly_train.yml` → Scheduled retraining

---

# **ML Pipeline Internals**

### **1. Preprocessing**

All preprocessing is done inside a Scikit-Learn `ColumnTransformer`, ensuring **no training-serving skew**:

* **Numerical features**

  * Median imputation
  * Standard scaling
  * Log transforms for skew
  * Ratio features (loan-to-income, etc.)

* **Categorical features**

  * Most-frequent imputation
  * OneHotEncoding with `handle_unknown="ignore"`

### **2. Model**

The model is configurable (XGBoost, Logistic Regression, RandomForest).
XGBoost is currently used due to:

* Strong performance on tabular data
* Robust handling of non-linear structure
* Good interpretability via SHAP (future improvement)

### **3. Evaluation Metrics**

* ROC-AUC
* PR-AUC
* Precision/Recall
* Confusion matrix
* Thresholded prediction metrics
* Test-set probability calibration

### **4. Artifact Packaging**

The full pipeline (preprocessing + model) is saved as:

```
artifacts/credit_risk_pipeline.joblib
```

### **Why full pipeline packaging matters:**

* Guarantees identical transformations at inference
* No risk of mismatching preprocessing logic
* Ensures versioning & reproducibility
* Makes deployment environment-agnostic

### **5. Inference (FastAPI)**

At startup:

```python
model = joblib.load("artifacts/credit_risk_pipeline.joblib")
```

A prediction takes one HTTP call and ~2–5 ms of inference time.

---

# **MLOps Workflow**

This project includes three automated workflows using GitHub Actions:

---

## **1. Continuous Integration — `test.yml`**

* Installs dependencies
* Runs basic tests / lint
* Ensures code stability before deployment

---

## **2. Continuous Deployment — `deploy.yml`**

Triggered on push to `main`:

1. Copies updated code to EC2 using SCP
2. Rebuilds Docker images
3. Restarts containers
4. Ensures no manual SSH interaction is needed

This provides **hands-free deployment** from GitHub to production.

---

## **3. Automated Weekly Retraining — `weekly_train.yml`**

Runs every Sunday at 3 AM (Cairo time):

1. Builds training Docker image
2. Runs full training pipeline
3. Logs experiment metrics to MLflow
4. Saves new model artifact
5. Uploads artifact back to GitHub

This ensures the model:

* Stays fresh
* Adapts to dataset drift
* Always has an up-to-date version ready for deployment

---

# **Demo**

![Demo](docs/demo.png)

### Video Demo

```html
<video controls width="700">
  <source src="docs/demo.mp4" type="video/mp4">
</video>
```

---

# **Run Locally**

### Docker

```bash
docker compose up -d --build
```

Visit:

```
http://localhost:8000/
```

### Run FastAPI directly (dev mode)

```bash
uvicorn credit_risk_mlops.api.server:app --host 0.0.0.0 --port 8000 --reload
```

---

# **Deployment (CI/CD + EC2)**

CI/CD pipeline:

### **test.yml**

* Basic CI checks

### **deploy.yml**

* Triggered on push
* Syncs code to EC2
* Rebuilds/starts inference container

### **weekly_train.yml**

* Full retraining
* Uploads new artifact
* Enables model versioning

Together these form a complete **train → version → deploy → retrain** MLOps lifecycle.

---

# **API Usage**

### Predict

**POST:** `/predict`

```json
{
  "person_age": 33,
  "person_income": 54000,
  "person_emp_length": 6,
  "loan_int_rate": 12.5,
  "loan_amnt": 9000,
  "loan_percent_income": 0.16,
  "loan_intent": "PERSONAL",
  "person_home_ownership": "RENT",
  "loan_grade": "B",
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**Response**

```json
{
  "default_probability": 0.1482,
  "prediction": 0
}
```

---

# **Future Improvements**

* Monitoring system (inference latency, prediction drift)
* Automated data validation (Great Expectations)
* Model registry (MLflow Models)
* Canary or blue-green model deployment
* SHAP-based explainability for regulators
* Upgrading EC2 to ECS/Fargate for autoscaling
* Feature store integration (Feast)
