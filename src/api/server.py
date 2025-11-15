import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(title="Credit Risk Predictor", version="1.0")

# ======================================================
# Load Model
# ======================================================
MODEL_PATH = "artifacts/credit_risk_pipeline.joblib"

try:
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Loaded model successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# ======================================================
# Feature List (NO AUTODETECT)
# ======================================================
EXPECTED_RAW_COLUMNS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

# ======================================================
# Create DataFrame from Request
# ======================================================
def build_input_df(payload):

    df = pd.DataFrame([payload])

    # Add missing raw columns as NaN
    for col in EXPECTED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[EXPECTED_RAW_COLUMNS].copy()

    # Derived features (same as notebook)
    df["log_person_income"] = np.log1p(df["person_income"].clip(lower=0))
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"].clip(lower=0))
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)

    return df

# ======================================================
# ROUTES
# ======================================================

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
async def predict(req: Request):

    try:
        data = await req.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    try:
        df = build_input_df(data)

        proba = pipeline.predict_proba(df)[:, 1]
        pred = (proba >= 0.5).astype(int)

        return {
            "probability": float(proba[0]),
            "prediction": int(pred[0]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# ======================================================
# Static Frontend Routes
# ======================================================
FRONTEND_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("src/api/templates/index.html")
