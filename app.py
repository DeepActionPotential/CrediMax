# src/api/server.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import joblib
import pandas as pd
import numpy as np
import os
import mlflow
import traceback

app = FastAPI(title="Credit Risk Predictor", version="0.1")

###
# CONFIG - update if autodetection fails
###
MODEL_PATH = "artifacts/credit_risk_pipeline.joblib"
# If autodetect can't find raw columns, set them here (names must match training CSV column names)
EXPECTED_RAW_COLUMNS = [
    # put the real dataset column names here (excluding 'loan_status')
    # e.g. "loan_amnt", "loan_int_rate", "loan_grade", "loan_intent", "person_home_ownership",
    # "person_income", "loan_percent_income", ...
]
###
# load model
###
try:
    print(f"Loading pipeline from {MODEL_PATH} ...")
    pipeline = joblib.load(MODEL_PATH)
    print("Pipeline loaded.")
except Exception as e:
    # If you also registered the model in MLflow registry, you can attempt to load it as:
    # try:
    #     model_uri = "models:/credit-risk-model/Production"
    #     pipeline = mlflow.pyfunc.load_model(model_uri)
    # except Exception:
    #     raise
    raise RuntimeError(f"Failed to load pipeline from '{MODEL_PATH}': {e}")

def extract_raw_input_columns_from_pipeline(pipe) -> Optional[List[str]]:
    """
    Try to extract the raw input column names used at training time by inspecting
    a fitted ColumnTransformer inside the pipeline (named 'prep' in your pipeline).
    Returns list or None if not found.
    """
    try:
        # The pipeline in your notebook: Pipeline([('prep', preprocess), ('clf', clf)])
        preproc = pipe.named_steps.get("prep", None)
        if preproc is None:
            return None

        cols = []
        # ColumnTransformer stores transformers_ attribute as list of (name, transformer, columns)
        # but in some sklearn versions it's transformers (without _). Use both names.
        transformers = getattr(preproc, "transformers_", None) or getattr(preproc, "transformers", None)
        if transformers is None:
            return None

        for name, transformer, cols_in in transformers:
            # cols_in might be slice, list, numpy array or 'remainder'
            if cols_in == "remainder":
                continue
            if isinstance(cols_in, (list, tuple, np.ndarray)):
                cols.extend(list(cols_in))
            else:
                # sometimes columns stored as np.array or pandas Index
                try:
                    cols.extend(list(cols_in))
                except Exception:
                    # ignore if not iterable
                    pass

        # Deduplicate and return
        cols = list(dict.fromkeys(cols))
        return cols if cols else None
    except Exception:
        return None

RAW_COLUMNS = extract_raw_input_columns_from_pipeline(pipeline) or EXPECTED_RAW_COLUMNS
if not RAW_COLUMNS:
    raise RuntimeError(
        "Could not autodetect raw input columns from pipeline and EXPECTED_RAW_COLUMNS is empty. "
        "Update EXPECTED_RAW_COLUMNS with your dataset column names (exclude target 'loan_status')."
    )

print("Using raw input columns:", RAW_COLUMNS)


def build_dataframe_from_payload(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Build a DataFrame with the raw columns expected by the pipeline.
    Accepts either a single dict (one record) or a list of dicts (batch).
    Missing keys are set to NaN.
    """

    if isinstance(payload, dict):
        rows = [payload]
    else:
        rows = payload

    # create DataFrame with RAW_COLUMNS as columns to guarantee same column order
    df = pd.DataFrame(rows)
    # ensure all expected columns exist
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # keep only RAW_COLUMNS (avoid stray keys)
    df = df[RAW_COLUMNS].copy()

    # Derived features (match notebook)
    # log_person_income, log_loan_amnt, loan_to_income
    if "person_income" in df.columns:
        # avoid negative or null values
        df["log_person_income"] = np.log1p(df["person_income"].clip(lower=0).astype(float))
    if "loan_amnt" in df.columns:
        df["log_loan_amnt"] = np.log1p(df["loan_amnt"].clip(lower=0).astype(float))
    if {"loan_amnt", "person_income"}.issubset(df.columns):
        # loan_to_income as in notebook - careful with division by zero
        df["loan_to_income"] = df["loan_amnt"].replace(0, np.nan) / df["person_income"].replace(0, np.nan)

    # After derived features are added, the pipeline expects numeric + categorical lists
    # If your ColumnTransformer was built using these derived column names, we're good.
    return df


class PredictRequest(BaseModel):
    # accept free-form dict or list of dicts - validation below
    records: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(pipeline is not None), "raw_columns_count": len(RAW_COLUMNS)}


@app.post("/predict")
async def predict(req: Request):
    """
    Accepts JSON body with either:
    - a single record: { "loan_amnt": 10000, "person_income": 5000, ... }
    - a batch: [ {...}, {...} ]
    Or wrapped in {"records": ...} as per PredictRequest.
    Returns probabilities and class labels.
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # support {"records": ...} wrapper
    if isinstance(body, dict) and "records" in body:
        payload = body["records"]
    else:
        payload = body

    # allow either dict (single row) or list (batch)
    if not (isinstance(payload, dict) or isinstance(payload, list)):
        raise HTTPException(status_code=400, detail="Payload must be a dict (single record) or list (batch)")

    try:
        df = build_dataframe_from_payload(payload)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error building dataframe from payload: {e}")

    # Optionally: run basic type conversions (numbers)
    # NOTE: Pipeline's SimpleImputer will handle missing values

    # Predict
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)[:, 1]
        else:
            # If pipeline is a pyfunc style model, use predict and interpret accordingly
            preds = pipeline.predict(df)
            # if preds are probabilities:
            if isinstance(preds, np.ndarray) and preds.ndim == 1 and ((preds >= 0).all() and (preds <= 1).all()):
                proba = preds
            else:
                # fallback: run predict -> binary 0/1; convert to probs 0/1
                proba = np.array(preds).astype(float)

        pred_label = (proba >= 0.5).astype(int)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # format response to match input length
    results = []
    for p, lab in zip(proba.tolist(), pred_label.tolist()):
        results.append({"probability": float(p), "prediction": int(lab)})

    # if single record input, return single object
    if isinstance(payload, dict):
        return {"result": results[0], "raw_columns_used": RAW_COLUMNS}
    else:
        return {"results": results, "raw_columns_used": RAW_COLUMNS}


# Example usage note printed in server logs
print("API ready. POST /predict with JSON body (single record dict or list of dicts).")
print("Example single record payload:")
print({col: "<value>" for col in RAW_COLUMNS[:8]})
