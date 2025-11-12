# src/models/predict.py
from pathlib import Path
import joblib
import mlflow.pyfunc

def load_local_pipeline(local_path: str = "artifacts/credit_risk_pipeline.joblib"):
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Local pipeline not found at {p.resolve()}")
    return joblib.load(p)

def load_mlflow_model(model_uri: str):
    """
    Load a model logged to MLflow.
    Example model_uri: "runs:/<run_id>/pipeline" (if logged with artifact_path "pipeline")
    or "models:/MyModel/Production" for registry.
    """
    return mlflow.pyfunc.load_model(model_uri)

def predict_df(pipeline, X_df):
    """
    Returns a dict with probabilities and labels.
    """
    proba = pipeline.predict_proba(X_df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"proba": proba, "pred": pred}
