# src/models/register.py
import mlflow

def register_model(run_id: str, artifact_path: str = "pipeline", model_name: str = "credit_risk_model"):
    """
    Register a model artifact from a run in the MLflow Model Registry.
    - run_id: the MLflow run id that contains the model artifact
    - artifact_path: where the model was logged (e.g., "pipeline")
    - model_name: desired registry name
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result = mlflow.register_model(model_uri, model_name)
    print(f"Requested registration of {model_uri} as {model_name}. Registered model version: {result.version}")
    return result
