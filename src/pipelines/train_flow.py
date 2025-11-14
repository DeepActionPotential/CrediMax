# src/pipelines/train_flow.py

from prefect import flow, task
from omegaconf import OmegaConf
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.data.load_data import load_dataset
from src.data.split import split_data
from src.data.preprocess import feature_engineering, build_preprocessor

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import joblib
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ---------------------------
#       PREFECT TASKS
# ---------------------------

@task
def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    return cfg


@task
def load_and_prepare_data(cfg):
    df = load_dataset(cfg.dataset.path)
    df = feature_engineering(df)
    return df


@task
def prepare_splits(df, cfg):
    return split_data(
        df,
        target_col=cfg.dataset.target,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state,
    )


@task
def build_pipeline(cfg, X_train):
    preproc = build_preprocessor(X_train)

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=cfg.model.n_jobs,
        random_state=cfg.model.random_state,
        **dict(cfg.model.params)
    )

    pipeline = Pipeline([
        ("preproc", preproc),
        ("clf", clf)
    ])

    return pipeline


@task
def train_and_log(pipeline, X_train, y_train, X_test, y_test, cfg):
    """
    Train the model, log metrics/artifacts/models to MLflow, and return run_id + metrics.
    """
    # Set MLflow tracking
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Enable autolog (optional)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=cfg.run.name) as run:
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.eval.threshold).astype(int)

        roc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)
        cm = confusion_matrix(y_test, pred)
        report = classification_report(y_test, pred)

        # Log metrics explicitly (autolog may also capture)
        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("pr_auc", float(pr))

        # Save and log confusion matrix plot
        tmp_dir = Path(tempfile.mkdtemp())
        cm_path = tmp_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Log the pipeline explicitly (artifact path 'pipeline')
        mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")

        # Also save a local copy for convenience
        Path("artifacts").mkdir(exist_ok=True)
        joblib.dump(pipeline, "artifacts/credit_risk_pipeline.joblib")

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    # Return results (useable by downstream tasks)
    return {
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "report": report,
        "run_id": run_id,
        "experiment_id": experiment_id,
    }


@task
def model_gating(results, cfg):
    """
    Compare trained model with current Production model.
    Promote ONLY if new model exceeds thresholds and is better.
    """

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    model_name = cfg.model_registry.model_name

    new_roc = float(results["roc_auc"])
    new_pr = float(results["pr_auc"])

    print("\n=== Model Gating ===")
    print(f"New ROC: {new_roc:.4f}")
    print(f"New PR : {new_pr:.4f}")

    # --- Check basic thresholds ---
    if new_roc < cfg.model_registry.min_roc_auc:
        print(" New model ROC below threshold — rejecting.")
        return False

    if new_pr < cfg.model_registry.min_pr_auc:
        print(" New model PR-AUC below threshold — rejecting.")
        return False

    print("Passed quality thresholds.")

    # --- Check existing production model ---
    try:
        prod_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if not prod_versions:
            print("No production model found → deploying first model.")
            return True

        prod = prod_versions[0]
        prod_tags = prod.tags or {}

        prod_roc = float(prod_tags.get("roc_auc", 0))
        prod_pr = float(prod_tags.get("pr_auc", 0))

        print(f"Production ROC: {prod_roc:.4f}")
        print(f"Production PR : {prod_pr:.4f}")

        if new_roc <= prod_roc:
            print(" New ROC is not better — rejecting.")
            return False

        if new_pr <= prod_pr:
            print(" New PR-AUC is not better — rejecting.")
            return False

        print(" New model outperforms production model.")
        return True

    except Exception as exc:
        # If something goes wrong reading production model, be conservative and approve first model
        print(f"Warning reading production model: {exc}")
        print(" Approving model (no production model or error reading it).")
        return True


@task
def promote_model(cfg, run_id, results):
    """
    Register the model in MLflow registry (creating it if missing)
    and promote the run's model to Production.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)

    model_name = cfg.model_registry.model_name

    # -------------------------------------------
    # 1. Ensure registered model exists
    # -------------------------------------------
    try:
        client.get_registered_model(model_name)
        print(f"Registered model '{model_name}' already exists.")
    except Exception:
        print(f"Registered model '{model_name}' does NOT exist. Creating it...")
        client.create_registered_model(model_name)

    # -------------------------------------------
    # 2. Build correct model artifact source path
    # -------------------------------------------
    exp_id = results.get("experiment_id")

    if exp_id is None:
        exp = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
        if exp is None:
            raise RuntimeError(f"Experiment '{cfg.mlflow.experiment_name}' not found.")
        exp_id = exp.experiment_id

    source = f"mlruns/{exp_id}/{run_id}/artifacts/pipeline"
    print(f"Registering model version from: {source}")

    # -------------------------------------------
    # 3. Create model version
    # -------------------------------------------
    mv = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
    )

    # -------------------------------------------
    # 4. Attach REAL metrics as tags
    # -------------------------------------------
    client.set_model_version_tag(model_name, mv.version, "roc_auc", str(results["roc_auc"]))
    client.set_model_version_tag(model_name, mv.version, "pr_auc", str(results["pr_auc"]))

    # -------------------------------------------
    # 5. Promote to Production
    # -------------------------------------------
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"Model promoted to PRODUCTION. Version: {mv.version}")


# ---------------------------
#       PREFECT FLOW
# ---------------------------

@flow(name="credit-risk-training-flow")
def training_flow(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)

    df = load_and_prepare_data(cfg)

    X_train, X_test, y_train, y_test = prepare_splits(df, cfg)

    pipeline = build_pipeline(cfg, X_train)

    results = train_and_log(
        pipeline,
        X_train, y_train,
        X_test, y_test,
        cfg
    )

    # Gating decision
    should_promote = model_gating(results, cfg)

    if should_promote:
        promote_model(cfg, results["run_id"], results)
    else:
        print("- Model NOT promoted.")

    print("Training done:")
    print("ROC AUC:", results["roc_auc"])
    print("PR AUC:", results["pr_auc"])
    print(results["report"])


if __name__ == "__main__":
    training_flow()
