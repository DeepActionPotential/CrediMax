# src/models/train.py
import os
import tempfile
from pathlib import Path

import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
import joblib

# local imports (from your data module)
from credit_risk_mlops.data.load_data import load_dataset
from credit_risk_mlops.data.split import split_data
from credit_risk_mlops.data.preprocess import feature_engineering, build_preprocessor

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Train entrypoint.
    Run from project root:
      poetry run python -m credit_risk_mlops.models.train
    Hydra will pick configs/config.yaml (which should import data/model/mlflow blocks).
    """
    # 1) Prepare MLflow
    mlflow_uri = cfg.mlflow.tracking_uri
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name if "mlflow" in cfg else "credit-risk-experiment")

    # Optionally enable autologging for sklearn/xgboost
    mlflow.sklearn.autolog()  # logs params, metrics, model for sklearn; for XGBoost internal booster, sklearn wrapper works fine

    # 2) Load & prepare data
    data_path = cfg.dataset.path
    df = load_dataset(data_path)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_col=cfg.dataset.target,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state
    )

    # 3) Build preprocessing + model pipeline
    preprocessor = build_preprocessor(X_train)

    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=cfg.model.n_jobs if "model" in cfg else -1,
        random_state=cfg.model.random_state if "model" in cfg else 42,
        **(dict(cfg.model.params) if ("model" in cfg and "params" in cfg.model) else {})
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("clf", clf)
    ])

    # 4) Fit and evaluate inside MLflow run
    with mlflow.start_run(run_name=cfg.run.name if "run" in cfg and "name" in cfg.run else None) as run:
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.eval.threshold) if ("eval" in cfg and "threshold" in cfg.eval) else (proba >= 0.5)

        roc = roc_auc_score(y_test, proba)
        pr  = average_precision_score(y_test, proba)
        report = classification_report(y_test, pred, output_dict=False)
        cm = confusion_matrix(y_test, pred)

        # Log custom metrics
        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("pr_auc", float(pr))

        # Save and log confusion matrix plot
        tmp_dir = Path(tempfile.mkdtemp())
        cm_path = tmp_dir / "confusion_matrix.png"
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion matrix")
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Save ROC / PR curves too (optional)
        try:
            from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
            fig_roc, axroc = plt.subplots(figsize=(6,4))
            RocCurveDisplay.from_predictions(y_test, proba, ax=axroc)
            fig_roc.savefig(tmp_dir / "roc_curve.png", bbox_inches="tight")
            plt.close(fig_roc)
            mlflow.log_artifact(str(tmp_dir / "roc_curve.png"), artifact_path="plots")

            fig_pr, axpr = plt.subplots(figsize=(6,4))
            PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axpr)
            fig_pr.savefig(tmp_dir / "pr_curve.png", bbox_inches="tight")
            plt.close(fig_pr)
            mlflow.log_artifact(str(tmp_dir / "pr_curve.png"), artifact_path="plots")
        except Exception:
            pass

        # Optionally print classification report to console
        print("=== Classification Report ===")
        print(report)

        # 5) Log final pipeline as an MLflow artifact (sklearn)
        # MLflow autologging may already save the model; we save the pipeline explicitly too.
        artifact_path = "pipeline"
        mlflow.sklearn.log_model(pipeline, artifact_path)

        # Also save a local copy for quick local use (joblib)
        local_model_path = Path("artifacts")
        local_model_path.mkdir(exist_ok=True)
        joblib.dump(pipeline, local_model_path / "credit_risk_pipeline.joblib")
        print(f"Saved local pipeline to {local_model_path / 'credit_risk_pipeline.joblib'}")

    print(f"Run finished. MLflow run_id: {run.info.run_id}")


if __name__ == "__main__":
    main()
