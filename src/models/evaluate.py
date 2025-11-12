# src/models/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score

sns.set(style="whitegrid")

def evaluate_model(pipeline, X_test, y_test, out_dir):
    """
    Compute metrics, save confusion matrix and returns a dict of key metrics.
    """
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr  = average_precision_score(y_test, proba)
    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred)

    # Save confusion matrix figure
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    metrics = {"roc_auc": float(roc), "pr_auc": float(pr), "report": report}
    return metrics
