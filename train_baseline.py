"""Train and evaluate a simple baseline on the synthetic feature dataset.

This is an academic demonstration only. It is not used for live resume scoring.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA)
    y = df["shortlisted"].astype(int)
    categorical = ["job_role"]
    numerical = ["years_exp", "skills_matched", "education_score", "keyword_density"]
    # Exclude predicted_score: it was generated from related features and could leak target-generation information.
    X = df[categorical + numerical]
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numerical),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "purpose": "Synthetic-data educational baseline; not used for production/live resume scoring.",
        "rows_total": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "positive_rate_full_dataset": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction, zero_division=0)),
        "recall": float(recall_score(y_test, prediction, zero_division=0)),
        "f1": float(f1_score(y_test, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, prediction, labels=[0, 1], target_names=["not_shortlisted", "shortlisted"], output_dict=True, zero_division=0),
        "caveat": "The labels and features are synthetic/simulated; random held-out performance does not establish real-world validity, fairness, or suitability for employment decisions. predicted_score is intentionally excluded from model features to reduce target leakage because it is related to the synthetic target-generation process."
    }
    (OUT / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    image = ax.imshow(matrix, cmap="PuBuGn")
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"]); ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("Synthetic baseline — confusion matrix", loc="left", fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12, color="white" if matrix[i,j] > matrix.max() * .55 else "#25253B")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04, label="Count")
    fig.tight_layout(); fig.savefig(OUT / "baseline_confusion_matrix.png", dpi=180); plt.close(fig)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
