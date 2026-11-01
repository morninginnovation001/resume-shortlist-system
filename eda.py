"""Exploratory data analysis and schema validation for the synthetic CSV."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
EXPECTED = ["resume_id", "job_role", "years_exp", "skills_matched", "education_score", "keyword_density", "predicted_score", "shortlisted"]


def main() -> None:
    df = pd.read_csv(DATA)
    issues: list[str] = []
    if list(df.columns) != EXPECTED:
        issues.append(f"Column schema mismatch. Expected {EXPECTED}; found {list(df.columns)}")
    if df["resume_id"].duplicated().any():
        issues.append("resume_id contains duplicates")
    if df.isna().any().any():
        issues.append("Dataset contains missing cells")
    bounds = {
        "years_exp": (0, 15), "skills_matched": (0, 30),
        "education_score": (0, 100), "keyword_density": (0, 1),
        "predicted_score": (0, 100), "shortlisted": (0, 1),
    }
    for col, (low, high) in bounds.items():
        if not df[col].between(low, high).all():
            issues.append(f"{col} contains values outside [{low}, {high}]")
    if not set(df["shortlisted"].unique()).issubset({0, 1}):
        issues.append("shortlisted must be binary 0/1")

    purple, teal, grey = "#5C2D91", "#118D83", "#8B8D9A"
    # Figure 1: role distribution
    counts = df["job_role"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.barh(counts.index.str.replace("_", " "), counts.values, color=purple)
    ax.set_title("Synthetic records by job role", loc="left", fontweight="bold")
    ax.set_xlabel("Number of records")
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "eda_role_distribution.png", dpi=180); plt.close(fig)
    # Figure 2: label balance
    label_counts = df["shortlisted"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    bars = ax.bar(["Not shortlisted (0)", "Shortlisted (1)"], label_counts.values, color=[grey, teal])
    ax.set_title("Synthetic shortlist-label balance", loc="left", fontweight="bold")
    ax.set_ylabel("Number of records")
    ax.bar_label(bars, padding=3); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "eda_shortlist_balance.png", dpi=180); plt.close(fig)
    # Figure 3: predicted score distributions by target
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    for label, color, name in [(0, grey, "Not shortlisted"), (1, teal, "Shortlisted")]:
        values = df.loc[df.shortlisted == label, "predicted_score"]
        ax.hist(values, bins=20, density=True, alpha=.58, color=color, label=f"{name} (n={len(values):,})")
    ax.set_title("Simulated score distribution by label", loc="left", fontweight="bold")
    ax.set_xlabel("Simulated score (0–100)"); ax.set_ylabel("Density"); ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True); ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "eda_score_by_label.png", dpi=180); plt.close(fig)
    # Figure 4: numeric correlations
    numeric = ["years_exp", "skills_matched", "education_score", "keyword_density", "predicted_score", "shortlisted"]
    corr = df[numeric].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr.values, cmap="PuBuGn", vmin=-1, vmax=1, aspect="auto")
    labels = [x.replace("_", "\n") for x in corr.columns]
    ax.set_xticks(range(len(labels)), labels, fontsize=8); ax.set_yticks(range(len(labels)), [x.replace("_", " ") for x in corr.index], fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > .58 else "#25253B")
    ax.set_title("Numeric feature correlation matrix", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.04, pad=.04, label="Pearson correlation")
    fig.tight_layout(); fig.savefig(OUT / "eda_correlation.png", dpi=180); plt.close(fig)

    summary = {
        "rows": int(len(df)), "columns": int(len(df.columns)), "column_names": list(df.columns),
        "job_roles": int(df["job_role"].nunique()), "duplicate_resume_ids": int(df["resume_id"].duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()), "shortlisted_count": int(df.shortlisted.sum()),
        "shortlisted_rate": float(df.shortlisted.mean()), "mean_predicted_score": float(df.predicted_score.mean()),
        "role_counts": {k: int(v) for k, v in df.job_role.value_counts().sort_index().items()},
        "range_checks": {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in bounds},
        "validation_issues": issues,
        "note": "Synthetic feature data. Visual patterns and summary statistics are not evidence about real hiring outcomes or fairness."
    }
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if issues:
        raise SystemExit("Dataset validation failed; inspect outputs/eda_summary.json")


if __name__ == "__main__":
    main()
