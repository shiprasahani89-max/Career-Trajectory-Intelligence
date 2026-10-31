"""Career Trajectory Intelligence: reproducible analysis pipeline.

Important:
- The supplied CSV resembles the commonly used IBM HR Analytics dataset.
- Do not claim the rows are Palo Alto Networks employees unless the source is verified.
- This is exploratory analysis, not a validated prediction or employment decision system.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "employee_hr_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CAREER_NUMERIC = [
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "TrainingTimesLastYear",
    "YearsWithCurrManager",
]
CATEGORICAL = ["Department", "JobRole"]
REQUIRED = CAREER_NUMERIC + CATEGORICAL + [
    "Attrition", "JobSatisfaction", "JobInvolvement"
]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def engineer_features(df):
    """Create ratios while retaining missing/undefined values for zero tenure."""
    result = df.copy()
    tenure = result["YearsAtCompany"].replace(0, np.nan)

    result["PromotionGapRatio"] = (
        result["YearsSinceLastPromotion"] / tenure
    )
    result["RoleStagnationIndex"] = (
        result["YearsInCurrentRole"] / tenure
    )
    result["TrainingIntensityScore"] = (
        result["TrainingTimesLastYear"] / tenure
    )
    # Proxy only: time with current manager relative to company tenure.
    # This is not a direct count of manager changes.
    result["ManagerContinuityProxy"] = (
        result["YearsWithCurrManager"] / tenure
    )
    result["ZeroCompanyTenureFlag"] = (
        result["YearsAtCompany"].eq(0).astype(int)
    )
    return result


def promotion_gap_category(row):
    """Transparent starter thresholds; validate with HR/domain experts."""
    gap = row["PromotionGapRatio"]
    stagnation = row["RoleStagnationIndex"]
    if pd.isna(gap) or pd.isna(stagnation):
        return "Insufficient tenure"
    if gap >= 0.50 and stagnation >= 0.60:
        return "High"
    if gap >= 0.25 or stagnation >= 0.40:
        return "Medium"
    return "Low"


def build_cluster_features(df):
    numeric = [
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "TrainingTimesLastYear",
        "YearsWithCurrManager",
        "PromotionGapRatio",
        "RoleStagnationIndex",
        "TrainingIntensityScore",
        "ManagerContinuityProxy",
    ]
    categorical = CATEGORICAL

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    matrix = transformer.fit_transform(df)
    return matrix, transformer


def run_analysis(input_path=DATA_PATH, n_clusters=None):
    df = load_data(input_path)
    original_column_count = int(df.shape[1])
    initial_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    # Keep valid long-tenure employees. Flag and inspect extreme values rather
    # than deleting records automatically.
    df = df.drop_duplicates().copy()
    df = engineer_features(df)

    # Assign the transparent rule-based score separately from unsupervised clusters.
    df["PromotionGapScore"] = df.apply(promotion_gap_category, axis=1)

    matrix, _ = build_cluster_features(df)
    max_k = min(6, len(df) - 1)
    silhouette_results = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        silhouette_results.append({
            "k": k,
            "silhouette_score": float(silhouette_score(matrix, labels)),
        })

    # Choose k by silhouette score among the tested values. This is a starting
    # selection criterion, not proof that the clusters are business-valid.
    best_k = max(silhouette_results, key=lambda x: x["silhouette_score"])["k"]
    selected_k = best_k if n_clusters is None else max(2, min(int(n_clusters), len(df) - 1))
    # Keep user's requested/default k for a predictable, explainable baseline;
    # record silhouette results so k can be reviewed.
    km = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    df["CareerCluster"] = km.fit_predict(matrix)

    hierarchical = AgglomerativeClustering(n_clusters=selected_k)
    df["HierarchicalCluster"] = hierarchical.fit_predict(matrix)

    # Starter labels are assigned only after inspecting cluster profiles.
    profile_cols = [
        "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "TrainingTimesLastYear", "YearsWithCurrManager",
        "PromotionGapRatio", "RoleStagnationIndex",
        "TrainingIntensityScore", "ManagerContinuityProxy"
    ]
    cluster_profile = df.groupby("CareerCluster")[profile_cols].mean().round(3)
    cluster_profile["EmployeeCount"] = df.groupby("CareerCluster").size()
    cluster_profile["AttritionRatePct"] = (
        df.groupby("CareerCluster")["Attrition"].mean() * 100
    ).round(2)
    cluster_profile = cluster_profile.reset_index()

    # Retention opportunity is a triage aid, not a prediction. "Attrition=0"
    # is historical status in this dataset, not a future risk label.
    df["RetentionOpportunityIndex"] = 0
    df.loc[df["PromotionGapScore"].eq("High"), "RetentionOpportunityIndex"] += 2
    df.loc[df["PromotionGapScore"].eq("Medium"), "RetentionOpportunityIndex"] += 1
    df.loc[df["TrainingIntensityScore"].fillna(0) < df["TrainingIntensityScore"].median(),
           "RetentionOpportunityIndex"] += 1
    df.loc[df["Attrition"].astype(str).str.lower().isin(["0", "no", "false"]),
           "RetentionOpportunityIndex"] += 1
    df.loc[df["JobSatisfaction"] >= 3, "RetentionOpportunityIndex"] += 1
    df["RetentionOpportunityBand"] = pd.cut(
        df["RetentionOpportunityIndex"],
        bins=[-1, 1, 3, 5],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # Training need is an indicator for discussion, not a performance judgement.
    training_median = df["TrainingTimesLastYear"].median()
    df["TrainingNeedIndicator"] = np.where(
        df["TrainingTimesLastYear"] < training_median,
        "Review training needs",
        "No low-training flag",
    )

    # Do not use protected attributes such as age or gender for clustering/risk.
    df.to_csv(OUTPUT_DIR / "employee_career_analysis.csv", index=False)
    cluster_profile.to_csv(OUTPUT_DIR / "cluster_profiles.csv", index=False)
    pd.DataFrame(silhouette_results).to_csv(
        OUTPUT_DIR / "silhouette_scores.csv", index=False
    )

    attrition_value = pd.to_numeric(df["Attrition"], errors="coerce")
    summary = {
        "input_file": Path(input_path).name,
        "rows_before_deduplication": initial_rows,
        "rows_after_deduplication": int(len(df)),
        "duplicate_rows_found": duplicate_rows,
        "missing_cells_in_original_data": missing_cells,
        "columns": original_column_count,
        "attrition_yes_or_one_count": int((attrition_value == 1).sum()),
        "attrition_rate_pct_if_binary_0_1": round(float((attrition_value == 1).mean() * 100), 2),
        "selected_cluster_count": int(selected_k),
        "best_silhouette_k_among_tested": int(best_k),
        "silhouette_results": silhouette_results,
        "note": "Cluster labels are arbitrary numeric IDs; interpret from cluster_profiles.csv.",
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save charts generated from the actual data.
    plt.figure(figsize=(8, 5))
    df["CareerCluster"].value_counts().sort_index().plot(kind="bar")
    plt.title("Employee Distribution by Career Cluster")
    plt.xlabel("Career Cluster")
    plt.ylabel("Employee Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    order = ["Low", "Medium", "High", "Insufficient tenure"]
    df["PromotionGapScore"].value_counts().reindex(order, fill_value=0).plot(kind="bar")
    plt.title("Promotion Gap Score Distribution")
    plt.xlabel("Promotion Gap Score")
    plt.ylabel("Employee Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "promotion_gap_distribution.png", dpi=160)
    plt.close()

    return df, cluster_profile, summary


if __name__ == "__main__":
    result_df, profile, summary = run_analysis()
    print("\nAnalysis completed.")
    print(json.dumps(summary, indent=2))
    print("\nCluster profiles:")
    print(profile.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
