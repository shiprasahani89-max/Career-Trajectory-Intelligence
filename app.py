"""Streamlit dashboard for Career Trajectory Intelligence."""
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import run_analysis, engineer_features, promotion_gap_category, build_cluster_features

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "employee_hr_dataset.csv"

st.set_page_config(
    page_title="Career Trajectory Intelligence",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Career Trajectory Intelligence")
st.caption("Career progression patterns, promotion gaps, and development opportunities")

st.warning(
    "Dataset source note: the provided file resembles the commonly used IBM HR "
    "Analytics Employee Attrition & Performance dataset. Do not present the records "
    "as Palo Alto Networks employees unless the source is independently verified. "
    "Scores are exploratory indicators, not validated predictions."
)

with st.sidebar:
    st.header("Data and filters")
    uploaded = st.file_uploader("Upload a CSV (optional)", type=["csv"])
    st.caption("If no file is uploaded, the project dataset in data/ is used.")
    cluster_count = st.slider("Number of career clusters", 2, 6, 3)
    gap_threshold = st.slider("Promotion gap ratio threshold", 0.0, 1.0, 0.50, 0.05)
    st.caption("Ratio thresholds are configurable assumptions, not HR standards.")

try:
    if uploaded is not None:
        temp_path = ROOT / "uploaded_dashboard_data.csv"
        temp_path.write_bytes(uploaded.getvalue())
        input_path = temp_path
    else:
        input_path = DEFAULT_DATA

    if not input_path.exists():
        st.error(f"Dataset not found: {input_path}")
        st.stop()

    # Recompute for chosen cluster count. Results are saved under outputs/.
    df, profiles, summary = run_analysis(input_path=input_path, n_clusters=cluster_count)
except Exception as exc:
    st.error(f"Could not process this dataset: {exc}")
    st.info("Check that the CSV includes the required HR and career columns. See README.md.")
    st.stop()

with st.sidebar:
    departments = sorted(df["Department"].dropna().astype(str).unique().tolist())
    selected_departments = st.multiselect("Department", departments, default=departments)
    roles = sorted(df["JobRole"].dropna().astype(str).unique().tolist())
    selected_roles = st.multiselect("Job role", roles, default=roles)
    career_stage = st.selectbox(
        "Career stage",
        ["All", "Early career (0–2 years)", "Developing (3–5 years)", "Established (6+ years)"],
    )
    selected_clusters = st.multiselect(
        "Career cluster",
        sorted(df["CareerCluster"].unique().tolist()),
        default=sorted(df["CareerCluster"].unique().tolist()),
    )

filtered = df[
    df["Department"].astype(str).isin(selected_departments)
    & df["JobRole"].astype(str).isin(selected_roles)
    & df["CareerCluster"].isin(selected_clusters)
].copy()

if career_stage == "Early career (0–2 years)":
    filtered = filtered[filtered["YearsAtCompany"] <= 2]
elif career_stage == "Developing (3–5 years)":
    filtered = filtered[filtered["YearsAtCompany"].between(3, 5)]
elif career_stage == "Established (6+ years)":
    filtered = filtered[filtered["YearsAtCompany"] >= 6]

# Apply user-selected threshold to the promotion gap chart/table.
filtered["AboveSelectedGapThreshold"] = (
    filtered["PromotionGapRatio"] >= gap_threshold
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees in current filters", f"{len(filtered):,}")
c2.metric("High promotion-gap score", f"{(filtered['PromotionGapScore'] == 'High').sum():,}")
c3.metric("Above gap threshold", f"{filtered['AboveSelectedGapThreshold'].sum():,}")
c4.metric("Retention opportunity: High", f"{(filtered['RetentionOpportunityBand'] == 'High').sum():,}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Career Path Clustering",
    "Promotion Gap Monitor",
    "Retention Opportunity",
    "Managerial Insights",
])

with tab1:
    st.subheader("Career Path Clustering Dashboard")
    if filtered.empty:
        st.info("No employees match the selected filters.")
    else:
        cluster_counts = filtered["CareerCluster"].value_counts().sort_index().rename_axis("CareerCluster").reset_index(name="EmployeeCount")
        fig = px.bar(cluster_counts, x="CareerCluster", y="EmployeeCount",
                     title="Employee distribution by cluster", text="EmployeeCount")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Cluster profile averages**")
        profile_filtered = filtered.groupby("CareerCluster")[[
            "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
            "TrainingTimesLastYear", "YearsWithCurrManager",
            "PromotionGapRatio", "RoleStagnationIndex"
        ]].mean(numeric_only=True).round(2)
        st.dataframe(profile_filtered, use_container_width=True)
        st.caption("Cluster numbers are arbitrary. Assign business labels only after reviewing these profiles.")

with tab2:
    st.subheader("Promotion Gap Monitor")
    if not filtered.empty:
        risk_counts = filtered["PromotionGapScore"].value_counts().rename_axis("PromotionGapScore").reset_index(name="Employees")
        fig = px.bar(risk_counts, x="PromotionGapScore", y="Employees",
                     title="Promotion gap score distribution", text="Employees")
        st.plotly_chart(fig, use_container_width=True)
        role_profile = filtered.groupby("JobRole").agg(
            Employees=("JobRole", "size"),
            AvgYearsSincePromotion=("YearsSinceLastPromotion", "mean"),
            AvgYearsInCurrentRole=("YearsInCurrentRole", "mean"),
            HighGapCount=("PromotionGapScore", lambda s: int((s == "High").sum())),
        ).reset_index().round(2)
        st.markdown("**Role-level stagnation indicators**")
        st.dataframe(role_profile, use_container_width=True)
        st.markdown("**Employees above selected promotion gap ratio threshold**")
        st.dataframe(
            filtered.loc[filtered["AboveSelectedGapThreshold"], [
                "Department", "JobRole", "YearsAtCompany", "YearsSinceLastPromotion",
                "YearsInCurrentRole", "PromotionGapRatio", "RoleStagnationIndex",
                "PromotionGapScore", "CareerCluster"
            ]].sort_values("PromotionGapRatio", ascending=False),
            use_container_width=True
        )

with tab3:
    st.subheader("Retention Opportunity Panel")
    st.write(
        "This panel flags employees for voluntary career-development review. "
        "It does not predict an individual's future resignation."
    )
    if not filtered.empty:
        band_counts = filtered["RetentionOpportunityBand"].value_counts().rename_axis("RetentionOpportunityBand").reset_index(name="Employees")
        fig = px.bar(band_counts, x="RetentionOpportunityBand", y="Employees",
                     title="Retention opportunity indicator bands", text="Employees")
        st.plotly_chart(fig, use_container_width=True)
        opp = filtered[
            (filtered["RetentionOpportunityBand"].isin(["Medium", "High"]))
            & (filtered["PromotionGapScore"].isin(["Medium", "High"]))
            & (filtered["Attrition"].astype(str).str.lower().isin(["0", "no", "false"]))
        ].copy()
        st.markdown("**Suggested voluntary development review list**")
        st.dataframe(opp[[
            "Department", "JobRole", "YearsAtCompany", "YearsSinceLastPromotion",
            "YearsInCurrentRole", "TrainingTimesLastYear", "JobSatisfaction",
            "JobInvolvement", "PromotionGapScore", "RetentionOpportunityIndex",
            "RetentionOpportunityBand", "TrainingNeedIndicator"
        ]].sort_values("RetentionOpportunityIndex", ascending=False), use_container_width=True)
        st.markdown("**Possible actions**")
        st.write("- Offer a career conversation and ask about employee goals.")
        st.write("- Review mentoring, training, internal mobility, and promotion criteria.")
        st.write("- Check workload and manager support before recommending a role change.")

with tab4:
    st.subheader("Managerial Insight Dashboard")
    if not filtered.empty:
        manager_plot = px.scatter(
            filtered,
            x="YearsWithCurrManager",
            y="YearsSinceLastPromotion",
            color="Department",
            hover_data=["JobRole", "YearsAtCompany", "CareerCluster"],
            title="Current-manager tenure vs. time since last promotion",
        )
        st.plotly_chart(manager_plot, use_container_width=True)
        manager_profile = filtered.groupby("Department").agg(
            Employees=("Department", "size"),
            AvgYearsWithCurrentManager=("YearsWithCurrManager", "mean"),
            AvgYearsSincePromotion=("YearsSinceLastPromotion", "mean"),
            AvgYearsInCurrentRole=("YearsInCurrentRole", "mean"),
        ).reset_index().round(2)
        st.dataframe(manager_profile, use_container_width=True)
        st.caption(
            "YearsWithCurrManager is only a proxy for continuity; this dataset does not "
            "record the full history or number of manager changes."
        )

st.divider()
st.subheader("Export filtered results")
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered employee analysis (CSV)",
    data=csv_bytes,
    file_name="career_trajectory_filtered_results.csv",
    mime="text/csv",
)
st.caption("For research use, document thresholds, inspect cluster profiles, and validate findings with domain experts.")
