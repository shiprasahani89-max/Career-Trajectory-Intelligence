# Career Trajectory Intelligence

A beginner-friendly Python + Streamlit project for exploring employee career progression patterns, promotion gaps, training participation, manager-tenure proxies, and potential career-development opportunities.

## Important dataset-source note

The supplied `Palo Alto Networks(1).csv` file contains 1,470 rows and 31 columns and resembles the commonly used **IBM HR Analytics Employee Attrition & Performance** dataset. The file itself does not verify that the records belong to Palo Alto Networks. In a report or presentation, describe it as the *provided HR analytics dataset* unless you independently verify the source and permission to use the company name.

This is an exploratory analytics project, not a validated attrition prediction model or a tool for making employment decisions. Use results to support voluntary career conversations, not to rank or penalize employees.

## What is included?

- `data/employee_hr_dataset.csv` — the supplied dataset, copied into the project folder.
- `analysis.py` — feature engineering, preprocessing, K-Means, hierarchical clustering, summary tables, and charts.
- `app.py` — Streamlit dashboard with four modules and interactive filters.
- `requirements.txt` — Python libraries.
- `outputs/` — analysis results are created here after running `analysis.py`.

## 1. Install Python

Use Python 3.10 or newer (Python 3.11 is recommended). During installation on Windows, select **Add Python to PATH** if that option appears.

Open Command Prompt or the VS Code terminal and check:

```bash
python --version
```

## 2. Open the project folder

Extract `Career_Trajectory_Intelligence.zip`. Open the extracted `Career_Trajectory_Intelligence` folder in VS Code.

The folder should look like:

```text
Career_Trajectory_Intelligence/
├── app.py
├── analysis.py
├── requirements.txt
├── README.md
├── data/
│   └── employee_hr_dataset.csv
└── outputs/
```

## 3. Create a virtual environment

In the VS Code terminal, make sure the current directory is the project folder.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt in VS Code or run `.venv\Scripts\python.exe -m pip install -r requirements.txt`.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the Python analysis first

In the project folder terminal:

```bash
python analysis.py
```

The script prints a summary and cluster profile. It creates these files in `outputs/`:

- `employee_career_analysis.csv`
- `cluster_profiles.csv`
- `silhouette_scores.csv`
- `analysis_summary.json`
- `cluster_distribution.png`
- `promotion_gap_distribution.png`

These outputs are calculated from the included CSV when you run the script. Do not write expected cluster labels or results into the report before checking these files.

## 5. Run the Streamlit dashboard

In the same terminal:

```bash
streamlit run app.py
```

A browser tab should open automatically. If it does not, use the local URL printed in the terminal, commonly `http://localhost:8501`.

Dashboard modules:

1. **Career Path Clustering** — cluster counts and career-feature profiles.
2. **Promotion Gap Monitor** — risk-score distribution, role-level indicators, and threshold-based employee list.
3. **Retention Opportunity** — career-development review indicators and suggested actions.
4. **Managerial Insights** — manager tenure compared with promotion gap, plus department-level summaries.

Sidebar controls include department, job role, career stage, cluster selection, cluster count, and promotion-gap ratio threshold. You can optionally upload a compatible CSV.

## Methodology

### Feature engineering

- `PromotionGapRatio = YearsSinceLastPromotion / YearsAtCompany`
- `RoleStagnationIndex = YearsInCurrentRole / YearsAtCompany`
- `TrainingIntensityScore = TrainingTimesLastYear / YearsAtCompany`
- `ManagerContinuityProxy = YearsWithCurrManager / YearsAtCompany`

When `YearsAtCompany` is zero, ratios are undefined. The code retains them as missing values for preprocessing instead of interpreting them as zero risk. The manager feature is a proxy: the dataset does not contain manager-change history.

### Preprocessing

- Remove exact duplicate rows.
- Impute missing numerical feature values with medians and scale numerical features.
- Impute categorical values with the most frequent category and one-hot encode Department and JobRole.
- Do not automatically delete valid long-tenure employees as outliers. Inspect extreme values and correct only data errors.

### Clustering

K-Means is the primary clustering method. Hierarchical clustering is used as a comparison. The script calculates silhouette scores for several candidate cluster counts. The default dashboard starts at four clusters, but the cluster count is configurable. Cluster IDs are arbitrary; name them only after inspecting `cluster_profiles.csv`.

### Promotion Gap Score

The starter thresholds in `analysis.py` are illustrative:
- **High:** Promotion Gap Ratio >= 0.50 AND Role Stagnation Index >= 0.60
- **Medium:** Promotion Gap Ratio >= 0.25 OR Role Stagnation Index >= 0.40
- **Low:** neither medium nor high rule is met
- **Insufficient tenure:** ratio values are undefined because company tenure is zero or missing

These are not established HR standards. Review and validate them with domain experts before drawing conclusions.

### Retention Opportunity Index

The index is a transparent exploratory indicator built from promotion-gap score, lower relative training participation, historical attrition status, and job satisfaction. It is not a future attrition probability. The dashboard's review list is a prompt for supportive, voluntary career conversations, not an automated decision.

## Troubleshooting

### `ModuleNotFoundError`

Activate `.venv` and run:

```bash
python -m pip install -r requirements.txt
```

### `FileNotFoundError`

Make sure you run commands from the extracted project folder and that `data/employee_hr_dataset.csv` exists.

### `OneHotEncoder` does not accept `sparse_output`

Upgrade scikit-learn:

```bash
python -m pip install --upgrade scikit-learn
```

### Dashboard says columns are missing

Use the supplied CSV or upload a CSV containing at least the columns listed in `analysis.py`'s `REQUIRED` variable.

## Suggested research paper structure

1. Abstract
2. Background and problem statement
3. Dataset description and data-source limitation
4. Exploratory data analysis
5. Feature engineering
6. Data preprocessing
7. K-Means and hierarchical clustering methodology
8. Cluster interpretation and promotion-gap scoring
9. Retention opportunity indicator
10. Dashboard implementation
11. Results from generated output files
12. Limitations, fairness, privacy, and ethical use
13. Recommendations and future work
14. Conclusion and references

## Limitations to discuss

- Dataset provenance is not verified by the filename alone.
- A single snapshot does not reveal every employee's promotion timeline or manager changes.
- `YearsSinceLastPromotion` and tenure ratios can be misleading for new employees; zero-tenure cases are treated as undefined.
- Clustering reveals similarity patterns, not causal explanations.
- Thresholds are illustrative and should be validated.
- Historical `Attrition` values do not tell us whether a currently employed person will leave in the future.
- Avoid using protected or sensitive attributes to make employment decisions. Keep employee-level outputs access-controlled and use aggregate insights where possible.
