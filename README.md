<!-- BANNER -->
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</div>

<br/>

<h1 align="center">🛒 Online Retail Customer Segmentation</h1>
<h3 align="center">RFM Feature Engineering + KMeans Clustering in Python</h3>

A reproducible customer-segmentation project for the Online Retail II transaction dataset. The canonical workflow is a **Python CLI** (`main.py`) backed by reusable modules in `src/customer_segmentation/`. The Jupyter notebook is retained only for historical exploration and is **not** the production workflow.

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
- [Running the Analysis](#-running-the-analysis)
- [Methodology](#-methodology)
  - [Data Cleaning](#1-data-cleaning)
  - [How KMeans Clustering Works](#2-how-kmeans-clustering-works)
  - [RFM Feature Engineering](#3-rfm-feature-engineering)
  - [Clustering & Evaluation](#4-clustering--evaluation)
  - [Preprocessing & Algorithm Comparison](#5-preprocessing--algorithm-comparison)
  - [Outlier Handling](#6-outlier-handling)
- [Current Measured Results](#-current-measured-results)
- [Generated Outputs](#-generated-outputs)
- [Tests](#-tests)
- [Architecture](#-architecture)
- [Known Limitations](#-known-limitations)
- [Project History](#-project-history)
- [Tech Stack](#-tech-stack)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🔍 Overview

This project applies **KMeans clustering** to a real-world UK online retail dataset to uncover customer segments. Raw transactional data is cleaned under an explicit, documented policy, converted into customer-level **RFM (Recency, Frequency, Monetary)** features, and clustered, with diagnostics and machine-readable outputs generated at every step.

> The project is based on a real client engagement and was originally inspired by a published whitepaper on customer mining. It has since been restructured from a single exploratory notebook into a tested, reproducible CLI pipeline.

The current run evaluates KMeans cluster counts on customer-level RFM features. The strongest silhouette result is **K=2**, but that partition is **extremely imbalanced** (4,271 vs 14 customers). Treat it as a *statistical baseline*, not as a validated final segmentation strategy or a prediction of future customer behavior.

**Key Learning Outcomes:**
- Loading and validating large Excel-based datasets with `pandas`
- Applying a transparent, regex-based data-cleaning policy with data-quality tracking
- Engineering domain-specific features (RFM) for unsupervised learning
- Evaluating candidate cluster counts with inertia, silhouette scores, and cluster-balance diagnostics
- Comparing preprocessing strategies and clustering algorithms
- Profiling clusters and interpreting results honestly, including their limitations
- Structuring a data-science project as an installable, tested Python package with a CLI

---

## 📦 Dataset

| Property        | Details                                                                  |
|-----------------|--------------------------------------------------------------------------|
| **Name**        | Online Retail II                                                         |
| **Source**      | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| **License**     | CC BY 4.0 (as listed by UCI)                                             |
| **Coverage**    | UK-based online retail transactions                                      |
| **Sheet Used**  | First worksheet (2009–2010); cleaned data spans 2009-12-01 to 2010-12-09 |
| **Raw Records** | 525,461 rows × 8 columns                                                 |
| **Currency**    | British Pounds Sterling (£)                                              |

### Dataset Columns

| Column         | Type     | Description                                              |
|----------------|----------|----------------------------------------------------------|
| `Invoice`      | String   | 6-digit invoice number; prefix `C` = cancellation, `A` = accounting adjustment |
| `StockCode`    | String   | 5-digit product code (may include letter suffix)         |
| `Description`  | String   | Product name/description                                 |
| `Quantity`     | Integer  | Units purchased per transaction line                     |
| `InvoiceDate`  | Datetime | Date and time of invoice                                 |
| `Price`        | Float    | Unit price in GBP (£)                                    |
| `Customer ID`  | Float    | Unique customer identifier (nullable)                    |
| `Country`      | String   | Country of customer                                      |

### Raw Data Quality Findings

| Issue                          | Count   |
|--------------------------------|--------:|
| Missing `Customer ID`          | 107,927 |
| Negative quantities            | 12,326  |
| Negative prices                | 3       |
| Zero prices                    | 3,687   |
| Non-6-digit invoices           | 10,209  |
| ↳ Cancellation invoices (`C`)  | 10,206  |
| ↳ Accounting invoices (`A`)    | 3       |
| Invalid stock codes            | 3,083   |

> These counts overlap (e.g. a cancellation also has a negative quantity), so they should **not** be summed to infer total rows removed. The same counts are exported to `outputs/data_quality_summary.csv` on every run.

---

## 🗂 Project Structure

```text
K-mean/
├── main.py                          # Canonical CLI entry point
├── pyproject.toml
├── uv.lock
├── .python-version
├── README.md
├── data/
│   └── README.md                    # Dataset source, attribution, setup
├── notebooks/
│   ├── online_retail_data_clustering.ipynb   # Historical exploration
│   └── legacy_output.png                     # Old 5-cluster plot (not from the CLI)
├── outputs/
│   ├── algorithm_comparison.csv
│   ├── cluster_profiles.csv
│   ├── data_quality_summary.csv
│   ├── k_evaluation.csv
│   ├── preprocessing_comparison.csv
│   ├── segmentation_summary.json
│   └── diagnostics/
│       ├── elbow_curve.png
│       └── silhouette_curve.png
├── src/
│   └── customer_segmentation/
│       ├── __init__.py
│       ├── clustering.py
│       ├── config.py
│       ├── data_loader.py
│       ├── reporting.py
│       └── rfm.py
└── tests/
    └── test_customer_segmentation.py
```

The source workbook (`data/online_retail_II.xlsx`) is intentionally not listed because it is a local input excluded from version control. See [data/README.md](data/README.md) for the official source, attribution, and setup instructions.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.12 or higher**
- [uv](https://docs.astral.sh/uv/) package manager

The `.python-version` file selects the preferred interpreter for uv when available, and `uv.lock` pins the resolved dependency versions.

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/trentpark8800/online-retail-data-clustering.git
cd online-retail-data-clustering
```

**2. Install dependencies (including dev tools such as pytest):**

```bash
uv sync --extra dev
```

### Dataset Setup

Download the workbook from the [UCI Online Retail II page](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and save it as:

```text
data/online_retail_II.xlsx
```

The dataset is listed by UCI under CC BY 4.0. Citation and local setup details are in [data/README.md](data/README.md). The application **never downloads or modifies** the raw workbook.

---

## ▶️ Running the Analysis

From the repository root:

```bash
uv run python main.py --data-path data/online_retail_II.xlsx --max-k 8
```

The CLI also accepts an explicit RFM reference date and output directory:

```bash
uv run python main.py --data-path data/online_retail_II.xlsx --reference-date 2010-12-10 --max-k 8 --output-dir outputs
```

| Argument           | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `--data-path`      | Path to the Online Retail II workbook                                   |
| `--max-k`          | Largest K evaluated (candidates run from 2 to `max-k`)                  |
| `--reference-date` | Fixed ISO date used for Recency (default: latest valid invoice timestamp) |
| `--output-dir`     | Where reports and diagnostics are written (default: `outputs/`)         |

Relative input and output paths resolve from the **project root**, independent of the current working directory. If the dataset is missing, the CLI reports the missing path.

> **Windows PowerShell:** the commands above work as-is. For multi-line commands use a backtick (`` ` ``) instead of `\` for line continuation.

---

## 🔬 Methodology

### 1. Data Cleaning

The loader reads the first worksheet and validates the expected transaction columns. For modeling, the cleaning policy:

- removes records with missing `Customer ID` or unparseable invoice dates
- retains six-digit numeric invoices (`^\d{6}$`), excluding cancellations (`C...`) and accounting adjustments (`A...`)
- retains stock codes matching `^\d{5}$`, `^\d{5}[a-zA-Z]+$`, or `PADS`
- requires **positive** unit price **and** positive quantity
- calculates each line's sales as `SalesLineTotal = Price × Quantity`

```python
# Sketch of the policy (see src/customer_segmentation/data_loader.py)
mask = (
    df['Invoice'].astype(str).str.match(r'^\d{6}$') &
    (
        df['StockCode'].astype(str).str.match(r'^\d{5}$') |
        df['StockCode'].astype(str).str.match(r'^\d{5}[a-zA-Z]+$') |
        df['StockCode'].astype(str).str.match(r'^PADS$')
    ) &
    df['Customer ID'].notna() &
    (df['Price'] > 0) &
    (df['Quantity'] > 0)
)
```

**Data retention**

```text
Raw records       : 525,461
Cleaned records   : 406,309
Data retained     : ~77.3%
Data dropped      : ~22.7%
```

Returns, cancelled invoices, zero/negative prices, invalid stock codes, and rows without customer IDs are **excluded from the modeling data only** — nothing is deleted from the source workbook. This policy targets positive purchase behavior and is not a complete accounting or returns analysis.

Duplicate source rows are *measured* in the quality report but **not** automatically removed, because repeated transaction lines may be legitimate and the dataset does not define a safe deduplication key.

### 2. How KMeans Clustering Works

KMeans is an **unsupervised ML algorithm** that partitions data into `K` clusters by iteratively:

1. **Initializing** `K` centroids in feature space
2. **Assigning** each data point to the nearest centroid (Euclidean distance)
3. **Recomputing** each centroid as the mean of its cluster
4. **Repeating** steps 2–3 until centroids converge

> Analogy: think of the **Urgency/Importance Matrix** used in time management — tasks are grouped into quadrants based on two features. KMeans does this automatically across N-dimensional feature space.

**Why KMeans for this problem?**
- No labeled customer data is available → an unsupervised approach is required
- Scales to hundreds of thousands of records
- Interpretable cluster centers allow actionable business insights
- Well suited to the low-dimensional, numeric RFM feature space

### 3. RFM Feature Engineering

Transactions that survive cleaning are grouped by customer:

| Feature            | Definition                                                                     |
|--------------------|--------------------------------------------------------------------------------|
| **R** – Recency    | Whole days between a customer's most recent retained purchase and the reference date |
| **F** – Frequency  | Number of distinct retained invoice numbers                                    |
| **M** – MonetaryValue | Sum of retained positive sales-line totals (GBP)                            |

```python
rfm = cleaned_df.groupby('Customer ID', as_index=False).agg(
    MonetaryValue=('SalesLineTotal', 'sum'),
    Frequency=('Invoice', 'nunique'),
    LastPurchaseDate=('InvoiceDate', 'max'),
)
rfm['Recency'] = (reference_date - rfm['LastPurchaseDate']).dt.days
```

By default the reference date is the latest valid invoice timestamp in the cleaned data. Pass `--reference-date` for a fixed ISO date when you need a repeatable business reporting period. Recency is clipped at zero if the chosen reference date predates a transaction.

**Why RFM?**
- Proven framework for customer value segmentation in retail analytics
- Directly interpretable: high F + high M + low R = high-value, active customer
- Reference: [Mining Online Retail Data (Whitepaper)](https://link.springer.com/article/10.1057/dbm.2012.17)

### 4. Clustering & Evaluation

The three RFM columns are standardized with `StandardScaler` and clustered with KMeans for each candidate K from 2 through `--max-k`. Standardization is mandatory because KMeans is distance-based and sensitive to feature scale. The model is fit and used for prediction with a **named DataFrame**, which avoids the scikit-learn feature-name warning. A fixed random seed and `n_init=10` keep runs repeatable.

For each K the report includes:

- **Inertia** – sum of squared distances to the assigned cluster centers. It cannot increase as K grows, so read the curve for diminishing returns rather than minimizing it alone.
- **Silhouette score** – compares within-cluster cohesion to separation from the nearest other cluster (range −1 to 1). A high score does not establish useful cluster sizes or business value.
- **Cluster sizes and imbalance** – expose tiny or degenerate segments that a single overall score can hide.

The **elbow plot** (inertia vs K) is a heuristic, not a precise optimum detector. The **silhouette plot** shows average silhouette across the same candidates.

**K selection:** the current selection uses silhouette subject to the existing minimum-cluster-share rule where possible, falling back to the highest silhouette if every candidate fails that size threshold. The report still flags any resulting imbalance — no K is forced to create a more appealing story.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled_df)
score = silhouette_score(X_scaled_df, labels)
```

### 5. Preprocessing & Algorithm Comparison

Beyond the baseline, the project compares:

- **Baseline** standardized RFM vs. a non-negative **`log1p` transform followed by standardization** (to address skewed RFM distributions)
- **KMeans vs. agglomerative clustering** on the baseline standardized features

Both comparisons use the same customers and feature definitions. KMeans inertia is not defined for agglomerative clustering. Silhouette scores across different transformations or algorithms are useful diagnostics, but are **not** a fully controlled measure of business utility.

### 6. Outlier Handling

The original notebook removed outliers using IQR filtering on Monetary Value and Frequency and re-fit the model on the filtered set (producing five clusters). That approach was judged heavy-handed for a segmentation baseline, since it can remove valid but high-spending customers and make results less representative.

**The current CLI retains outliers.** Extreme wholesale customers therefore influence the cluster centers, which is the main reason the K=2 result isolates a tiny high-value cohort. Log-transform comparisons (see above) are the current way to examine sensitivity to skew.

---

## 📊 Current Measured Results

The checked-in summary reflects a run on the first worksheet of the Online Retail II workbook with `--max-k 8`:

| Measure                          | Result                 |
|----------------------------------|-----------------------:|
| Raw input rows                   | 525,461                |
| Rows retained after cleaning     | 406,309                |
| Customers represented in RFM     | 4,285                  |
| Selected K                       | 2                      |
| Best and selected silhouette     | 0.9230 (0.9229566…)    |
| Cluster sizes                    | 4,271 and 14           |

### K=2 Cluster Profiles

Monetary values are in GBP, calculated from retained positive purchase lines only. Revenue share is each cluster's total MonetaryValue divided by the total across modeled customers.

| Cluster | Customers | Customer share | Mean Recency (days) | Median Recency (days) | Mean Frequency | Median Frequency | Mean Monetary | Median Monetary | Revenue share |
|--------:|----------:|---------------:|--------------------:|----------------------:|---------------:|-----------------:|--------------:|----------------:|--------------:|
| 0       | 4,271     | 99.67%         | 89.83               | 51                    | 4.11           | 2                | £1,677.79     | £694.72         | 82.62%        |
| 1       | 14        | 0.33%          | 3.14                | 2                     | 101.79         | 92.5             | £107,680.30   | £70,597.48      | 17.38%        |

**Interpretation**
- **Cluster 0** is the broad majority, with lower average purchase frequency and monetary value.
- **Cluster 1** contains only 14 customers with much lower recency and markedly higher frequency and monetary value — a very small, recently active, high-value *observed* cohort. This does **not** prove loyalty, future spend, or churn risk, and the tiny size makes it sensitive to data and modeling changes.
- Across K=2..8, inertia falls as expected; silhouette is highest at K=2 and drops substantially at K=3. Several higher-K solutions also contain very small clusters.

The plots therefore do **not** establish a balanced operational segmentation. K=2 remains the statistical baseline; practical use requires business validation and careful handling of the 14-customer segment.

### Suggested Business Reading (Illustrative)

| Segment pattern                          | Typical action                    |
|------------------------------------------|-----------------------------------|
| Low recency + high frequency + high spend | Reward and retain (e.g. VIP/wholesale handling) |
| Broad base with lower spend and frequency | Growth and re-engagement campaigns |

> Richer labels (Champions, At-Risk, Lost, New Customers) require a more balanced clustering than the current K=2 baseline provides.

---

## 📁 Generated Outputs

The default output directory is `outputs/`. It is created automatically and is **not** ignored by `.gitignore`, so representative outputs can remain part of the repository.

| Output                          | Contents                                                                   |
|---------------------------------|----------------------------------------------------------------------------|
| `segmentation_summary.json`     | Selected K, row/customer counts, silhouette results, cluster sizes and profiles |
| `cluster_profiles.csv`          | Per-cluster counts, percentages, mean/median RFM, revenue and revenue share |
| `k_evaluation.csv`              | KMeans inertia, silhouette, cluster sizes and balance diagnostics per K    |
| `preprocessing_comparison.csv`  | Baseline vs. log1p + scaling metrics for each candidate K                  |
| `algorithm_comparison.csv`      | KMeans vs. agglomerative silhouette and cluster-size comparison            |
| `data_quality_summary.csv`      | Counts of missing IDs, duplicates, invalid invoices/codes, negative/zero values and rows retained |
| `diagnostics/elbow_curve.png`   | Inertia by K                                                               |
| `diagnostics/silhouette_curve.png` | Average silhouette by K                                                 |

> `data_quality_summary.csv` reports overlapping issue counts, so don't sum them to infer total rows removed. The exact retained row count comes from applying the full cleaning mask.

---

## 🧪 Tests

The suite in `tests/test_customer_segmentation.py` covers:

- workbook loading and schema validation
- data cleaning
- RFM calculation and reference-date behavior
- clustering metrics and K selection
- cluster-profile generation
- preprocessing and algorithm comparisons
- diagnostic file generation
- data-quality summaries

```bash
uv run pytest -q
```

The last observed full test run before repository restructuring was **10 passed**. Re-run the command after setup or code changes to verify your environment.

---

## 🏗 Architecture

| Module                                      | Responsibility                                              |
|---------------------------------------------|-------------------------------------------------------------|
| `main.py`                                   | Coordinates the end-to-end CLI flow and file exports        |
| `src/customer_segmentation/config.py`       | Centralizes project-root path resolution                    |
| `src/customer_segmentation/data_loader.py`  | Reads the workbook and applies transaction filtering        |
| `src/customer_segmentation/rfm.py`          | Builds customer-level RFM features                          |
| `src/customer_segmentation/clustering.py`   | Standardizes features, evaluates K, fits KMeans             |
| `src/customer_segmentation/reporting.py`    | Profiles, diagnostic plots, quality counts, model comparisons |
| `tests/`                                    | Validates core behaviors                                    |

---

## ⚠️ Known Limitations

- This is **unsupervised descriptive segmentation**, not a churn, lifetime-value, or causal model.
- The highest-silhouette result (K=2) is severely imbalanced and may not be useful for broad campaign segmentation.
- The dataset is a historical snapshot; results depend on the chosen worksheet, cleaning policy, and reference date.
- Returns and cancellations are excluded, so revenue profiles describe retained positive purchases — **not** net accounting revenue.
- Outliers are retained. Standard scaling does not make skewed distributions normal, and extreme wholesale customers can influence KMeans centers.
- Statistical comparisons do not replace stability analysis across resamples/time periods or validation against business outcomes.
- The notebook and `notebooks/legacy_output.png` are historical exploratory material and may show methodology/results (e.g. IQR outlier removal, five clusters) that differ from the current CLI.

---

## 📜 Project History

The project was restructured from a notebook-only implementation into a tested CLI package. Highlights:

- Replaced the placeholder `main.py` with a reproducible CLI workflow and repository-relative path resolution
- Extracted reusable logic into `src/customer_segmentation/`
- Added automated tests, K-evaluation diagnostics, preprocessing/algorithm comparisons, and machine-readable outputs
- Added a data-quality summary and an explicit retention policy
- Fixed the scikit-learn feature-name warning
- Moved the notebook to `notebooks/` and the workbook to `data/`; expanded `.gitignore`
- Reconciled `uv.lock` with the declared Python requirement (`>=3.12`) and dev dependencies

For full details, see [CHANGELOG.md](CHANGELOG.md), [PROJECT_AUDIT.md](PROJECT_AUDIT.md), and [PROJECT_CHECKLIST.md](PROJECT_CHECKLIST.md).

---

## 🛠 Tech Stack

| Tool / Library   | Purpose                                                   |
|------------------|-----------------------------------------------------------|
| `uv`             | Dependency management and command running                 |
| `pandas`         | Data loading, cleaning, aggregation                       |
| `scikit-learn`   | Scaling, KMeans, agglomerative clustering, silhouette metrics |
| `matplotlib`     | Elbow and silhouette diagnostic plots                     |
| `openpyxl`       | Excel engine for reading `.xlsx` files with pandas        |
| `pytest`         | Automated testing                                         |
| `re` (stdlib)    | Regex validation of invoice numbers and stock codes       |

---

## 🙏 Acknowledgements

- **Dataset:** [UCI ML Repository – Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (CC BY 4.0)
- **Whitepaper inspiration:** [Mining Online Retail Customer Data](https://link.springer.com/article/10.1057/dbm.2012.17)
- **Urgency/Importance Matrix image credit:** [Rushcutters Health](https://rushcuttershealth.com.au/how-to-prioritise-tasks-and-get-more-of-the-important-things-done-the-urgent-vs-important-matrix/)

---

## 📄 License

This project is licensed under the **MIT License**.
