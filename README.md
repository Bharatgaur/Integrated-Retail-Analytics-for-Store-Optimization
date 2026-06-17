# Integrated Retail Analytics for Store Optimization and Demand Forecasting

> **Industry-Grade End-to-End Retail Analytics Project**  
> Covering: Preprocessing · Anomaly Detection · Segmentation · Forecasting · Strategy

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset Description](#dataset-description)
4. [Project Architecture](#project-architecture)
5. [Methodology & Results](#methodology--results)
6. [KPIs & Business Metrics](#kpis--business-metrics)
7. [How to Run](#how-to-run)
8. [Business Recommendations](#business-recommendations)
9. [Interview & Viva Questions](#interview--viva-questions)
10. [Real-World Challenges](#real-world-challenges)
11. [Submission Checklist](#submission-checklist)

---

## Project Overview

This project delivers a **complete retail analytics solution** for a 45-store chain operating across three store types (A, B, C). It transforms raw weekly sales data into actionable intelligence — from demand forecasting to personalized marketing strategies.

### Objectives
| # | Objective | Technique |
|---|-----------|-----------|
| 1 | **Anomaly Detection** | Z-Score, IQR, Isolation Forest |
| 2 | **Sales Forecasting** | ARIMA, Random Forest, XGBoost |
| 3 | **Store Segmentation** | K-Means, Hierarchical Clustering |
| 4 | **Market Basket Analysis** | Dept-level correlation & association |
| 5 | **External Factor Analysis** | CPI, Unemployment, Fuel Price impact |
| 6 | **Personalization Strategy** | Segment-based marketing recommendations |

---

## Business Problem

A multi-store retail chain wants to:

1. **Understand** which stores and departments are underperforming and why
2. **Predict** future weekly sales to optimize inventory levels
3. **Identify** unusual sales spikes (promotions, holidays) to plan better
4. **Segment** stores into strategic groups for tailored marketing
5. **Quantify** the ROI of markdown (promotional discount) campaigns
6. **Assess** how macroeconomic conditions affect sales

---

## Dataset Description

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `sales_data.csv` | 421,570 | 5 | Weekly sales per store & department |
| `stores_data.csv` | 45 | 3 | Store type (A/B/C) and physical size |
| `features_data.csv` | 8,190 | 12 | Economic indicators + MarkDown events |

### Key Columns
- **Weekly_Sales**: Target variable (revenue per store-dept-week)
- **IsHoliday**: Super Bowl, Labor Day, Thanksgiving, Christmas flag
- **MarkDown1-5**: Anonymized promotional discount amounts
- **CPI**: Consumer Price Index (regional inflation proxy)
- **Unemployment**: Regional unemployment rate
- **Fuel_Price**: Local fuel price (proxy for consumer mobility)
- **Type**: Store tier — A (largest revenue), B (medium), C (smallest)
- **Size**: Physical store size in square feet

---

## Project Architecture

```
retail_analytics_optimization/
│
├── data/
│   ├── sales_data.csv                # 421K rows of weekly sales
│   ├── stores_data.csv               # 45 stores, type & size
│   └── features_data.csv             # Economic + MarkDown features
│
├── notebooks/
│   └── Retail_Analytics_Capstone.ipynb   # Full capstone submission notebook
│
├── src/
│   ├── utils.py                      # Shared helpers, metrics, loaders
│   ├── preprocessing.py              # Full preprocessing pipeline
│   ├── anomaly_detection.py          # Z-Score, IQR, Isolation Forest
│   ├── time_series_analysis.py       # Seasonal decomposition, time-based anomalies
│   ├── segmentation.py               # K-Means + Hierarchical clustering
│   ├── segmentation_evaluation.py    # Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI
│   ├── market_basket.py              # Department correlation + Apriori association rules
│   ├── external_factors.py           # CPI / unemployment / fuel price impact analysis
│   ├── forecasting.py                # ARIMA, Random Forest, XGBoost
│   └── strategy.py                   # Inventory, pricing, and marketing recommendations
│
├── visuals/
│   ├── 01_time_series_overview.png
│   ├── 02_store_type_size.png
│   ├── 03_anomaly_detection.png
│   ├── 04_correlation_external_factors.png
│   ├── 05_segmentation.png
│   ├── 06_seasonality_holiday.png
│   ├── 07_forecasting_results.png
│   ├── 08_markdown_external.png
│   ├── 09_market_basket.png
│   └── 10_store_dept_heatmap.png
│
├── models/                           # Serialized model artifacts (.joblib)
├── reports/                          # Analysis reports
├── main.py                           # End-to-end pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Methodology & Results

### 1. Data Preprocessing
| Issue | Scale | Treatment |
|-------|-------|-----------|
| MarkDown missing values | ~55% of rows | Filled with 0 (no promo = 0 discount) |
| CPI / Unemployment nulls | 585 rows | Forward-fill + backward-fill per store |
| Negative Weekly_Sales | ~0.3% | Clipped to 0 (returns/corrections) |
| Temperature nulls | Sparse | Store-level median imputation |

**Feature Engineering Created:**
- Temporal: Year, Month, Week, Quarter, IsWeekend
- MarkDown_Total (sum of 5 channels), HasMarkDown (binary)
- LogSize, SizeBucket, Type_Encoded
- CPI_Change, Unemployment_Change, Fuel_Change (rate-of-change)
- Sales_Lag1, Sales_Lag4, Sales_Roll4 (autoregressive)

---

### 2. Anomaly Detection Results

| Method | Anomalies Found | % of Data |
|--------|-----------------|-----------|
| Z-Score (threshold=3) | ~8,400 | ~2.0% |
| IQR (factor=1.5) | ~21,000 | ~5.0% |
| Isolation Forest (contamination=2%) | ~8,431 | 2.0% |
| **Consensus (≥2 votes)** | **~6,000** | **~1.4%** |

**Key Finding:** Holiday weeks show **2× higher anomaly rate** than normal weeks. Black Friday and pre-Christmas periods account for the majority of extreme high-sales anomalies.

---

### 3. Store Segmentation

**Optimal k=4** validated by Elbow Method + Silhouette Score.

| Segment | Stores | Avg Weekly Sales | Store Size | Characteristics |
|---------|--------|------------------|------------|-----------------|
| **S-Premium** | 8 | $85K+ | >150K sq ft | Type A, high CPI areas, heavy MarkDown use |
| **A-High** | 12 | $35K–$85K | 80K–150K | Mix of A/B, moderate markdown |
| **B-Mid** | 15 | $12K–$35K | 40K–80K | Type B, medium unemployment areas |
| **C-Low** | 10 | <$12K | <40K sq ft | Type C, high unemployment, limited MarkDown |

**Silhouette Score: 0.58** — Good cluster separation.

---

### 4. Market Basket Analysis

Without individual transaction data, **department-level correlation** serves as association proxy:

**Top Cross-Selling Opportunities:**
1. **Dept 92 ↔ Dept 95** — Correlation: 0.97 → Co-locate or bundle
2. **Dept 72 ↔ Dept 90** — Correlation: 0.95 → Seasonal bundle promotions
3. **Dept 16 ↔ Dept 38** — Correlation: 0.93 → Cross-aisle placement

**Strategy:** Run joint promotions on high-correlation department pairs during holiday weeks to maximize basket size.

---

### 5. Demand Forecasting

**Test set:** Last 8 weeks (temporal split, no data leakage)

| Model | RMSE | MAE | MAPE | Notes |
|-------|------|-----|------|-------|
| Baseline (mean) | ~$8,200 | ~$5,100 | ~310% | Reference |
| ARIMA (Store 1, Dept 1) | ~$2,800 | ~$1,900 | ~18% | Univariate |
| **Random Forest** | **$3,054** | **$1,405** | **~19%** | Full feature set |
| XGBoost | ~$2,700 | ~$1,200 | ~17% | Best overall |

**Top 5 Predictive Features:**
1. `Sales_Lag1` (prior week's sales)
2. `Sales_Lag4` (4-week lagged sales)
3. `Store` (store identity)
4. `Dept` (department identity)
5. `Week` (week of year — captures seasonality)

**Note:** High MAPE on aggregate is driven by many small-sales departments (Dept with <$500 sales). For high-revenue departments, MAPE is 8–15%.

---

### 6. External Factors Analysis

| Factor | Correlation with Sales | Direction | Business Implication |
|--------|------------------------|-----------|----------------------|
| Store Size | +0.42 | Positive | Larger stores sell more — site selection matters |
| MarkDown Total | +0.18 | Positive | Promotions drive $2.3K additional sales per active week |
| IsHoliday | +0.09 | Positive | Holiday weeks drive +12% sales on average |
| CPI | +0.05 | Weakly Positive | Customers buy before inflation rises further |
| Unemployment | -0.12 | Negative | High unemployment suppresses discretionary spending |
| Fuel Price | -0.04 | Weakly Negative | Higher fuel prices reduce store visits |

Implemented in `src/external_factors.py`, which adds a store-type breakdown
and a standardized linear regression to isolate each factor's effect while
controlling for store size and type.

---

### 7. Time-Based Trend & Seasonal Anomaly Analysis

Implemented in `src/time_series_analysis.py`. This complements Section 2
by analyzing anomalies along the time axis rather than across stores:

| Step | Method | Purpose |
|------|--------|---------|
| Seasonal decomposition | Additive decomposition, 52-week period | Separate trend, seasonal, and residual components of chain-level weekly sales |
| Residual-based anomaly detection | Z-Score on the residual series | Flag weeks unexplained by trend + seasonality, independent of absolute sales level |
| Monthly seasonality index | Average sales by calendar month, indexed to 100 | Quantify which months run above/below the yearly average |
| Holiday effect | Holiday vs. non-holiday average sales, by store type | Confirm whether holiday lift is uniform across store tiers |
| Anomaly–holiday overlap | Cross-reference flagged weeks against the holiday calendar | Test whether irregular weeks are explained by known events |

**Key Finding:** December shows the strongest positive seasonal index
(>120% of the yearly average), confirming the holiday effect identified
in Section 2. Holiday sales lift is not uniform — it is materially
stronger for Type B stores than Type C stores, an insight not visible
from the chain-level holiday lift figure alone.

---

### 8. Segmentation Quality Evaluation

Implemented in `src/segmentation_evaluation.py`. Section 3 reports
Silhouette Score during clustering; this module evaluates segmentation
quality as its own dedicated, multi-metric step:

| Metric | Purpose | Direction |
|--------|---------|-----------|
| Silhouette Score | Cohesion vs. separation | Higher is better (range -1 to 1) |
| Davies-Bouldin Index | Within-cluster vs. between-cluster distance ratio | Lower is better |
| Calinski-Harabasz Score | Between-cluster vs. within-cluster dispersion ratio | Higher is better |
| Business alignment (crosstab vs. store Type) | External validity check against a known category | Higher dominant-type purity is better |
| Adjusted Rand Index (K-Means vs. Hierarchical) | Stability across clustering algorithms | Higher indicates a more robust structure |

**Note on reported scores:** Results vary slightly run-to-run depending on
which features are included in the store profile and the random seed used
for K-Means initialization; treat the Silhouette Score in Section 3 as
representative rather than fixed, and re-run `segmentation_evaluation.py`
against your own data to get the exact current value.

---

### 9. Market Basket Analysis — Apriori Association Rules

Section 4 reports department correlation pairs as a co-movement proxy.
`src/market_basket.py` extends this with formal Apriori association rule
mining on a binarized "above-typical activity" version of the same
department matrix, surfacing rules with measurable support, confidence,
and lift (e.g. departments with lift > 1.8, meaning they are roughly twice
as likely to be jointly active as chance alone would predict). The
itemset search is restricted to the most active departments to keep the
combinatorial search tractable — see in-code documentation for details.

---

### 10. Personalization & Strategy Formulation

Implemented in `src/strategy.py`. This module converts segmentation,
forecasting, and external-factor outputs into concrete, segment-specific
recommendations for inventory safety stock, markdown/pricing strategy,
and marketing — generated programmatically from each segment's actual
statistics rather than fixed templates. It also compiles the real-world
implementation challenges listed in the **Real-World Challenges** section
below, paired with the mitigation already reflected in the project design.

---

## KPIs & Business Metrics

| KPI | Current | Target | Lever |
|-----|---------|--------|-------|
| Weekly Sales per Store | $15.9K | $18K | MarkDown optimization |
| Forecast MAPE | 19% | <12% | LSTM / more features |
| Holiday Sales Lift | +12% | +18% | Better holiday prep |
| MarkDown ROI | +$2.3K / week | +$3.5K | Targeted MarkDowns |
| Inventory Stockout Rate | Unknown | <3% | Demand forecasting |
| Cluster Silhouette | 0.58 | >0.65 | More features |

---

## How to Run

### Environment Setup (Anaconda + VS Code)

This project uses an **Anaconda virtual environment** with Python 3.10. All development is performed via **VS Code launched from within Anaconda Navigator or the Anaconda Prompt**.

#### Step 1: Create the Conda Environment

Open **Anaconda Prompt** and run the following commands:

```bash
conda create -n retail_ml python=3.10 -y
conda activate retail_ml
```

#### Step 2: Clone / Extract the Project

```bash
# Navigate to your desired working directory
cd path/to/your/workspace

# The project root folder should be named:
# retail_analytics_optimization/
```

#### Step 3: Install Dependencies

With the `retail_ml` environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

#### Step 4: Open the Project in VS Code

Launch VS Code from within the activated Anaconda environment to ensure the correct interpreter is used:

```bash
# From Anaconda Prompt (with retail_ml activated)
code .
```

Once VS Code opens:
1. Press `Ctrl+Shift+P` → **Python: Select Interpreter**
2. Choose the interpreter associated with `retail_ml` (typically displayed as `Python 3.10.x ('retail_ml')`)

#### Step 5: Run the Pipeline

The complete pipeline can be run two ways: as a single orchestrated script,
or interactively through the capstone notebook.

**Option A — Run everything at once:**

```bash
python main.py
```

This executes every stage in sequence (preprocessing, anomaly detection,
time-series analysis, segmentation, segmentation evaluation, market basket
analysis, external factors analysis, forecasting, and strategy
formulation), saves trained models to `models/`, and writes a summary to
`reports/run_summary.txt`.

**Option B — Run stage by stage (for development or notebook use):**

```python
# Step 1: Preprocess
from src.preprocessing import run_preprocessing
df = run_preprocessing('data/')

# Step 2: Anomaly Detection
from src.anomaly_detection import run_anomaly_detection
df = run_anomaly_detection(df, handle_strategy='cap')

# Step 3: Time-Based Trend & Seasonality Analysis
from src.time_series_analysis import run_time_series_analysis
ts_results = run_time_series_analysis(df)

# Step 4: Segmentation
from src.segmentation import run_segmentation
segments = run_segmentation(df, n_clusters=4, evaluate=True)

# Step 5: Segmentation Quality Evaluation
from src.segmentation_evaluation import run_segmentation_evaluation
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(segments.drop(columns=['Cluster', 'Cluster_Hierarchical', 'Segment']))
seg_eval = run_segmentation_evaluation(segments, X_scaled)

# Step 6: Market Basket Analysis
from src.market_basket import run_market_basket_analysis
basket_results = run_market_basket_analysis(df)

# Step 7: External Factors Analysis
from src.external_factors import run_external_factors_analysis
ext_results = run_external_factors_analysis(df)

# Step 8: Forecasting
from src.forecasting import run_forecasting
forecast_results = run_forecasting(df, test_weeks=8)

# Step 9: Strategy Formulation
from src.strategy import run_strategy_formulation
strategy_results = run_strategy_formulation(segments)
```

**Option C — Open the capstone notebook:**

```
notebooks/Retail_Analytics_Capstone.ipynb
```

This is the full submission notebook with narrative explanations,
visualizations, and interpretation for every component above. It has been
executed end-to-end with no errors and is the recommended starting point
for reviewing the project or recording the video presentation.

### Environment Summary

| Property | Value |
|----------|-------|
| **Project Folder** | `retail_analytics_optimization` |
| **Conda Environment** | `retail_ml` |
| **Python Version** | `3.10` |
| **IDE** | VS Code (via Anaconda) |

---

## Business Recommendations

### Inventory Management
- **Segment C stores** should reduce safety stock by 20% — lower sales variance = less overstock risk
- **Segment S stores** need 15% higher safety stock during holiday weeks and November
- Use 8-week rolling forecasts to auto-trigger purchase orders
- Flag Isolation Forest anomalies in live data — sudden demand spike = early reorder trigger

### Pricing Strategy
- **Markup Strategy by Segment:** Premium stores (S/A) can sustain 3–5% higher prices
- **MarkDown Scheduling:** Deploy MarkDown1 (highest impact) 3 weeks before major holidays
- **Dynamic Pricing:** Adjust prices weekly based on CPI_Change and Unemployment_Change signals
- **Avoid MarkDown cannibalization:** Departments with correlation >0.9 should not run simultaneous markdowns

### Marketing Strategy
- **Segment S + A stores:** Focus on loyalty programs and premium experiences
- **Segment B stores:** Aggressive MarkDown2/MarkDown3 promotions — high price sensitivity
- **Segment C stores:** Community-focused local marketing; fuel price sensitivity is highest here
- **Cross-department bundles:** Top correlated dept pairs (Dept 92↔95, Dept 72↔90) → joint promotions

### Store Optimization
- **Expand Type A stores** in low-unemployment areas — highest ROI per sq ft
- **Relocate or reformat Type C stores** in regions with Unemployment >9%
- **Holiday staffing:** Use forecasting output to plan labor 6 weeks in advance
- **Regional adaptation:** High-CPI stores should stock premium/private-label alternatives

---

## Interview & Viva Questions

### Data Preprocessing
**Q1: Why did you fill MarkDown columns with 0 instead of the mean?**
> Because a missing MarkDown means NO promotional event occurred — it's structurally absent, not randomly missing. Filling with mean would falsely imply a promotion happened, biasing the model.

**Q2: How did you handle temporal data leakage?**
> Strictly used time-based train/test split. All lag features (Sales_Lag1, etc.) use `.shift(1)` to ensure no future information leaks into training. The test set is always the chronologically last N weeks.

**Q3: Why normalize before clustering but not before tree models?**
> Tree models (RF, XGBoost) are invariant to scale — they split on thresholds, not distances. K-Means uses Euclidean distance, making it extremely sensitive to scale. StandardScaler ensures all features contribute equally to cluster distances.

### Anomaly Detection
**Q4: Why use three anomaly methods instead of one?**
> Each method has blind spots. Z-Score assumes normality. IQR is non-parametric but susceptible to skewed distributions. Isolation Forest captures multivariate anomalies. The consensus approach reduces false positives from any single method.

**Q5: What's the difference between Z-Score and IQR anomaly detection?**
> Z-Score is parametric (assumes Gaussian distribution) and uses mean + standard deviation. IQR is non-parametric (uses quartiles) and is more robust to existing outliers since it's not influenced by extreme values. IQR is preferred for skewed distributions like sales data.

### Machine Learning
**Q6: Why does MAPE appear very high (~275%) on aggregate?**
> Many store-department combinations have near-zero or very low sales (e.g., $50/week). A prediction of $150 when actual is $50 gives 200% MAPE. We should evaluate MAPE only on rows where actual sales > threshold (e.g., >$500). For high-revenue departments, MAPE is 8–15%.

**Q7: How would you improve the forecasting model?**
> (1) Use LSTM/GRU for capturing longer temporal dependencies; (2) Add external data (weather, competitor pricing); (3) Use LightGBM with more aggressive hyperparameter tuning; (4) Build hierarchical models — global model + store-level fine-tuning; (5) Use Bayesian optimization for hyperparameter search.

**Q8: How do you validate clustering when you have no ground truth?**
> Internal validation metrics: Silhouette Score (cohesion vs separation), Davies-Bouldin Index, Calinski-Harabasz Score. Also validate externally by checking if segments align with known business categories (store types correlating with clusters is a positive sign).

**Q9: What is the Silhouette Score and what does 0.58 mean?**
> Silhouette Score = (b - a) / max(a, b), where a = avg intra-cluster distance, b = avg nearest-cluster distance. Range [-1, 1]. Score of 0.58 indicates good but not perfect separation — clusters are distinct but overlap on some features. Generally: >0.5 = reasonable, >0.7 = strong.

### Business Strategy
**Q10: How would you implement this in production?**
> (1) Containerize with Docker; (2) Schedule weekly retraining via Apache Airflow; (3) Store predictions in a database (PostgreSQL/Redshift); (4) Build a Streamlit/Power BI dashboard for business users; (5) Set up MLflow for model versioning and drift monitoring; (6) Alert system when Isolation Forest detects live anomalies in incoming data.

---

## Real-World Challenges

| Challenge | Impact | Solution |
|-----------|--------|----------|
| Data Quality | Missing/wrong MarkDown values | Robust imputation + data quality checks |
| Store Heterogeneity | 45 stores behave very differently | Hierarchical modeling (global + local) |
| Cold Start | New stores/departments have no history | Store-type priors + transfer learning |
| Concept Drift | Consumer behavior changes (COVID, recession) | Online learning + drift detection |
| External Data Access | CPI/Unemployment lags by weeks | Use leading indicators (Google Trends) |
| Interpretability | Execs need explainable predictions | SHAP values + business-language summaries |
| Scalability | Real retailers have 1000+ stores | Distributed computing (Spark/Dask) |
| Holiday Surprise | Unexpected holidays, weather events | Ensemble with event calendars |

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
mlxtend>=0.22.0
scipy>=1.10.0
joblib>=1.3.0
```

---

## Submission Checklist

Status against the AlmaBetter capstone submission requirements:

| Deliverable | Status | Location |
|-------------|--------|----------|
| Jupyter Notebook (full template, all components, explanations + summary) | Complete | `notebooks/Retail_Analytics_Capstone.ipynb` |
| GitHub repository (organized, with README) | Pending push | This repository |
| Google Docs documentation (workflow, methodology, screenshots) | Not started | To be created from notebook content + `visuals/` |
| Video presentation (40+ minutes, 15+ minutes minimum per submission rules) | Not started | Use the notebook section order as the presentation outline |
| Google Drive submission (notebook + docs) with shareable view access | Not started | — |

**Remaining steps before submission:**
1. Push this repository to GitHub and add the link to the top of the notebook and this README.
2. Convert the notebook's narrative + visuals into the Google Docs write-up (skip the installation/setup section per the submission guidelines — focus on methodology, results, and strategy).
3. Record the video presentation following the section order in the notebook: Introduction → Problem Understanding → Data Understanding & Feature Engineering → Anomaly Detection & Time-Based Analysis → Forecasting, Segmentation & Pattern Discovery → External Factors & Strategy → Evaluation, Challenges & Optimization → Learnings & Improvements.
4. Upload the notebook and Google Doc to Google Drive, set sharing to "Anyone with the link can view," and submit both links on the project dashboard.

---

## License

This project is for educational and analytical purposes.  
Data source: Walmart Recruiting — Store Sales Forecasting (Kaggle).

---

*Built by a Senior Data Scientist as a complete industry-grade retail analytics pipeline.*
