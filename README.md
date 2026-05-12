<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.27+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.103+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Deployed-Live-brightgreen?style=for-the-badge" />
</p>

# Retail Customer Segmentation — End-to-End ML Pipeline

> ```
>
> ```
>
> **Turn 110K raw e-commerce transactions into four actionable customer personas — from**
>
> **data ingestion to a live, interactive dashboard.**

 **[Live Demo → customer-segmentation-os.streamlit.app](https://customer-segmentation-os.streamlit.app/)**

---

## Problem Statement

A Brazilian e-commerce marketplace (Olist) has **93,000+ customers** generating **110,000+ transactions** worth **R$ 13.2 million** over two years. The marketing team has no systematic way to:

- Identify their most valuable customers before they churn
- Distinguish one-time buyers with growth potential from genuinely dissatisfied ones
- Allocate limited campaign budgets to the right cohort
- Personalize outreach instead of blasting generic promotions

**Goal:** Build an unsupervised ML system that automatically segments the customer base into distinct, business-interpretable personas — each with a concrete marketing action — and serve it through a real-time prediction interface.

---

## Approach & Methodology

### Data Engineering — Relational Join Pipeline

The raw Olist dataset is split across **9 normalized CSV tables** (customers, orders, items, payments, reviews, products, sellers, geolocation, translations). The pipeline performs a multi-step relational join chain with pre-aggregation to avoid Cartesian explosion:

```
Customers ──┐
             ├── Orders + Payments (agg to order-level) + Reviews (latest per order)
Products ───┤
Sellers ────┘
```

### Data Cleaning & Validation

- Filtered to **delivered orders only** (removed cancelled/unavailable)
- Parsed 6 datetime columns, handled missing review scores (median imputation)
- Removed anomalous records (zero/negative payments)
- Extracted temporal features (year, month, day-of-week)

### Feature Engineering — Extended RFM

Classic RFM (Recency, Frequency, Monetary) extended with behavioral signals:

| Feature                    | Description              | Business Signal             |
| :------------------------- | :----------------------- | :-------------------------- |
| **Recency**          | Days since last purchase | Engagement freshness        |
| **Frequency**        | Count of unique orders   | Purchase habit strength     |
| **Monetary**         | Total spend (BRL)        | Revenue contribution        |
| **Avg Review Score** | Mean rating given        | Satisfaction level          |
| **Avg Installments** | Mean installment usage   | Price sensitivity indicator |

### Modeling — K-Means Clustering

- **Log-transformation** on skewed features (Recency, Frequency, Monetary)
- **StandardScaler** normalization to unit variance
- **K-Means** (k=4) selected via Elbow Method + Silhouette Analysis
- **PCA** (2 components) for 2D visualization projection

### Deployment — Streamlit Cloud

- Models serialized with `joblib` (scaler, K-Means, PCA)
- Interactive dashboard with real-time prediction on user inputs
- GitHub Actions cron job pings app every 5 days to prevent hibernation

---

## Customer Personas Discovered

<table>
<tr>
<td align="center" width="25%">

---

## Project Architecture

```
customer-segmentation/
│
├── data/
│   ├── raw/                        # 9 Olist CSV tables (~125 MB)
│   └── processed/                  # Parquet checkpoints
│
├── src/                            # Core ML pipeline modules
│   ├── ingestion.py                # Multi-file CSV loader with validation
│   ├── merge.py                    # Relational join chain (pre-aggregated)
│   ├── cleaning.py                 # Filtering, imputation, type casting
│   ├── eda.py                      # Automated EDA plots & summary report
│   ├── features.py                 # Extended RFM feature engineering
│   └── model.py                    # Scaling → K-Means → PCA → Serialization
│
├── api/
│   └── main.py                     # FastAPI REST endpoint (/predict-segment)
│
├── frontend/
│   └── app.py                      # Streamlit interactive dashboard
│
├── outputs/
│   ├── models/                     # scaler.pkl, kmeans_model.pkl, pca_transformer.pkl
│   ├── plots/                      # Auto-generated EDA visualizations
│   └── reports/                    # EDA text summary
│
├── config.py                       # Centralized paths & constants
├── run_pipeline.py                 # One-command full pipeline execution
├── run_app.py                      # Launch API + Frontend together
├── profile_clusters.py             # Cluster profiling utility
├── requirements.txt                # Pinned dependencies
└── .github/workflows/
    └── keep-alive.yml              # Cron job to prevent Streamlit sleep
```

---

## ⚙️ Tech Stack

| Layer                      | Technology                                 | Purpose                               |
| :------------------------- | :----------------------------------------- | :------------------------------------ |
| **Data Processing**  | Pandas, NumPy, PyArrow                     | ETL, feature engineering, Parquet I/O |
| **Visualization**    | Matplotlib, Seaborn, Plotly                | EDA plots, interactive charts         |
| **Machine Learning** | Scikit-Learn (KMeans, PCA, StandardScaler) | Clustering, dimensionality reduction  |
| **Model Serving**    | Joblib                                     | Artifact serialization & loading      |
| **API**              | FastAPI, Pydantic, Uvicorn                 | REST endpoint with input validation   |
| **Frontend**         | Streamlit                                  | Interactive prediction dashboard      |
| **Deployment**       | Streamlit Community Cloud                  | Free-tier cloud hosting               |
| **CI/CD**            | GitHub Actions                             | Keep-alive cron, version control      |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (place CSVs in `data/raw/`)

### Installation

```bash
git clone https://github.com/ayushcode001/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
```

### Run the Full Pipeline (Training)

```bash
python run_pipeline.py
```

This executes: `Ingestion → Merge → Clean → EDA → Feature Engineering → K-Means Training → Model Export`

### Launch the Dashboard

```bash
streamlit run frontend/app.py
```

### Launch with API Backend (optional)

```bash
python run_app.py
```

---

## Key Results

| Metric                       | Value                      |
| :--------------------------- | :------------------------- |
| Total transactions processed | **110,194**          |
| Unique customers segmented   | **93,357**           |
| Total revenue analyzed       | **R$ 13.2M**         |
| Date range                   | Oct 2016 — Aug 2018       |
| Number of clusters           | **4**                |
| Clustering algorithm         | K-Means                    |
| Validation metric            | Silhouette Score           |
| Deployment                   | Live on Streamlit Cloud |

---

## API Reference

**`POST /predict-segment`**

```json
{
  "customer_id": "CUST_9999",
  "recency_days": 30,
  "frequency": 3,
  "monetary_value": 450.0,
  "avg_review_score": 4.2,
  "avg_installments": 3.0
}
```

**Response:**

```json
{
  "customer_id": "CUST_9999",
  "cluster_id": 3,
  "persona": "Champion",
  "pca_coordinates": { "x": 1.24, "y": -0.87 }
}
```




<p align="center">
  <b>Built by <a href="https://github.com/ayushcode001">Ayush</a></b>
</p>
