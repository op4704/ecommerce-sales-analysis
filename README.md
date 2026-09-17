# E-Commerce Sales Data Analysis & Prediction System

A full end-to-end data science pipeline built on real e-commerce transaction data:
load → clean → preprocess → explore (EDA) → visualize → engineer features →
train ML models → evaluate → predict future sales → report findings.

Built not just to produce a working system, but to understand **why** each step
matters and **how** it's done — every notebook explains the reasoning, not just the code.

## Dataset

**Online Retail II** (UCI Machine Learning Repository / Dr. Daqing Chen, LSBU)
Real transaction-level data from a UK-based online retailer, 01/12/2009–09/12/2011.
~1 million rows, genuinely messy (missing Customer IDs, cancelled orders, negative
quantities/returns) — good for real-world cleaning practice.

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

## Goal

Predict future e-commerce sales revenue via time-series forecasting, using
historical transaction patterns, seasonality, and engineered features.

## Project Structure

```
data/
  raw/            # original untouched dataset (not committed — see .gitignore)
  processed/      # cleaned/feature-engineered data
notebooks/        # step-by-step learning pipeline (01 -> 07)
src/ecommerce_sales_analysis/   # reusable pipeline as clean Python modules
reports/          # final findings, charts, conclusions
```

## Pipeline Stages

1. **Data Loading** — pull the dataset, inspect shape/dtypes/basic structure
2. **Data Cleaning** — handle missing values, duplicates, invalid rows, cancellations
3. **EDA & Visualization** — trends, seasonality, top products/customers/countries
4. **Feature Engineering** — lag features, rolling averages, calendar features, RFM
5. **Model Training** — baseline + ML models for sales forecasting
6. **Model Evaluation** — MAE/RMSE/MAPE, train vs test, residual analysis
7. **Prediction & Final Report** — forecast future sales, summarize insights

## Setup

```bash
uv sync
uv run jupyter lab
```

## Status

✅ Pipeline complete — load → clean → EDA → feature engineering → ML →
evaluation → prediction, all steps built, executed, and tested end-to-end.

## Final Results

- **Best model:** Random Forest — 33.6% RMSE improvement over a naive
  "predict yesterday's revenue" baseline.
- **14-day forecast (from Dec 10, 2011):** ~£494,802 total predicted revenue,
  ~£35,343/day average (vs. ~£43,968/day actual average over the prior 30 days).
- **Residual analysis:** mean residual ~£2,523 — a mild but not alarming bias
  relative to typical daily revenue (~£20-30K/day).
- Full write-up, limitations, and next steps: see the "Final Report" section
  at the bottom of `notebooks/07_prediction_report.ipynb`.

## Key Findings So Far (from EDA)

- UK accounts for 82.8% of total revenue — heavily concentrated in one market.
- Top 20% of customers generate 77.2% of total revenue (classic 80/20 pattern).
- Revenue per line item has a long right tail (99th percentile ≈ £204, max ≈ £168K)
  from large wholesale orders — legitimate sales, not data errors.
- See `notebooks/03_eda_visualization.ipynb` for trend/seasonality charts.
