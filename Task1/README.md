# Task 1: Data Merge, EDA & Product Category Prediction

## Overview

This task merges two customer datasets, performs exploratory data analysis (EDA), engineers features, and trains a model to predict which **product category** a customer is likely to purchase.

## Datasets

| Dataset | Shape | Description |
|---|---|---|
| `customer_social_profiles.xlsx` | 155 × 5 | Social media activity, engagement scores, and review sentiment per customer |
| `customer_transactions.xlsx` | 150 × 6 | Purchase history, product categories, amounts, and ratings |

## Steps

### 1. Loading the Datasets
Both Excel files are loaded and inspected for shape, column types, and sample rows before any processing.

### 2. Data Cleaning

| Issue | Dataset | Fix Applied |
|---|---|---|
| 5 duplicate rows | `customer_social_profiles` | Dropped duplicates |
| 10 missing `customer_rating` values | `customer_transactions` | Filled with median |
| Inconsistent customer IDs (`A178` vs `178`) | Both | Stripped `A` prefix and cast to integer |
| Date features | `customer_transactions` | Extracted `purchase_month` and `purchase_dayofweek` |

### 3. Merging the Datasets
An **inner join** on `customer_id` is used — only customers present in both datasets are kept, since both social behaviour and transaction data are needed for prediction.

- **Merged shape:** 213 rows × 12 columns
- **Null values after merge:** None

### 4. Post-Merge Validation
Checks performed:
- Total rows and unique customers
- Duplicate rows (none found)
- Column data types
- Summary statistics

### 5. Exploratory Data Analysis (EDA)

| Plot | Key Finding |
|---|---|
| Target distribution (product categories) | Classes are reasonably balanced — no class imbalance techniques needed |
| Purchase amount distribution + boxplot | Amounts spread uniformly between ~50–500 with no extreme outliers |
| Correlation heatmap | No strong multicollinearity — features are largely independent |
| Engagement score vs. purchase amount by category | Visible patterns across product categories |
| Review sentiment breakdown | Distribution of Positive / Neutral / Negative sentiment |

### 6. Feature Engineering & Encoding
Categorical features (`social_media_platform`, `review_sentiment`) are label-encoded for model compatibility.

### 7. Model Training & Evaluation
Three classifiers are trained and compared:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosting model |

Models are evaluated using **accuracy** and **F1 score**, with confusion matrices visualised for each.

## Output

- `merged_customer_dataset.csv` — the cleaned, merged dataset ready for downstream tasks

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
openpyxl
```

## Notebook

[`Formative2_task1_Data_merge.ipynb`](Formative2_task1_Data_merge.ipynb) — open in Google Colab or Jupyter.
