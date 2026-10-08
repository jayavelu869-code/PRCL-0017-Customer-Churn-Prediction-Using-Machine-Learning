# PRCL-0017: Customer Churn Prediction Using Machine Learning

An end-to-end machine learning project that predicts which telecom customers are likely to leave (churn), identifies the main factors behind churn, and suggests ways to retain them.

> **Note:** This is a training/internship project based on a **simulated** telecom dataset for a fictional company ("No-Churn Telecom"). It does not use real customer data.

---

## Business Problem

No-Churn Telecom is losing customers to competitors. The goal is to:

1. Understand which factors are associated with churn
2. Build a model that gives each customer a **churn risk score**
3. Create a **CHURN_FLAG** so the retention team can target the right customers with offers

---

## Dataset

- **Rows:** 243,553 customers
- **Source:** SQL database (MySQL), loaded with SQLAlchemy + PyMySQL
- **Target:** `churn` (1 = customer left, 0 = customer stayed)
- **Class balance:** about 80% stayed, 20% churned (imbalanced)

| Column | Description |
|---|---|
| customer_id | Unique customer ID (dropped before modeling) |
| telecom_partner | Telecom provider of the customer |
| gender, age | Demographics |
| state, city, pincode | Location |
| date_of_registration | Used to create `tenure_days` |
| num_dependents | Number of people who depend on the customer |
| estimated_salary | Estimated income |
| calls_made, sms_sent, data_used | Usage behavior |
| churn | Target variable |

---

## Project Workflow

1. Connect to the SQL database and load the data
2. Data checks: shape, data types, missing values, duplicates
3. Target distribution analysis (class imbalance)
4. Preprocessing: drop ID, encode categorical variables, create `tenure_days`
5. Correlation analysis
6. Train/test split (80/20, stratified)
7. Feature scaling with `StandardScaler`
8. Handle class imbalance with **SMOTE**
9. Train **Logistic Regression** (baseline) and **Random Forest**
10. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curve
11. Feature importance analysis
12. Generate `CHURN_FLAG` and `CHURN_RISK_SCORE` for every customer
13. Save predictions and trained model

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| Random Forest | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

**Best model:** _fill in after checking your comparison table_

### Key Findings

- About **20%** of customers churned, so the data is imbalanced and SMOTE was used on the training set.
- The most important feature for predicting churn was **`num_dependents`**, followed by `age`, `sms_sent` and `estimated_salary`.
- Individual correlations with churn were weak, so a non-linear model (Random Forest) was needed to capture patterns.
- Feature importance shows **general patterns across all customers**, not the exact reason any one customer left.

---

## Retention Recommendations

- Offer **family bundle / shared data plans** to customers with more dependents, who are likely more price-sensitive.
- Use the **churn risk score** to contact high-risk customers with a targeted offer *before* they switch providers.
- Review pricing for the age and income groups that show higher churn risk.

---

## Challenges Faced

- Column names in the real dataset did not match the project document, so the pipeline had to be rewritten using the actual columns
- Class imbalance (80/20)
- High-cardinality categorical columns (state, city) creating many encoded features
- Weak individual correlations
- Long training time on a large dataset
- Trade-off between model interpretability and performance

---

## Future Improvements

- Hyperparameter tuning with GridSearchCV / RandomizedSearchCV
- Richer EDA: box plots for outliers and churn-vs-feature comparison charts
- Check and clean data quality issues (for example, negative `data_used` values)
- Try gradient boosting models (XGBoost, LightGBM)
- Link feature importance more directly to specific retention strategies

---

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn (SMOTE), SQLAlchemy, PyMySQL, Joblib, Jupyter Notebook

---

## How to Run

1. Clone the repository
2. Install the requirements:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn sqlalchemy pymysql joblib
   ```
3. Open the notebook in Jupyter
4. When the database cell runs, enter the connection details when prompted. **Credentials are never stored in this repository.**

You can also set them as environment variables: `DB_USER`, `DB_PASSWORD`, `DB_HOST`.

---

## Author

**Your Name**
LinkedIn: _add your link_
GitHub: _add your link_
