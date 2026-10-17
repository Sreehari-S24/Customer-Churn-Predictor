# Customer Churn Prediction & Retention Intelligence

A machine learning project that analyzes customer churn, identifies customers at risk of leaving, and explores how predictive analytics can support customer retention strategies.

## Objectives

- Explore customer behavior and churn patterns.
- Clean and prepare customer data for machine learning.
- Train and compare classification models.
- Evaluate predictions using precision, recall, F1-score, and ROC-AUC.
- Explain model predictions using SHAP.
- Develop an interactive dashboard for churn analysis and risk exploration.

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Dashboard:** Streamlit
- **Version Control:** Git and GitHub

## Project Structure

```text
customer-churn-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow

1. Data loading and quality checks
2. Exploratory data analysis
3. Feature engineering and preprocessing
4. Logistic Regression baseline
5. Random Forest model comparison
6. Model evaluation
7. SHAP-based model explanations
8. Interactive retention intelligence dashboard

## Dataset

This project uses the IBM Telco Customer Churn dataset. Obtain the dataset from an authorized public source and place the CSV in `data/raw/`.

The dataset is not committed to this repository.

## Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook in the `notebooks/` folder and run the cells in order.

## Evaluation

Model performance will be reported using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Results will be added after the experiments are finalized.

## Disclaimer

Predicted churn probabilities are estimates, not guarantees that a customer will leave. Model outputs should be evaluated in the context of retention costs and business objectives.
