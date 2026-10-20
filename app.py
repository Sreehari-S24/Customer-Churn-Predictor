
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Retention Intelligence")
st.caption("Explore customer churn patterns and identify customers at risk.")

# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    artifact = load_model()
    model = artifact["model"]
    threshold = artifact["threshold"]
except FileNotFoundError:
    st.error("Model file not found. Run the model-saving notebook cell first.")
    st.stop()

# --------------------------------------------------
# 3. Upload customer data
# --------------------------------------------------
st.header("1. Customer Data")

uploaded_file = st.file_uploader(
    "Upload a customer CSV file",
    type=["csv"],
    help="Upload a CSV with the same input columns as the training dataset."
)

if uploaded_file is None:
    st.info(
        "Please upload a processed/clean CSV file to begin the analysis."
    )
    st.stop()

df = pd.read_csv(uploaded_file)

actual_churn = df["Churn"].copy() if "Churn" in df.columns else None

target_column = "Churn"
id_column = "customerID"

if target_column in df.columns:
    df = df.drop(columns=[target_column])

if id_column in df.columns:
    customer_ids = df[id_column].copy()
    df = df.drop(columns=[id_column])
else:
    customer_ids = pd.Series(
        [f"Customer {i + 1}" for i in range(len(df))]
    )

# --------------------------------------------------
# 4. Validate input columns
# --------------------------------------------------
expected_columns = list(model.feature_names_in_) if hasattr(
    model, "feature_names_in_"
) else None

if expected_columns is None:
    st.error(
        "Could not determine expected input columns from the saved model. "
        "Check the saved pipeline."
    )
    st.stop()

missing_columns = sorted(set(expected_columns) - set(df.columns))
extra_columns = sorted(set(df.columns) - set(expected_columns))

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

if extra_columns:
    st.warning(f"Ignoring extra columns: {extra_columns}")
    df = df[expected_columns]

# Use the same column order as during model training.
df = df[expected_columns]

# --------------------------------------------------
# 5. Predict churn
# --------------------------------------------------
try:
    probabilities = model.predict_proba(df)[:, 1]
except Exception as error:
    st.error(f"Could not generate predictions: {error}")
    st.stop()

predictions = (probabilities >= threshold).astype(int)

results = pd.DataFrame({
    "Customer ID": customer_ids,
    "Churn Probability": probabilities,
    "Predicted Churn": [
        "Likely to Churn" if p == 1 else "Likely to Stay"
        for p in predictions
    ]
})

# --------------------------------------------------
# 6. Interactive churn analysis
# --------------------------------------------------
st.header("3. Customer Churn Analysis")

# Use actual churn labels when available.
# Otherwise, clearly label the analysis as model predictions.
analysis_df = df.copy()

if actual_churn is not None:
    actual_labels = (
        actual_churn.astype(str)
        .str.strip()
        .map({"Yes": 1, "No": 0, "1": 1, "0": 0})
    )

    if actual_labels.notna().all():
        analysis_df["Churn Status"] = actual_labels.map(
            {0: "No Churn", 1: "Churn"}
        ).values
        analysis_label = "Actual"
    else:
        analysis_df["Churn Status"] = results["Predicted Churn"].values
        analysis_label = "Predicted"
else:
    analysis_df["Churn Status"] = results["Predicted Churn"].values
    analysis_label = "Predicted"

chart_col1, chart_col2 = st.columns(2)

# Chart 1: Churn by contract type
with chart_col1:
    st.subheader(f"{analysis_label} Churn by Contract")

    if "Contract" in analysis_df.columns:
        contract_data = (
            analysis_df.groupby("Contract", observed=False)["Churn Status"]
            .apply(lambda s: (s == (
                "Churn" if analysis_label == "Actual"
                else "Likely to Churn"
            )).mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=contract_data,
            x="Contract",
            y="Churn Rate (%)",
            ax=ax
        )
        ax.set_ylabel("Customers who churn (%)")
        ax.set_xlabel("Contract Type")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("The uploaded dataset has no Contract column.")

# Chart 2: Churn by tenure group
with chart_col2:
    st.subheader(f"{analysis_label} Churn by Tenure")

    if "tenure" in analysis_df.columns:
        tenure_data = analysis_df.copy()
        tenure_data["tenure"] = pd.to_numeric(
            tenure_data["tenure"], errors="coerce"
        )

        tenure_data = tenure_data.dropna(subset=["tenure"])

        tenure_data["Tenure Group"] = pd.cut(
            tenure_data["tenure"],
            bins=[-1, 12, 24, 48, float("inf")],
            labels=[
                "0–12 months",
                "13–24 months",
                "25–48 months",
                "49+ months"
            ]
        )

        positive_label = (
            "Churn" if analysis_label == "Actual"
            else "Likely to Churn"
        )

        tenure_summary = (
            tenure_data.groupby(
                "Tenure Group", observed=False
            )["Churn Status"]
            .apply(lambda s: (s == positive_label).mean() * 100)
            .reset_index(name="Churn Rate (%)")
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(
            data=tenure_summary,
            x="Tenure Group",
            y="Churn Rate (%)",
            ax=ax
        )
        ax.set_xlabel("Customer Tenure")
        ax.set_ylabel("Customers who churn (%)")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("The uploaded dataset has no tenure column.")

# Chart 3: Distribution of churn probabilities
st.subheader("Customer Churn Risk Distribution")

fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(
    results["Churn Probability"] * 100,
    bins=20,
    kde=True,
    ax=ax
)
ax.axvline(
    threshold * 100,
    linestyle="--",
    label=f"Decision threshold: {threshold:.0%}"
)
ax.set_xlabel("Predicted Churn Probability (%)")
ax.set_ylabel("Number of Customers")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 7. Key performance indicators
# --------------------------------------------------
st.header("2. Churn Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Customers Analyzed", f"{len(results):,}")
col2.metric(
    "Customers at Risk",
    f"{(predictions == 1).sum():,}"
)
col3.metric(
    "Predicted Churn Rate",
    f"{predictions.mean():.1%}"
)

# --------------------------------------------------
# 8. Visualize prediction distribution
# --------------------------------------------------
st.subheader("Churn Prediction Distribution")

distribution = results["Predicted Churn"].value_counts()

fig, ax = plt.subplots(figsize=(7, 4))
distribution.plot(kind="bar", ax=ax)
ax.set_xlabel("Prediction")
ax.set_ylabel("Number of Customers")
ax.set_title("Predicted Customer Churn")
ax.tick_params(axis="x", rotation=0)
st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# 9. Customer risk table
# --------------------------------------------------
st.subheader("Customer Risk Details")

results["Churn Probability (%)"] = (
    results["Churn Probability"] * 100
).round(2)

st.dataframe(
    results.sort_values(
        "Churn Probability", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 10. Download predictions
# --------------------------------------------------
csv_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predictions as CSV",
    data=csv_data,
    file_name="customer_churn_predictions.csv",
    mime="text/csv"
)
