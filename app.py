"""
app.py
------
Main Streamlit application — Page 1: Customer Churn Prediction.

Loads the trained Random Forest pipeline (model.pkl) and lets the user
enter customer details, get a churn prediction, and store the result
in MongoDB via mongodb_client.py.
"""

import os
import sys
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))
from mongodb_client import save_prediction

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model (cached so it's only read once per session)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

@st.cache_resource
def load_model():
    """Load the saved pipeline and metadata from model.pkl."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    artifact = joblib.load(MODEL_PATH)
    return artifact["pipeline"], artifact["meta"]


pipeline, meta = load_model()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Bank Churn App")
st.sidebar.markdown("**Navigate**")
st.sidebar.page_link("app.py",                          label="🔮 Churn Prediction")
st.sidebar.page_link("pages/1_Prediction_History.py",   label="📋 Prediction History")

if meta:
    st.sidebar.divider()
    st.sidebar.markdown("**Model Info**")
    st.sidebar.markdown(f"Model: **{meta['model_name']}**")
    m = meta["metrics"]
    st.sidebar.markdown(f"Accuracy: `{m['accuracy']*100:.2f}%`")
    st.sidebar.markdown(f"F1 Score: `{m['f1']:.4f}`")
    st.sidebar.markdown(f"Precision: `{m['precision']:.4f}`")
    st.sidebar.markdown(f"Recall: `{m['recall']:.4f}`")
    st.sidebar.markdown(f"R² Score: `{m['r2_score']:.4f}`")
    st.sidebar.markdown(f"MAE: `{m['mae']:.4f}`")

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🔮 Customer Churn Prediction")
st.markdown("Enter customer details below and click **Predict Churn** to get an instant prediction.")

if pipeline is None:
    st.error(
        "⚠️ **Model not found!** Please run the training script first:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("prediction_form", clear_on_submit=False):
    st.subheader("👤 Customer Information")

    user_name = st.text_input(
        "Your Name (for record-keeping)",
        placeholder="e.g. John Smith",
        help="This name is saved with the prediction in MongoDB.",
    )

    st.subheader("📊 Customer Features")

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",       min_value=300, max_value=850, value=650, step=1)
        age          = st.number_input("Age",                min_value=18,  max_value=100, value=35,  step=1)
        balance      = st.number_input("Balance ($)",        min_value=0.0, max_value=300000.0, value=50000.0, step=100.0)
        has_cr_card  = st.selectbox("Has Credit Card",       options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        geography    = st.selectbox("Geography",             options=["France", "Germany", "Spain"])

    with col2:
        tenure           = st.number_input("Tenure (years)",    min_value=0,   max_value=10,  value=5,   step=1)
        num_of_products  = st.number_input("Number of Products",min_value=1,   max_value=4,   value=1,   step=1)
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=80000.0, step=500.0)
        is_active_member = st.selectbox("Active Member",        options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        gender           = st.selectbox("Gender",               options=["Male", "Female"])

    st.divider()
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------
if submitted:
    # Validate name
    if not user_name.strip():
        st.warning("⚠️ Please enter your name before predicting.")
        st.stop()

    # Build input DataFrame — same columns as training
    input_data = pd.DataFrame([{
        "CreditScore":     credit_score,
        "Geography":       geography,
        "Gender":          gender,
        "Age":             age,
        "Tenure":          tenure,
        "Balance":         balance,
        "NumOfProducts":   num_of_products,
        "HasCrCard":       has_cr_card,
        "IsActiveMember":  is_active_member,
        "EstimatedSalary": estimated_salary,
    }])

    # Predict
    prediction       = pipeline.predict(input_data)[0]          # 0 or 1
    churn_probability = pipeline.predict_proba(input_data)[0][1] # float

    prediction_label = "Likely to Churn" if prediction == 1 else "Likely to Stay"

    # Display result
    st.divider()
    st.subheader("📢 Prediction Result")

    if prediction == 1:
        st.error(f"### 🚨 {prediction_label}")
    else:
        st.success(f"### ✅ {prediction_label}")

    col_a, col_b = st.columns(2)
    col_a.metric("Churn Probability", f"{churn_probability*100:.1f}%")
    col_b.metric("Prediction", prediction_label)

    # Build MongoDB record
    record = {
        "user_name":         user_name.strip(),
        "credit_score":      int(credit_score),
        "geography":         geography,
        "gender":            gender,
        "age":               int(age),
        "tenure":            int(tenure),
        "balance":           float(balance),
        "num_of_products":   int(num_of_products),
        "has_cr_card":       int(has_cr_card),
        "is_active_member":  int(is_active_member),
        "estimated_salary":  float(estimated_salary),
        "prediction_label":  prediction_label,
        "churn_probability": round(float(churn_probability), 4),
        "model_name":        meta["model_name"],
        "timestamp":         datetime.utcnow(),
    }

    # Save to MongoDB
    saved = save_prediction(record)
    if saved:
        st.info("💾 Prediction saved to MongoDB successfully.")
    else:
        st.warning(
            "⚠️ Could not save to MongoDB. "
            "Check your connection settings in `.streamlit/secrets.toml`."
        )

    st.divider()
    with st.expander("📄 View Input Features"):
        st.dataframe(input_data, use_container_width=True)
