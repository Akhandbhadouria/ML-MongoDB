import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

from mongodb_client import save_prediction


# Page settings
st.set_page_config(
    page_title="Bank Churn Prediction",
    page_icon="",
    layout="centered"
)


# Load trained models
MODEL_PATH = "model.pkl"

# Backward compatibility for models pickled in different scikit-learn versions
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass

@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH):
        saved = joblib.load(MODEL_PATH)
        if isinstance(saved, dict):
            if "pipeline" in saved:
                # Legacy single model structure
                name = saved.get("meta", {}).get("model_name", "Random Forest")
                return {name: saved["pipeline"]}
            return saved
    return None


models = load_models()


# Sidebar
st.sidebar.title(" Bank Churn App")
st.sidebar.page_link("app.py", label=" Churn Prediction")
st.sidebar.page_link(
    "pages/1_Prediction_History.py",
    label=" Prediction History"
)

if models:
    st.sidebar.divider()
    st.sidebar.markdown("**🤖 Available Models**")
    for m in models.keys():
        st.sidebar.write(f"- {m}")


# Main page
st.title("Customer Churn Prediction")
st.write("Enter customer details, select an ML model, and predict whether they may leave the bank.")

if not models:
    st.error("No models found. Please run the training notebook first.")
    st.stop()


# Customer input form
with st.form("prediction_form"):
    st.subheader("Customer Information")

    col_name, col_model = st.columns(2)
    with col_name:
        user_name = st.text_input("Your Name", placeholder="Enter your name")
    with col_model:
        selected_model_name = st.selectbox(
            "Choose ML Model",
            options=list(models.keys()),
            help="Select which algorithm to use for this prediction."
        )

    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score", 300, 850, 650
        )
        age = st.number_input("Age", 18, 100, 35)
        balance = st.number_input(
            "Balance ($)", min_value=0.0, value=50000.0
        )
        has_cr_card = st.selectbox(
            "Has Credit Card?", [1, 0],
            format_func=lambda x: "Yes" if x else "No"
        )
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )

    with col2:
        tenure = st.number_input("Tenure (years)", 0, 10, 5)
        num_of_products = st.number_input(
            "Number of Products", 1, 4, 1
        )
        estimated_salary = st.number_input(
            "Estimated Salary ($)", min_value=0.0, value=80000.0
        )
        is_active_member = st.selectbox(
            "Active Member?", [1, 0],
            format_func=lambda x: "Yes" if x else "No"
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    submitted = st.form_submit_button(
        "Predict Churn", use_container_width=True
    )


# Make prediction
if submitted:
    if not user_name.strip():
        st.warning("Please enter your name.")
        st.stop()

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }])

    # Use the chosen model pipeline
    chosen_pipeline = models[selected_model_name]

    prediction = chosen_pipeline.predict(input_data)[0]
    probability = chosen_pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        prediction_label = "Likely to Churn"
        st.error(" Customer is likely to churn.")
    else:
        prediction_label = "Likely to Stay"
        st.success(" Customer is likely to stay.")

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.metric("Prediction", prediction_label)
    st.caption(f"Algorithm used: **{selected_model_name}**")

    # Save prediction to MongoDB with selected model name
    record = {
        "user_name": user_name.strip(),
        "credit_score": int(credit_score),
        "geography": geography,
        "gender": gender,
        "age": int(age),
        "tenure": int(tenure),
        "balance": float(balance),
        "num_of_products": int(num_of_products),
        "has_cr_card": int(has_cr_card),
        "is_active_member": int(is_active_member),
        "estimated_salary": float(estimated_salary),
        "prediction_label": prediction_label,
        "churn_probability": round(float(probability), 4),
        "model_name": selected_model_name,
        "timestamp": datetime.utcnow()
    }

    try:
        save_prediction(record)
        st.info(f"Prediction saved to MongoDB (Model: {selected_model_name}).")
    except Exception as e:
        st.warning(f"Could not save prediction to MongoDB: {e}")

    with st.expander("View Customer Details"):
        st.dataframe(input_data, use_container_width=True)
