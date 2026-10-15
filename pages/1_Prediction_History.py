import os
import sys
import pandas as pd
import streamlit as st

# Import functions from mongodb_client.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mongodb_client import (
    get_prediction_history,
    delete_prediction,
    clear_all_predictions,
    get_analytics_summary,
)

st.set_page_config(
    page_title="Prediction History",
    page_icon="",
    layout="wide",
)

st.sidebar.title("🏦 Bank Churn App")
st.sidebar.page_link("app.py", label=" Churn Prediction")
st.sidebar.page_link("pages/1_Prediction_History.py", label=" Prediction History")

st.title(" Prediction History & Analytics")

analytics_tab, history_tab = st.tabs([" Analytics", " Prediction History"])

# Analytics tab
with analytics_tab:
    stats = get_analytics_summary()

    if stats["total_count"] == 0:
        st.info("No prediction records found. Make a prediction first.")
    else:
        st.metric("Total Predictions", stats["total_count"])

        st.subheader("Predictions by Result")
        prediction_data = stats["prediction_breakdown"]

        if prediction_data:
            prediction_df = pd.DataFrame(prediction_data)
            prediction_df = prediction_df.rename(
                columns={
                    "_id": "Prediction",
                    "count": "Count",
                    "avg_probability": "Average Churn Probability",
                }
            )
            if "Average Churn Probability" in prediction_df.columns:
                prediction_df["Average Churn Probability"] = (
                    prediction_df["Average Churn Probability"].fillna(0).map(
                        lambda value: f"{value:.1%}"
                    )
                )
            st.dataframe(prediction_df, use_container_width=True, hide_index=True)

        st.subheader("Predictions by Geography")
        geography_data = stats["geography_breakdown"]

        if geography_data:
            geography_df = pd.DataFrame(geography_data).rename(
                columns={
                    "_id": "Geography",
                    "total": "Total Predictions",
                    "avg_balance": "Average Balance",
                }
            )
            if "Average Balance" in geography_df.columns:
                geography_df["Average Balance"] = (
                    geography_df["Average Balance"].fillna(0).map(
                        lambda value: f"${value:,.2f}"
                    )
                )
            st.dataframe(geography_df, use_container_width=True, hide_index=True)

        st.subheader("Predictions by Model")
        model_data = stats["model_breakdown"]

        if model_data:
            model_df = pd.DataFrame(model_data).rename(
                columns={"_id": "Model Name", "count": "Prediction Count"}
            )
            st.dataframe(model_df, use_container_width=True, hide_index=True)

# Prediction history tab
with history_tab:
    name_filter = st.text_input(
        "Search by customer name",
        placeholder="Enter a customer name...",
    )

    col_refresh, col_clear = st.columns(2)

    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    with col_clear:
        with st.popover("🗑️ Clear All Records"):
            st.warning("This will permanently delete all prediction records.")
            if st.button("Confirm Delete All", type="primary", use_container_width=True):
                deleted = clear_all_predictions()
                st.success(f"Deleted {deleted} records.")
                st.rerun()

    records = get_prediction_history(name_filter)

    if not records:
        st.info("No matching prediction records found.")
    else:
        st.write(f"Found **{len(records)}** prediction(s).")

        for record in records:
            record_id = str(record.get("_id", ""))
            customer = record.get("user_name", "Unknown")
            label = record.get("prediction_label", "N/A")
            probability = record.get("churn_probability", 0)
            model_name = record.get("model_name", "N/A")
            timestamp = record.get("timestamp")
            time_text = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else "N/A"

            with st.expander(f"{customer} — {label}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Churn Probability", f"{probability:.1%}")
                col2.write(f"**Model:** {model_name}")
                col3.write(f"**Date:** {time_text}")

                col1, col2, col3 = st.columns(3)
                col1.write(f"**Credit Score:** {record.get('credit_score', 'N/A')}")
                col2.write(f"**Geography:** {record.get('geography', 'N/A')}")
                col3.write(f"**Gender:** {record.get('gender', 'N/A')}")

                col1, col2, col3 = st.columns(3)
                col1.write(f"**Age:** {record.get('age', 'N/A')}")
                col2.write(f"**Tenure:** {record.get('tenure', 'N/A')} years")
                col3.write(f"**Balance:** ${record.get('balance', 0):,.2f}")

                col1, col2, col3 = st.columns(3)
                col1.write(f"**Products:** {record.get('num_of_products', 'N/A')}")
                col2.write(
                    f"**Credit Card:** {'Yes' if record.get('has_cr_card') == 1 else 'No'}"
                )
                col3.write(
                    f"**Active Member:** {'Yes' if record.get('is_active_member') == 1 else 'No'}"
                )

                if st.button("🗑️ Delete This Record", key=f"delete_{record_id}"):
                    if delete_prediction(record_id):
                        st.success("Prediction record deleted.")
                        st.rerun()
                    else:
                        st.error("Could not delete this record.")
