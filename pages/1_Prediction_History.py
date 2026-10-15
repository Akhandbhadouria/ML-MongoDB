"""
pages/1_Prediction_History.py
-----------------------------
Page 2 of the Streamlit app — displays prediction history retrieved
directly from MongoDB, with optional name-based filtering.
"""

import os
import sys
import streamlit as st
from datetime import timezone

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mongodb_client import get_prediction_history

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediction History | Bank Churn",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏦 Bank Churn App")
st.sidebar.markdown("**Navigate**")
st.sidebar.page_link("app.py",                          label="🔮 Churn Prediction")
st.sidebar.page_link("pages/1_Prediction_History.py",   label="📋 Prediction History")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📋 Prediction History")
st.markdown(
    "All prediction records stored in **MongoDB** are displayed below. "
    "Use the name filter to search for a specific user's predictions."
)

st.divider()

# ---------------------------------------------------------------------------
# Name filter
# ---------------------------------------------------------------------------
col_filter, col_btn = st.columns([3, 1])
with col_filter:
    name_filter = st.text_input(
        "🔍 Filter by Name",
        placeholder="Enter a name to search …",
        label_visibility="collapsed",
    )
with col_btn:
    refresh = st.button("🔄 Refresh", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Fetch from MongoDB
# ---------------------------------------------------------------------------
records = get_prediction_history(name_filter=name_filter)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if not records:
    if name_filter.strip():
        st.info(f"🔍 No predictions found for **\"{name_filter}\"**.")
    else:
        st.info("📭 No prediction records found in MongoDB yet.")
else:
    count_label = f"**{len(records)}** record{'s' if len(records) != 1 else ''} found"
    if name_filter.strip():
        count_label += f" for **\"{name_filter}\"**"
    st.markdown(count_label)

    for doc in records:
        # Format timestamp
        ts = doc.get("timestamp")
        if ts:
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        else:
            ts_str = "N/A"

        label     = doc.get("prediction_label", "N/A")
        prob      = doc.get("churn_probability", 0.0)
        user      = doc.get("user_name", "Unknown")
        model_nm  = doc.get("model_name", "N/A")

        # High contrast card colors with dark text/clear badges
        if label == "Likely to Churn":
            card_icon = "🚨"
            card_bg = "#2a1515"
            border_color = "#ff4d4d"
            badge_bg = "#ff4d4d"
            badge_text = "#ffffff"
            text_main = "#ffffff"
            text_sub = "#fca5a5"
        else:
            card_icon = "✅"
            card_bg = "#112918"
            border_color = "#22c55e"
            badge_bg = "#22c55e"
            badge_text = "#ffffff"
            text_main = "#ffffff"
            text_sub = "#86efac"

        with st.container():
            st.markdown(
                f"""
                <div style="
                    border-left: 6px solid {border_color};
                    background: {card_bg};
                    padding: 16px 20px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
                    display: flex;
                    flex-direction: column;
                    gap: 8px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 700; color: {text_main};">
                            {card_icon} {user}
                        </span>
                        <span style="
                            background-color: {badge_bg};
                            color: {badge_text};
                            padding: 4px 12px;
                            border-radius: 20px;
                            font-weight: 700;
                            font-size: 0.9rem;
                            letter-spacing: 0.5px;
                        ">
                            {label.upper()}
                        </span>
                    </div>
                    <div style="font-size: 0.92rem; color: {text_sub}; display: flex; flex-wrap: wrap; gap: 16px;">
                        <span>📊 Churn Probability: <strong style="color: #ffffff;">{prob*100:.1f}%</strong></span>
                        <span>🤖 Model: <strong style="color: #ffffff;">{model_nm}</strong></span>
                        <span>🕒 <strong style="color: #ffffff;">{ts_str}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Expandable details
        with st.expander(f"📄 Full details for {user} — {ts_str}"):
            detail_cols = st.columns(3)
            detail_cols[0].metric("Credit Score",       doc.get("credit_score", "N/A"))
            detail_cols[1].metric("Geography",          doc.get("geography", "N/A"))
            detail_cols[2].metric("Gender",             doc.get("gender", "N/A"))

            detail_cols2 = st.columns(3)
            detail_cols2[0].metric("Age",               doc.get("age", "N/A"))
            detail_cols2[1].metric("Tenure (yrs)",      doc.get("tenure", "N/A"))
            detail_cols2[2].metric("Balance ($)",       f"{doc.get('balance', 0):,.2f}")

            detail_cols3 = st.columns(3)
            detail_cols3[0].metric("# Products",        doc.get("num_of_products", "N/A"))
            detail_cols3[1].metric("Credit Card",
                                   "Yes" if doc.get("has_cr_card") == 1 else "No")
            detail_cols3[2].metric("Active Member",
                                   "Yes" if doc.get("is_active_member") == 1 else "No")

            detail_cols4 = st.columns(2)
            detail_cols4[0].metric("Est. Salary ($)",  f"{doc.get('estimated_salary', 0):,.2f}")
            detail_cols4[1].metric("Churn Probability", f"{prob*100:.1f}%")
