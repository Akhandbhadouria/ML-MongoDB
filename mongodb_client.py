"""
mongodb_client.py
-----------------
Handles all MongoDB database operations for the Bank Churn Prediction app.
Database  : bank_churn_db
Collection: prediction_history
"""

import os
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_uri() -> str:
    """
    Return the MongoDB connection URI.
    Priority:
      1. Streamlit secrets  (st.secrets["MONGO_URI"])
      2. Environment variable MONGO_URI
      3. Default localhost URI
    """
    try:
        import streamlit as st
        uri = st.secrets.get("MONGO_URI", None)
        if uri:
            return uri
    except Exception:
        pass

    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    return uri


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_db():
    """
    Connect to MongoDB and return the database handle (bank_churn_db).

    Returns:
        pymongo.database.Database  — on success
        None                       — on connection failure (caller handles)
    """
    try:
        uri = _get_uri()
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Trigger connection check
        client.admin.command("ping")
        db = client["bank_churn_db"]
        return db
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        print(f"[mongodb_client] Connection failed: {exc}")
        return None


def save_prediction(record: dict) -> bool:
    """
    Store a prediction record in the prediction_history collection.

    Expected keys in record:
        user_name, credit_score, geography, gender, age, tenure,
        balance, num_of_products, has_cr_card, is_active_member,
        estimated_salary, prediction_label, churn_probability,
        model_name, timestamp (datetime)

    Returns:
        True on success, False on failure.
    """
    db = get_db()
    if db is None:
        return False

    try:
        collection = db["prediction_history"]
        collection.insert_one(record)
        return True
    except Exception as exc:
        print(f"[mongodb_client] save_prediction error: {exc}")
        return False


def get_prediction_history(name_filter: str = "") -> list:
    """
    Retrieve prediction history from MongoDB.

    Args:
        name_filter: Optional partial/full name to filter results.
                     Uses case-insensitive regex match.
                     If empty string, all records are returned.

    Returns:
        List of prediction document dicts, sorted newest-first.
        Returns empty list on connection failure.
    """
    db = get_db()
    if db is None:
        return []

    try:
        collection = db["prediction_history"]
        query = {}
        if name_filter.strip():
            query["user_name"] = {
                "$regex": name_filter.strip(),
                "$options": "i"   # case-insensitive
            }
        cursor = collection.find(query).sort("timestamp", DESCENDING)
        return list(cursor)
    except Exception as exc:
        print(f"[mongodb_client] get_prediction_history error: {exc}")
        return []
