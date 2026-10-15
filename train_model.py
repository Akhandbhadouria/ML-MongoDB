"""
train_model.py
--------------
Trains a Random Forest classifier on the Bank Churn dataset and saves
the trained pipeline to model.pkl using Joblib.

Run once before starting the Streamlit app:
    python train_model.py
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Bank_Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

NUMERIC_FEATURES     = ["CreditScore", "Age", "Tenure", "Balance",
                         "NumOfProducts", "HasCrCard", "IsActiveMember",
                         "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
TARGET_COL           = "Exited"
MODEL_NAME           = "Random Forest"


def load_data(path: str) -> pd.DataFrame:
    """Load and minimally preprocess the dataset."""
    df = pd.read_csv(path)
    # Drop columns not used for training
    df = df.drop(columns=["CustomerId", "Surname"], errors="ignore")
    df = df.dropna()
    return df


def build_pipeline() -> Pipeline:
    """Build sklearn Pipeline with preprocessing + Random Forest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(random_state=42)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])
    return pipeline


def train():
    """Train the pipeline, evaluate, and save to model.pkl."""
    print("=" * 60)
    print("  Bank Customer Churn — Model Training")
    print("=" * 60)

    # ------ Load data ------
    print(f"\n[1/4] Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    print(f"      Rows: {len(df)} | Churned: {df[TARGET_COL].sum()} "
          f"({df[TARGET_COL].mean()*100:.1f}%)")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ------ Split ------
    print("\n[2/4] Splitting dataset (80/20 train-test split)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train)} | Test: {len(X_test)}")

    # ------ Train ------
    print("\n[3/4] Training Random Forest classifier …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Training complete.")

    # ------ Evaluate ------
    print("\n[4/4] Evaluation Metrics")
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    r2        = r2_score(y_test, y_pred)
    mae       = mean_absolute_error(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  R² Score  : {r2:.4f}")
    print(f"  MAE       : {mae:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Stay (0)", "Churn (1)"]))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")

    # ------ Save ------
    meta = {
        "model_name"   : MODEL_NAME,
        "feature_cols" : FEATURE_COLS,
        "metrics": {
            "accuracy"  : round(acc, 4),
            "r2_score"  : round(r2, 4),
            "mae"       : round(mae, 4),
            "precision" : round(precision, 4),
            "recall"    : round(recall, 4),
            "f1"        : round(f1, 4),
        },
    }

    joblib.dump({"pipeline": pipeline, "meta": meta}, MODEL_PATH)
    print(f"\n  ✔ Model saved to: {MODEL_PATH}")
    print("=" * 60)

    return pipeline, meta


if __name__ == "__main__":
    train()
