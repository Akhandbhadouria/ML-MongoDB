"""
train_model.py — Run this to create a fresh model.pkl
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
df = pd.read_csv("Bank_Churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
df = df.dropna()

X = df.drop(columns=["Exited"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_cols     = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric",      StandardScaler(),                       numeric_cols),
    ("categorical",  OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier",    RandomForestClassifier(n_estimators=100, random_state=42)),
])

print("Training model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

joblib.dump(model, "model.pkl")
print("✅ model.pkl saved successfully!")

