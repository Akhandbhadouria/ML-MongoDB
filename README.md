# 🏦 Bank Customer Churn Prediction

A simple, end-to-end **Bank Customer Churn Prediction** application built with
**Streamlit**, **MongoDB (PyMongo)**, and a **Random Forest** machine-learning model.

---

## 📌 Project Overview

| Item | Detail |
|------|--------|
| **Goal** | Predict whether a bank customer will churn (leave) |
| **ML Model** | Random Forest Classifier (scikit-learn) |
| **Database** | MongoDB — `bank_churn_db.prediction_history` |
| **Frontend** | Streamlit (2 pages) |
| **Serialization** | Joblib (`model.pkl`) |

---

## 📁 Project Structure

```
bank-churn-mongodb/
├── app.py                         # Page 1 — Churn Prediction form
├── pages/
│   └── 1_Prediction_History.py   # Page 2 — History with name filter
├── mongodb_client.py              # MongoDB CRUD operations
├── train_model.py                 # Model training & evaluation
├── model.pkl                      # Saved trained pipeline (auto-generated)
├── Bank_Churn.csv                 # Source dataset
├── requirements.txt               # Python dependencies
├── .gitignore                     # Files excluded from Git
├── .streamlit/
│   ├── secrets.toml               # MongoDB URI (not committed to Git)
│   └── secrets.toml.example       # Example configuration
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

- Python 3.9+
- MongoDB installed locally **or** a MongoDB Atlas account

### 2. Clone / Open the Project

```bash
cd bank-churn-mongodb
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure MongoDB

Copy the example secrets file and update the URI:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
# Local MongoDB
MONGO_URI = "mongodb://localhost:27017/"

# MongoDB Atlas (replace with your actual URI)
# MONGO_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
```

> **Never commit `secrets.toml` to Git.** It is already listed in `.gitignore`.

### 6. Start MongoDB (Local)

```bash
mongod --dbpath /usr/local/var/mongodb   # macOS with Homebrew
# or
brew services start mongodb-community    # macOS service
```

### 7. Train the Model

Run this **once** before starting the Streamlit app:

```bash
python train_model.py
```

This creates `model.pkl` in the project directory.

### 8. Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## 🖥️ Application Pages

### Page 1 — Customer Churn Prediction (`app.py`)

Fill in the customer form:

| Field | Type |
|-------|------|
| Your Name | Text |
| Credit Score | Number (300–850) |
| Geography | Dropdown: France / Germany / Spain |
| Gender | Dropdown: Male / Female |
| Age | Number (18–100) |
| Tenure | Number (0–10 years) |
| Balance | Number |
| Number of Products | Number (1–4) |
| Has Credit Card | Yes / No |
| Active Member | Yes / No |
| Estimated Salary | Number |

Click **Predict Churn** → the app:
1. Passes features to the trained Random Forest pipeline.
2. Displays **"Likely to Churn"** or **"Likely to Stay"** with probability.
3. Saves the record to MongoDB (`bank_churn_db.prediction_history`).

---

### Page 2 — Prediction History (`pages/1_Prediction_History.py`)

- Fetches all records **directly from MongoDB** (not from session state).
- Provides a **name-based filter** using a case-insensitive regex query.
- Displays: Name · Result · Probability · Model · Timestamp.
- Expandable card shows all input features.
- Records are sorted **newest-first**.

---

## 🗄️ MongoDB Integration

### Database & Collection

| Setting | Value |
|---------|-------|
| Database | `bank_churn_db` |
| Collection | `prediction_history` |

### Document Schema

```json
{
  "_id":               "ObjectId (auto)",
  "user_name":         "John Smith",
  "credit_score":      650,
  "geography":         "France",
  "gender":            "Male",
  "age":               35,
  "tenure":            5,
  "balance":           50000.0,
  "num_of_products":   1,
  "has_cr_card":       1,
  "is_active_member":  1,
  "estimated_salary":  80000.0,
  "prediction_label":  "Likely to Stay",
  "churn_probability": 0.12,
  "model_name":        "Random Forest",
  "timestamp":         "2026-10-15T07:30:00Z"
}
```

### `mongodb_client.py` Functions

| Function | Purpose |
|----------|---------|
| `get_db()` | Connects to MongoDB, returns database handle |
| `save_prediction(record)` | Inserts one prediction document |
| `get_prediction_history(name_filter)` | Retrieves records, optionally filtered by name (regex, case-insensitive), sorted newest-first |

---

## 🤖 Machine Learning Details

### Dataset

- **Source**: Bank_Churn.csv  
- **Rows**: 10,000 customers  
- **Target**: `Exited` (1 = Churned, 0 = Stayed)  
- **Class balance**: 79.6% Stay / 20.4% Churn

### Features Used

| Feature | Type |
|---------|------|
| CreditScore | Numeric |
| Geography | Categorical |
| Gender | Categorical |
| Age | Numeric |
| Tenure | Numeric |
| Balance | Numeric |
| NumOfProducts | Numeric |
| HasCrCard | Numeric (binary) |
| IsActiveMember | Numeric (binary) |
| EstimatedSalary | Numeric |

### Preprocessing

| Step | Details |
|------|---------|
| Numeric features | `StandardScaler` |
| Categorical features | `OneHotEncoder` (handle\_unknown="ignore") |

> Preprocessing is embedded inside an sklearn `Pipeline`, ensuring the **same
> transformations** are applied during both training and inference.

### Model

- **Algorithm**: `RandomForestClassifier(random_state=42)`
- **Default parameters** (no hyperparameter tuning)
- **Train/Test split**: 80% / 20% stratified

---

## 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | **86.00%** |
| **R² Score** | 0.1363 |
| **MAE** | 0.1400 |
| **Precision** | 0.7702 |
| **Recall** | 0.4447 |
| **F1 Score** | 0.5639 |

### Classification Report

```
              precision    recall  f1-score   support
   Stay (0)       0.87      0.97      0.92      1593
  Churn (1)       0.77      0.44      0.56       407
    accuracy                           0.86      2000
   macro avg       0.82      0.71      0.74      2000
weighted avg       0.85      0.86      0.84      2000
```

### Confusion Matrix

|  | Predicted Stay | Predicted Churn |
|--|---------------|----------------|
| **Actual Stay** | TN = 1539 | FP = 54 |
| **Actual Churn** | FN = 226 | TP = 181 |

---

## 📦 Dependencies

```
streamlit
pymongo
pandas
scikit-learn
joblib
```

---

## 🔄 How the Application Works (Flow)

```
User fills form (Page 1)
         │
         ▼
   app.py collects inputs
         │
         ▼
   pandas DataFrame (same column order as training)
         │
         ▼
   pipeline.predict() + pipeline.predict_proba()
         │
         ├──► Display result on screen
         │
         └──► mongodb_client.save_prediction(record)
                       │
                       ▼
              MongoDB: bank_churn_db.prediction_history


User opens Page 2
         │
         ▼
   User types a name filter (optional)
         │
         ▼
   mongodb_client.get_prediction_history(name_filter)
         │
         ▼
   PyMongo query: {"user_name": {"$regex": ..., "$options": "i"}}
   .sort("timestamp", DESCENDING)
         │
         ▼
   Display records (newest first) with expandable details
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `model.pkl not found` | Run `python train_model.py` first |
| `MongoDB connection failed` | Ensure `mongod` is running; check `secrets.toml` URI |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| App won't start | Confirm you are inside the `bank-churn-mongodb/` directory |
