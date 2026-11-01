# 🏦 Bank Customer Churn Prediction

A **minimal, end-to-end** ML + MongoDB project that predicts whether a bank customer will churn.  
Built with **Python**, **Scikit-learn**, and **PyMongo** — no Streamlit, no extra pages.

---

## 📁 Folder Structure

```
bank-churn-mongodb/
├── Bank_Churn.csv       # Source dataset (10,000 customers)
├── train_model.py       # Preprocess + train Random Forest → saves model.pkl
├── main.py              # predict_churn() + all MongoDB operations demo
├── model.pkl            # Saved sklearn pipeline (auto-generated)
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- MongoDB running locally **or** a [MongoDB Atlas](https://www.mongodb.com/atlas) account

### 2. Create & activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start MongoDB (local)
```bash
brew services start mongodb-community   # macOS (Homebrew)
# or
mongod --dbpath /usr/local/var/mongodb
```

> To use **MongoDB Atlas** instead, set the `MONGO_URI` environment variable:
> ```bash
> export MONGO_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
> ```

### 5. Train the model
```bash
python train_model.py
```
This reads `Bank_Churn.csv`, trains a Random Forest pipeline, and saves `model.pkl`.

### 6. Run the demo
```bash
python main.py
```
This loads the model, makes 3 sample predictions, stores them in MongoDB, then demonstrates every MongoDB operation.

---

## 🤖 Machine Learning

### Dataset
| Property | Value |
|----------|-------|
| File | `Bank_Churn.csv` |
| Rows | 10,000 customers |
| Target | `Exited` (1 = Churned, 0 = Stayed) |
| Churn rate | ~20.4% |

### Features Used
| Feature | Type |
|---------|------|
| CreditScore | Numeric |
| Geography | Categorical (France / Germany / Spain) |
| Gender | Categorical (Male / Female) |
| Age | Numeric |
| Tenure | Numeric |
| Balance | Numeric |
| NumOfProducts | Numeric |
| HasCrCard | Binary (1/0) |
| IsActiveMember | Binary (1/0) |
| EstimatedSalary | Numeric |

### Preprocessing (inside the pipeline)
| Step | Transformer |
|------|-------------|
| Numeric features | `StandardScaler` |
| Categorical features | `OneHotEncoder(handle_unknown="ignore")` |

> Preprocessing is embedded in an sklearn `Pipeline`, so the **exact same transformations** are applied during both training and prediction — no data leakage possible.

### Model & Evaluation
| Metric | Value |
|--------|-------|
| Algorithm | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| Accuracy | **86.00%** |
| Precision (Churn) | 0.77 |
| Recall (Churn) | 0.44 |
| F1 (Churn) | 0.56 |

---

## 🗄️ MongoDB Integration

### Database
| Setting | Value |
|---------|-------|
| Database | `bank_churn_db` |
| Collection | `prediction_history` |

### Document Schema
```json
{
  "_id":               "ObjectId (auto)",
  "customer_name":     "Alice",
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
  "prediction":        0,
  "prediction_label":  "Likely to Stay",
  "churn_probability": 0.12,
  "model_name":        "Random Forest",
  "timestamp":         "2026-10-29T07:47:00Z"
}
```

### Operations Demonstrated in `main.py`

| Operation | Function | PyMongo Method |
|-----------|----------|----------------|
| Insert record | `predict_churn()` | `insert_one()` |
| Get all records | `get_all_predictions()` | `find()` + `sort()` |
| Get one record | `get_prediction_by_id()` | `find_one()` |
| Update a record | `update_prediction_note()` | `update_one()` |
| Delete a record | `delete_prediction()` | `delete_one()` |
| Count records | `count_predictions()` | `count_documents()` |
| Filter records | `get_churn_predictions()` | `find({ filter })` |
| Analytics | `get_analytics()` | `aggregate()` |

---

## 📋 Sample MongoDB Shell Queries

Run these in **mongosh** (or MongoDB Compass) after running `python main.py`:

```javascript
// Select the database
use bank_churn_db

// ── insertOne ──────────────────────────────────────────────────────────────
// (done automatically by predict_churn() in main.py)

// ── find all records ───────────────────────────────────────────────────────
db.prediction_history.find()

// ── find with filter: only churn predictions ───────────────────────────────
db.prediction_history.find({ prediction_label: "Likely to Churn" })

// ── find with filter + projection (show only name, label, probability) ─────
db.prediction_history.find(
  { prediction_label: "Likely to Churn" },
  { customer_name: 1, prediction_label: 1, churn_probability: 1 }
)

// ── find with regex filter (case-insensitive name search) ──────────────────
db.prediction_history.find({ customer_name: { $regex: "ali", $options: "i" } })

// ── findOne ────────────────────────────────────────────────────────────────
db.prediction_history.findOne({ geography: "Germany" })

// ── sorting: highest churn probability first ───────────────────────────────
db.prediction_history.find().sort({ churn_probability: -1 })

// ── updateOne: add analyst note ────────────────────────────────────────────
db.prediction_history.updateOne(
  { customer_name: "Bob" },
  { $set: { note: "Needs retention offer" } }
)

// ── countDocuments ─────────────────────────────────────────────────────────
db.prediction_history.countDocuments()                              // all
db.prediction_history.countDocuments({ prediction_label: "Likely to Churn" })  // churn only

// ── aggregation: count & avg probability per label ─────────────────────────
db.prediction_history.aggregate([
  {
    $group: {
      _id: "$prediction_label",
      count: { $sum: 1 },
      avg_probability: { $avg: "$churn_probability" }
    }
  }
])

// ── aggregation: breakdown by geography ────────────────────────────────────
db.prediction_history.aggregate([
  {
    $group: {
      _id: "$geography",
      total: { $sum: 1 },
      avg_balance: { $avg: "$balance" },
      churn_count: { $sum: "$prediction" }
    }
  },
  { $sort: { total: -1 } }
])

// ── deleteOne ──────────────────────────────────────────────────────────────
db.prediction_history.deleteOne({ customer_name: "Carol" })
```

---

## 📦 Dependencies

```
pandas
scikit-learn
pymongo
joblib
```

---

## 🔄 How It Works

```
Bank_Churn.csv
      │
      ▼
train_model.py
  ├── Drop CustomerId, Surname
  ├── StandardScaler  (numeric)
  ├── OneHotEncoder   (categorical)
  ├── RandomForestClassifier
  └── Saves → model.pkl
      │
      ▼
main.py
  ├── Load model.pkl
  ├── Connect to MongoDB
  └── predict_churn(customer details)
        ├── Build DataFrame
        ├── pipeline.predict() + predict_proba()
        ├── insertOne → MongoDB
        └── Return { label, probability, inserted_id }
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `model.pkl not found` | Run `python train_model.py` first |
| `Cannot connect to MongoDB` | Start MongoDB with `brew services start mongodb-community` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Bank_Churn.csv not found` | Make sure the CSV is in the same directory as `train_model.py` |
