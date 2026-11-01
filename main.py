import json
from datetime import datetime
import joblib
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

client = MongoClient("mongodb://localhost:27017/")
db = client["bank_churn_db"]
records = db["prediction_history"]

model = joblib.load("model.pkl")


def predict_customer():
    print("\nExample input:")
    print("Name, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary")
    print("Rahul, 650, France, Male, 35, 5, 50000, 1, 1, 1, 80000")

    data = input().split(",")

    if len(data) != 11:
        print("Please enter exactly 11 values.")
        return

    try:
        name = data[0].strip()
        score = int(data[1])
        country = data[2].strip().title()
        gender = data[3].strip().title()
        age = int(data[4])
        tenure = int(data[5])
        balance = float(data[6])
        products = int(data[7])
        card = int(data[8])
        active = int(data[9])
        salary = float(data[10])

        customer = pd.DataFrame([{
            "CreditScore": score,
            "Geography": country,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": card,
            "IsActiveMember": active,
            "EstimatedSalary": salary
        }])

        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])
        label = "Likely to Churn" if prediction == 1 else "Likely to Stay"

        record = {
            "name": name,
            "credit_score": score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products": products,
            "has_card": card,
            "active_member": active,
            "salary": salary,
            "prediction": prediction,
            "result": label,
            "probability": round(probability, 4),
            "time": datetime.now()
        }

        saved = records.insert_one(record)

        print("\n--- Prediction Result ---")
        print("Customer:", name)
        print("Prediction:", label)
        print("Churn probability:", f"{probability:.2%}")
        print("Saved MongoDB ID:", saved.inserted_id)

    except ValueError:
        print("Invalid input.")


def show_all():
    docs = list(records.find().sort("_id", -1).limit(10))
    if not docs:
        print("No records found.")
    for doc in docs:
        print(json.dumps(doc, default=str, indent=2))


def find_one():
    print("\nExample ID: 665f1a2b3c4d5e6f78901234 (use a real ID from your records)")
    record_id = input("Enter MongoDB record ID: ").strip()
    try:
        doc = records.find_one({"_id": ObjectId(record_id)})
        print(json.dumps(doc, default=str, indent=2) if doc else "No record found.")
    except Exception:
        print("Invalid ID. Copy the ID printed after a prediction is saved.")


def update_note():
    print("\nExample input: <record_id>, Needs follow-up call")
    data = input("Enter record ID and note, separated by a comma: ").split(",", 1)
    try:
        result = records.update_one(
            {"_id": ObjectId(data[0].strip())},
            {"$set": {"note": data[1].strip()}}
        )
        print("Updated successfully." if result.modified_count else "No record changed; check the ID or note.")
    except Exception:
        print("Invalid MongoDB ID.")


def delete_one():
    print("\nExample input: <record_id> (use a real ID from your records)")
    record_id = input("Enter MongoDB record ID to delete: ").strip()
    if input("Type YES to confirm deletion: ").strip() != "YES":
        print("Delete cancelled.")
        return
    try:
        result = records.delete_one({"_id": ObjectId(record_id)})
        print("Deleted successfully." if result.deleted_count else "No record found.")
    except Exception:
        print("Invalid MongoDB ID.")


def count_records():
    print("\nExample: No input required. Counts all, churn, and stay records.")
    print("Total:", records.count_documents({}))
    print("Churn:", records.count_documents({"prediction": 1}))
    print("Stay:", records.count_documents({"prediction": 0}))


def filter_by_age():
    print("\nExample input: 30 (shows customers older than 30)")
    try:
        age = int(input("Show customers older than age: "))
        docs = list(records.find({"age": {"$gt": age}}).sort("age", -1))
        print(json.dumps(docs, default=str, indent=2) if docs else "No matching records.")
    except ValueError:
        print("Enter age as a whole number.")


def sort_by_age():
    print("\nExample input: 1 for ascending, 2 for descending")
    choice = input("Choose sort order: ").strip()
    if choice not in ["1", "2"]:
        print("Enter 1 or 2.")
        return
    direction = 1 if choice == "1" else -1
    docs = list(records.find().sort("age", direction).limit(10))
    print(json.dumps(docs, default=str, indent=2) if docs else "No records found.")


def summary():
    print("\nExample: No input required. Groups records by prediction result.")
    docs = list(records.aggregate([
        {"$group": {
            "_id": "$result",
            "total": {"$sum": 1},
            "average_probability": {"$avg": "$probability"}
        }},
        {"$sort": {"total": -1}}
    ]))
    print(json.dumps(docs, default=str, indent=2) if docs else "No records to summarize.")


def menu():
    while True:
        print("\n====== Bank Churn MongoDB Tester ======")
        print("1. Predict customer and save to MongoDB")
        print("2. Show all recent records")
        print("3. Find one record by ID")
        print("4. Update a record note")
        print("5. Delete one record")
        print("6. Count records")
        print("7. Filter customers by age")
        print("8. Sort records by age")
        print("9. Aggregation summary")
        print("0. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            predict_customer()
        elif choice == "2":
            show_all()
        elif choice == "3":
            find_one()
        elif choice == "4":
            update_note()
        elif choice == "5":
            delete_one()
        elif choice == "6":
            count_records()
        elif choice == "7":
            filter_by_age()
        elif choice == "8":
            sort_by_age()
        elif choice == "9":
            summary()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Choose a number from 0 to 9.")


if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("MongoDB connected.")
        menu()
    except Exception as error:
        print("Could not connect to MongoDB. Check that it is running.")
        print("Details:", error)
    finally:
        client.close()
