from pymongo import MongoClient, DESCENDING
from bson.objectid import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["bank_churn_db"]
collection = db["prediction_history"]


# Create
def save_prediction(record):
    return collection.insert_one(record)


# Read
def get_prediction_history(name_filter=""):
    query = {}

    if name_filter:
        query["user_name"] = {"$regex": name_filter, "$options": "i"}

    return list(collection.find(query).sort("timestamp", DESCENDING))


# Delete one prediction
# Delete one prediction
def delete_prediction(doc_id):
    from bson.objectid import ObjectId

    result = collection.delete_one({"_id": ObjectId(doc_id)})
    return result.deleted_count > 0
    
# Delete all predictions
def clear_all_predictions():
    return collection.delete_many({}).deleted_count


# Analytics
def get_analytics_summary():
    total = collection.count_documents({})

    if total == 0:
        return {"total_count": 0, "prediction_breakdown": [],
                "geography_breakdown": [], "model_breakdown": []}

    predictions = list(collection.aggregate([
        {"$group": {
            "_id": "$prediction_label",
            "count": {"$sum": 1},
            "avg_probability": {"$avg": "$churn_probability"}
        }}
    ]))

    geography = list(collection.aggregate([
        {"$group": {
            "_id": "$geography",
            "total": {"$sum": 1},
            "avg_balance": {"$avg": "$balance"}
        }},
        {"$sort": {"total": -1}}
    ]))

    models = list(collection.aggregate([
        {"$group": {
            "_id": "$model_name",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]))

    return {
        "total_count": total,
        "prediction_breakdown": predictions,
        "geography_breakdown": geography,
        "model_breakdown": models
    }
