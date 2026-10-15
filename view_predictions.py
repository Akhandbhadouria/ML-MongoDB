"""
view_predictions.py
-------------------
CLI tool to inspect predictions stored in MongoDB directly from the terminal.

Usage:
    python view_predictions.py            # Show all records
    python view_predictions.py John       # Filter by name (case-insensitive)
"""

import sys
from datetime import datetime
from mongodb_client import get_prediction_history

def main():
    name_filter = sys.argv[1] if len(sys.argv) > 1 else ""

    print("=" * 70)
    print(" 🏦 MongoDB - Bank Churn Prediction Records")
    if name_filter:
        print(f" 🔍 Filtering by user name: '{name_filter}'")
    print("=" * 70)

    records = get_prediction_history(name_filter=name_filter)

    if not records:
        print("\n❌ No records found in MongoDB matching your query.\n")
        return

    print(f"\nTotal Records Found: {len(records)}\n")
    print(f"{'#':<3} | {'User Name':<16} | {'Prediction':<16} | {'Prob':<7} | {'Model':<14} | {'Timestamp'}")
    print("-" * 70)

    for i, doc in enumerate(records, 1):
        ts = doc.get("timestamp")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
        user = doc.get("user_name", "N/A")[:15]
        label = doc.get("prediction_label", "N/A")[:15]
        prob = f"{doc.get('churn_probability', 0.0)*100:.1f}%"
        model = doc.get("model_name", "N/A")[:13]

        print(f"{i:<3} | {user:<16} | {label:<16} | {prob:<7} | {model:<14} | {ts_str}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
